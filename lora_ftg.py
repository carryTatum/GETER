import json
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    LlamaConfig,
    AutoConfig,
    Qwen2Config,
    LlamaForCausalLM,
    LlamaTokenizer,
    Qwen2Tokenizer,
    TrainingArguments,
    set_seed,
    Trainer,
)
from llm_arch.FtGForCausalLM import FtGForCausalLM
from llm_arch.FtGForCausalLM_mistral import FtGForCausalLM_mistral
from llm_arch.FtGForCausalLM_qwen import FtGForCausalLM_qwen
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.runtime.engine import DeepSpeedEngine
from functools import partial
import math
from multiprocessing import cpu_count
from prompter import (
    batch_grouped_sft_generate,
    generate_and_tokenize_prompt_with_graph,
)
from datasets import load_dataset
import transformers
from transformers.trainer_pt_utils import torch_distributed_zero_first
from transformers.trainer_utils import get_last_checkpoint
import sys
import logging
from dataclasses import dataclass, field
from typing import Optional, Union
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)
from transformers.utils import add_start_docstrings
from ftg_trainer import FtGTrainer


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return

def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def maybe_zero_3(param, ignore_status=False, name=None):
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={"help": "torch dtype"}
    )
    use_flash_attention: bool = field(
        default=False,
        metadata={"help": "Whether to use flash attention"}
    )

    model_type: str = field(
        default="llama",
        metadata={"help": "Model type to use. Options are 'llama', 'mistral', 'qwen'"}
    )

    # llama: bool = field(default=True, metadata={"help": "Whether to use llama"})
    # mistral: bool = field(default=False, metadata={"help": "Whether to use mistral"})
    # qwen: bool = field(default=False, metadata={"help": "Whether to use qwen"})

    pretrain_mm_mlp_adapter: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrain mm mlp adapter"}
    )
    graph_hidden_dim: Optional[int] = field(
        default=1536,
        metadata={"help": "Graph hidden dim"}
    )
    mm_projector_type: Optional[str] = field(
        default="linear",
        metadata={"help": "mm projector type"}
    )

@dataclass
class DataArguments:
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to training file"}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to validation file"}
    )
    group_sample: bool = field(
        default=False,
        metadata={"help": "Whether to group sample"}
    )
    train_graph_emb_path: Optional[str] = field(
        default="./data/ICEWS14/regcn/train_graph_emb.pt",
        metadata={"help": "Path to train graph emb"}
    )
    test_graph_emb_path: Optional[str] = field(
        default="./data/ICEWS14/regcn/test_graph_emb.pt",
        metadata={"help": "Path to test graph emb"}
    )


@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class TrainingArguments(TrainingArguments):
    model_max_length: int = field(
        default=2048,
        metadata={"help": "The maximum length of the model's input sequence"}
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use lora"}
    )
    use_int8_training: bool = field(
        default=False,
        metadata={"help": "Whether to use int8 training"}
    )
    lora_config: Optional[str] = field(
        default=None,
        metadata={"help": "Path to lora config"}
    )
    ddp_find_unused_parameters: bool = field(
        default=False,
        metadata={"help": "Whether to use ddp find unused parameters"}
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Whether to use gradient checkpointing"}
    )
    evaluation_strategy: str = field(
        default="steps",
        metadata={"help": "evaluation strategy"}
    )
    save_total_limit: Optional[int] = field(
        default=3,
        metadata={"help": "save total limit"}
    )
    load_best_model_at_end: bool = field(
        default=True,
        metadata={"help": "load best model at end"}
    )
    report_to: str = field(
        default=None,
        metadata={"help": "report to"}
    )
    deepspeed: str = field(
        default=None,
        metadata={"help": "deepspeed, please pass the path to deepspeed json config file, e.g., ds_config.json"}
    )
    do_train: bool = field(
        default=True,
        metadata={"help": "Whether to run training."}
    )
    tune_mm_mlp_adapter: bool = field(
        default=True,
        metadata={"help": "Whether to tune mm mlp adapter"}
    )
    lora_bias: str = field(
        default="none",
        metadata={"help": "lora bias"}
    )
    # remove_unused_columns: bool = field(
    #     default=False,
    #     metadata={"help": "Whether to remove unused columns"}
    # )

def print_rank_0(msg, log_file, rank=0):
    if rank <= 0:
        with open(log_file, "a") as f:
            print(msg)
            f.write(msg + "\n")


def get_model_param_count(
    model: Union[DeepSpeedEngine, torch.nn.Module], trainable_only=False
):
    """
    Calculate model's total param count. If trainable_only is True then count only those requiring grads
    """
    if is_deepspeed_zero3_enabled() and isinstance(model, DeepSpeedEngine):

        def numel(p):
            return p.ds_numel

    else:

        def numel(p):
            return p.numel()

    return sum(
        numel(p) for p in model.parameters() if not trainable_only or p.requires_grad
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_rank = torch.distributed.get_rank()
    # global_rank = int(os.environ.get("LOCAL_RANK", 0))
    log_file = os.path.join(training_args.output_dir, "log.txt")
    # setup logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, distributed training: {bool(training_args.local_rank != -1)}, fp16-bits training: {training_args.fp16}, bf16-bits training: {training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    
    # Set seed before initializing model.
    set_seed(training_args.seed)
    training_args._frozen = False
    training_args.data_seed = training_args.seed

    # if model_args.llama:
    if model_args.model_type == "llama":
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        print_rank_0(
            "Set the eos_token_id and bos_token_id of model tokenizer",
            log_file,
            global_rank
        )
        tokenizer.add_special_tokens(
            {
                "pad_token": tokenizer.eos_token,
            }
        )

    elif model_args.model_type == "mistral":
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path , use_fast=True)
        print_rank_0(
            "Set the eos_token_id and bos_token_id of model tokenizer",
            log_file,
            global_rank
        )
        tokenizer.add_special_tokens(
            {
                "pad_token": tokenizer.eos_token,
            }
        )

    elif model_args.model_type == "qwen":
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path , use_fast=True)
        print_rank_0(
            "Set the eos_token_id and bos_token_id of model tokenizer",
            log_file,
            global_rank
        )

    tokenizer.padding_side = "right"
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ['auto', None]
        else getattr(torch, model_args.torch_dtype)
    )

    # int8 is not compatible with DeepSpeed (require not to pass device_map)
    if training_args.use_int8_training:
        print_rank_0("int8 is not compatible with DeepSpeed", log_file, global_rank)
        device_map = (
            {"": int(os.environ.get("LOCAL_RANK") or 0)} if world_size != 1 else "auto"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            load_in_8bit=True,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
    else:
        if model_args.model_type == "llama":
            config = AutoConfig.from_pretrained(model_args.model_name_or_path)

            # config.vocab_size = tokenizer.vocab_size
            config.pad_token_id = tokenizer.pad_token_id
            config.unk_token_id = tokenizer.unk_token_id
            config._flash_attn_2_enabled = model_args.use_flash_attention
            model = FtGForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                torch_dtype=torch_dtype,
            )
            print(len(tokenizer))
            # we added a new padding token to the tokenizer, we have to extend the embddings
            model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

        elif model_args.model_type == "mistral":
            config = AutoConfig.from_pretrained(model_args.model_name_or_path)

            # config.vocab_size = tokenizer.vocab_size
            config.pad_token_id = tokenizer.pad_token_id
            config.unk_token_id = tokenizer.unk_token_id
            config._flash_attn_2_enabled = model_args.use_flash_attention
            model = FtGForCausalLM_mistral.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                torch_dtype=torch_dtype,
            )
            print(len(tokenizer))
            # we added a new padding token to the tokenizer, we have to extend the embddings
            model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
        elif model_args.model_type == "qwen":
            config = AutoConfig.from_pretrained(model_args.model_name_or_path)
            model = FtGForCausalLM_qwen.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                torch_dtype=torch_dtype,
            )
        else:
            raise NotImplementedError


    print_rank_0(
        "tokenizer.eos_token_id = {}".format(tokenizer.eos_token_id),
        log_file,
        global_rank,
    )
    print_rank_0(
        "tokenizer.pad_token_id = {}".format(tokenizer.pad_token_id),
        log_file,
        global_rank,
    )
    print_rank_0(
        "tokenizer.bos_token_id = {}".format(tokenizer.bos_token_id),
        log_file,
        global_rank,
    )

    # Set peft model
    if training_args.use_lora:
        print_rank_0(
            "Loading lora config from {}".format(training_args.lora_config),
            log_file,
            global_rank,
        )
        lora_config = json.load(open(training_args.lora_config))
        print_rank_0("Lora config: {}".format(lora_config), log_file, global_rank)
        if training_args.use_int8_training:
            print_rank_0(
                "training_args.use_int8_training !!! (int8 is not compatible with DeepSpeed)",
                log_file,
                global_rank
            )
            model = prepare_model_for_kbit_training(model)
        config = LoraConfig(
            r=lora_config["lora_r"],
            lora_alpha=lora_config["lora_alpha"],
            target_modules=lora_config["lora_target_modules"],
            lora_dropout=lora_config["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        if hasattr(model, "enable_input_requires_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        model = get_peft_model(model, config)
        if hasattr(model_args, 'graph_hidden_dim'):
            model.get_model().initialize_graph_modules(model_args)
        model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        print_rank_0("Use gradient checkpointing", log_file, global_rank)
        model.gradient_checkpointing_enable()

    assert os.path.exists(data_args.train_file), f"{data_args.train_file} file not exists"

    with torch_distributed_zero_first(global_rank):
        train_data = load_dataset(
            "json", data_files=data_args.train_file, cache_dir=model_args.cache_dir
        )
        val_data = load_dataset(
            "json", data_files=data_args.validation_file, cache_dir=model_args.cache_dir
        )
        train_graph_emb = torch.load(data_args.train_graph_emb_path)
        test_graph_emb = torch.load(data_args.test_graph_emb_path)
        if data_args.group_sample:
            train_data = (
                train_data['train'].shuffle().map(
                    partial(
                        batch_grouped_sft_generate,
                        training_args.model_max_length,
                        tokenizer,
                    ),
                    batched=True,
                    desc=f"Grouping texts in chunks of {training_args.model_max_length}",
                    remove_columns=["id", "conversations"],
                    num_proc=cpu_count(),
                )
            )
            val_data = val_data['train'].map(
                partial(
                    batch_grouped_sft_generate,
                    training_args.model_max_length,
                    tokenizer,
                ),
                batched=True,
                desc=f"Grouping texts in chunks of {training_args.model_max_length}",
                remove_columns=["id", "conversations"],
                num_proc=cpu_count(),
            )
        else:
            train_data = (
                train_data['train'].shuffle().map(
                    partial(
                        generate_and_tokenize_prompt_with_graph,
                        training_args.model_max_length,
                        tokenizer,
                        padding_side="right",
                        train_graph_emb=train_graph_emb,
                    ),
                    num_proc=cpu_count(),
                )
            )
            val_data = val_data['train'].map(
                partial(
                    generate_and_tokenize_prompt_with_graph,
                    training_args.model_max_length,
                    tokenizer,
                    padding_side="right",
                    train_graph_emb=test_graph_emb,
                ),
                num_proc=cpu_count(),
            )

    for i in range(2):
        print_rank_0(
            "Eval tokenized example: {}".format(val_data[i]), log_file, global_rank
        )
    for i in range(2):
        print_rank_0(
            "Train tokenized example: {}".format(train_data[i]), log_file, global_rank
        )

    training_nums = len(train_data)
    num_gpus = torch.cuda.device_count()

    batch_size = (
        training_args.per_device_train_batch_size
        * training_args.world_size
        * training_args.gradient_accumulation_steps
    )
    # train steps
    t_total = math.ceil(training_nums / batch_size) * training_args.num_train_epochs
    # eval steps
    training_args.eval_steps = max(t_total // (training_args.num_train_epochs * 4), 5)
    # save steps
    training_args.save_steps = training_args.eval_steps
    training_args.warmup_steps = (
        int(t_total * training_args.warmup_ratio)
        if training_args.warmup_ratio > 0.0
        else training_args.warmup_steps
    )
    print_rank_0(
        "num_gpus = {}, training_nums = {}, t_total = {}, warmup_steps = {}, eval_steps = {}, save_steps = {}".format(
            num_gpus,
            training_nums,
            t_total,
            training_args.warmup_steps,
            training_args.eval_steps,
            training_args.save_steps,
        ),
        log_file,
        global_rank,
    )
    print_rank_0(
        "val data nums = {}, training_nums = {}, batch_size = {}".format(
            len(val_data), training_nums, batch_size
        ),
        log_file,
        global_rank,
    )

    trainer = FtGTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    print_rank_0(
        f"Using {training_args.half_precision_backend} half precision backend", log_file, global_rank
    )
    # Train !
    len_dataloader = len(trainer.get_train_dataloader())
    num_update_steps_per_epoch = math.ceil(len_dataloader / training_args.gradient_accumulation_steps)
    total_train_batch_size = (
        training_args.train_batch_size
        * training_args.gradient_accumulation_steps
        * training_args.world_size
    )
    num_examples = trainer.num_examples(trainer.get_train_dataloader())
    num_train_samples = num_examples * training_args.num_train_epochs
    max_steps = math.ceil(training_args.num_train_epochs * num_update_steps_per_epoch)
    print_rank_0(f"  Num examples = {num_examples}", log_file, global_rank)
    print_rank_0(f"  Num train samples = {num_train_samples}", log_file, global_rank)
    print_rank_0(f"  world_size = {world_size}", log_file, global_rank)
    print_rank_0(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}",
        log_file,
        global_rank,
    )
    print_rank_0(
        f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}",
        log_file,
        global_rank,
    )
    print_rank_0(f"  Total optimization steps = {max_steps}", log_file, global_rank)
    print_rank_0(
        f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True)}",
        log_file,
        global_rank,
    )

    model.config.use_cache = False
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(training_args.output_dir)
    trainer.save_state()
    state_dict = get_peft_state_maybe_zero_3(
        model.named_parameters(), training_args.lora_bias
    )
    non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
        model.named_parameters()
    )
    if training_args.local_rank == 0 or training_args.local_rank == -1:
        model.config.save_pretrained(training_args.output_dir)
        model.save_pretrained(training_args.output_dir, state_dict=state_dict)
        torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'mm_projector_final.bin'))

    print_rank_0("Training done", log_file, global_rank)

#GETER main
if __name__ == "__main__":
     main()
    
