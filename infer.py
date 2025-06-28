import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from peft import PeftModel
import argparse
import json
import os

# Import your own model architectures
from llm_arch.FtGForCausalLM import FtGForCausalLM
from llm_arch.FtGForCausalLM_qwen import FtGForCausalLM_qwen
from llm_arch.FtGForCausalLM_mistral import FtGForCausalLM_mistral

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Set the GPU device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to main model weights, e.g., ./models/Mistral-7B-Instruct-v0.3')
    parser.add_argument('--lora_path', type=str, required=True, help='LoRA fine-tuned weights folder, e.g., ./output/ICEWS14/regcn/mistral')
    parser.add_argument('--use_lora', action='store_true', help='Whether to load LoRA weights')
    parser.add_argument('--infer_file', type=str, required=True, help='Input file for inference (json), e.g., ./data/ICEWS14/ICEWS14_test.json')
    parser.add_argument('--graph_emb_file', type=str, required=True, help='Graph embedding pt file, e.g., ./data/ICEWS14/regcn/test_graph_emb.pt')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save inference results, e.g., ./output/ICEWS14/regcn/graph_mistral.json')
    parser.add_argument('--model_name', type=str, required=True, help='Model name, e.g., mistral/llama/qwen')
    parser.add_argument('--max_new_tokens', type=int, default=512, help='Maximum number of tokens to generate')
    args = parser.parse_args()

    answer = []

    # Load inference data
    with open(args.infer_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # Load graph embeddings
    test_graph_emb = torch.load(args.graph_emb_file)
    load_type = torch.float16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)

    # Select model class based on model_name
    if args.model_name == "llama":
        ModelClass = FtGForCausalLM
    elif args.model_name == "mistral":
        ModelClass = FtGForCausalLM_mistral
    elif args.model_name == "qwen":
        ModelClass = FtGForCausalLM_qwen
    else:
        raise ValueError(f"Unknown model_name: {args.model_name}")

    # LoRA weights loading
    if args.use_lora:
        config_path = os.path.join(args.lora_path, "config.json")
        model_config = AutoConfig.from_pretrained(args.lora_path) if os.path.exists(config_path) else None
        model = ModelClass.from_pretrained(args.ckpt_path, low_cpu_mem_usage=False, config=model_config)
        model.to(load_type)

        # Load additional FtG weights if present
        mm_proj_path = os.path.join(args.lora_path, 'mm_projector_final.bin')
        if os.path.exists(mm_proj_path):
            non_lora = torch.load(mm_proj_path, map_location='cpu')
            non_lora = {k: v.to(torch.float16) for k, v in non_lora.items()}
            non_lora = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora.items()}
            if any(k.startswith('model.model.') for k in non_lora):
                non_lora = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora.items()}
            model.load_state_dict(non_lora, strict=False)

        # Merge LoRA weights
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()
    else:
        model = ModelClass.from_pretrained(args.ckpt_path, low_cpu_mem_usage=False)

    model.to(device)
    model.eval()
    print('Model loaded.')

    # Inference loop
    for item in tqdm(test_data, desc='Inference'):
        messages = [
            {"role": "system", "content": item['instruction']},
            {"role": "user", "content": item['input']},
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors='pt', truncation=True, add_special_tokens=False)

        input_ids = inputs["input_ids"][0].unsqueeze(0).to(device)
        attention_mask = inputs["attention_mask"].to(device)
        graph_emb = test_graph_emb[item['index']].unsqueeze(0).unsqueeze(0).to(device).to(load_type)
        graph = torch.FloatTensor([item['index']]).to(device)

        temp_json = {
            'idx': item['index'],
            'label': item['label'],
            'target_output': item['output']
        }

        with torch.inference_mode():
            generation_output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                graph_emb=graph_emb,
                graph=graph,
                max_new_tokens=args.max_new_tokens,
                temperature=0.1,
            )

        input_token_len = input_ids.shape[1]
        generation_text = tokenizer.batch_decode(generation_output[:, input_token_len:], skip_special_tokens=True)[0]
        temp_json['prediction'] = generation_text
        answer.append(temp_json)

    # Save inference results
    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(answer, f, ensure_ascii=False, indent=2)
    print(f"Inference results saved to: {args.output_file}")

if __name__ == '__main__':
    main()