<div align="center">

# GETER

Official code repository 🗂️ for *ACL 2025 Findings* paper  
**"Towards Explainable Temporal Reasoning in Large Language Models: A Structure-Aware Generative Framework"**.

![image](https://github.com/user-attachments/assets/087a9702-cb87-491f-be16-c7284b09a055)

</div>

## Step 1: Environment Preparation
```sh
conda create -n geter python=3.10
conda activate geter

# install related libraries
pip install -r requirements.txt

```
## Step 2: Data Preparation
To prepare the data, start by using a temporal encoder to generate embeddings for entities and relations (we used the RE-GCN model for this purpose). Next, run the script `./get_Graph.py` to obtain the graph embeddings. Finally, execute `./convert_instruction.py` to construct the training and validation instruction datasets. For assistance with generating embeddings for entities and relations, refer to this resource: https://github.com/LiaoMengqi/KGMH.


## Step 3: Training 
```sh
torchrun --nproc_per_node 2 --master_port 29500 lora_GETER.py \
    --ddp_timeout 180000000 \
    --model_name_or_path ${model_name_or_path} \
    --model_type ${llama/mistral/qwen}\
    --use_lora \
    --train_graph_emb_path ${train_graph_path}\
    --test_graph_emb_path ${test_graph_path}\
    --deepspeed configs/deepspeed_config_stage3.json \
    --lora_config configs/lora_config_llama.json \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 3 \
    --model_max_length ${cutoff_len} \
    --save_strategy "steps" \
    --save_total_limit 2 \
    --learning_rate 3e-4 \
    --weight_decay 0.00001 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --evaluation_strategy "steps" \
    --torch_dtype "bfloat16" \
    --bf16 \
    --seed 1234 \
    --gradient_checkpointing \
    --output_dir ${output_dir}
```
