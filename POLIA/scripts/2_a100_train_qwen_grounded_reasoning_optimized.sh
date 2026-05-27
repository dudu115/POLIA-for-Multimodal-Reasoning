#!/bin/bash

setting='a100_2gpu_qwen_polia_think_rethink_optimized_zero3'
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_PROJECT=$setting


OUTPUT_DIR="$setting"

LOCAL_MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"

python -m accelerate.commands.launch \
    --config_file ./accelerate_configs/deepspeed_zero3.yaml \
    --main_process_port 20092 \
    polia/POLIA.py \
    --train_data_path ./data/vsr_cot_train_10.jsonl,./data/tallyqa_train_10.jsonl \
    --train_image_folder_path ./data/vsr,./data/tallyqa \
    --eval_data_path ./data/vsr_val.jsonl,./data/tallyqa_val.jsonl \
    --eval_image_folder_path ./data/vsr,./data/tallyqa \
    --setting $setting \
    --max_turns 1 \
    --output_dir $OUTPUT_DIR \
    --project_root_path $(pwd) \
    --python_path_for_dino $(which python) \
    --dataset_name rr \
    --learning_rate 2e-6 \
    --num_generations 16 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 16 \
    --max_eval_samples_per_dataset 500 \
    --generation_batch_size 16 \
    --torch_empty_cache_steps 1 \
    --gradient_checkpointing True \
    --eval_strategy steps \
    --eval_steps 10 \
    --save_steps 10 \
    --save_total_limit 1 \
    --num_train_epochs 20 \
    --max_completion_length 1500 \
    --max_prompt_length 500 \
    --bf16 True \
    --bf16_full_eval True \
    --report_to wandb \
    --run_name polia \
    --logging_steps 1 \
    --beta 0.05 \
    --iou_weight 0.7 \
    --l1_weight 0.3 \
    --subgraph_advantage_weight 1.0 \
    --gpt_reward_weight 1.5 \
    --default_reward_weight 1.0 \
    --gpt_binary_reward False \
    --log_completions True \
    --model_name_or_path "$MODEL_PATH"

