#!/usr/bin/bash
export CUDA_VISIBLE_DEVICES=3,5
export TORCH_DISTRIBUTED_DEBUG=INFO

python3 -u -m torch.distributed.launch --master_port=1117 --nproc_per_node=2 --use_env \
 main_train.py --batch_size 8 --accum_iter 1 --stage 1 \
 --epochs 20 --split_epoch 1 --warmup_epochs 0 --lr 1e-4 --min_lr 1e-6 --weight_decay 0.05 \
 --vit_path google/vit-base-patch16-224 \
 --llama_path "./CKPTS/LLaMA3.2-1B-Instruct" \
 --output_dir "./ckpts/pretrain_llama3.2_1B_8331_raw_data" --max_words 600 --llama_type llama3  \
 --model_type cardio_llama --add_special_token \


