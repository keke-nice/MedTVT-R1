
export CKPT_PATH=./CKPTS/LLaMA3.2-1B-Instruct
export SAVE_PATH=./ckpts/GRPO_8331_raw_data_llama3.2_1B_fusion
export CUDA_VISIBLE_DEVICES=6,7



torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    grpo_train.py \
    --output_dir ${SAVE_PATH}  \
    --llama_path ${CKPT_PATH} \
    --model_ckpt ./ckpts/retrain_llama3.2_1B_8331_raw_data_Dyna_Weig_cycle_fusion/checkpoint_19.pth \
    --vit_path  ./CKPTS/vit-base-patch16-224 \
    --batch_size 8 \
    --deepspeed ./util/zero2.json \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --report_to wandb \
    --gradient_checkpointing false \
    --num_train_epochs 1 \
    --run_name llama3-1B_GRPO \
    --save_steps 500 \
    --save_only_model True \
    --save_safetensors False \
    --num_generations 8 \
    --max_prompt_length=256 \
    --max_completion_length=600 \
    --log_on_each_node=False \
   
