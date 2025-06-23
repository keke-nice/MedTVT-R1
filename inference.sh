export CUDA_VISIBLE_DEVICES=3
python inference.py --model ./ckpts/GRPO_8331_raw_data_llama3.2_1B_2GPU/checkpoint-500/pytorch_model.bin \
--json_path ./QA/test_dig_qa_dataset_gpt_7_category_add_spec_token.json \
--output_path ./results/infer_GRPO500_retrain_llama3.2_1B.json \
--llama_type llama3 \
--llama_dir ./CKPTS/LLaMA3.2-1B-Instruct \
--train_type GRPO \
--model_type cardio_llama \
--add_special_token

