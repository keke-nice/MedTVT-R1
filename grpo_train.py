# Copyright 2025 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk

# from math_verify import parse, verify

from vllm_grpo_trainer import llama3GRPOVLLMTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

import json
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms
from datasets import Dataset
from typing import List, Dict, Any
import wfdb
from scipy.signal import resample
from llama.Cardio_llama import Cardio_LLaMA
import argparse
import pandas as pd
from data.process_and_save_tensors import get_cxr, get_ecg, get_lab

@dataclass
class MyArguments:
    """
    Arguments for the ImageBind-LLM pre-training script.

    Args:
        batch_size (`int`): Batch size per GPU.
        llama_type (`str`): Type of LLaMA model.
        llama_path (`str`): Path to LLaMA pretrained checkpoint.
        vit_path (`str`): Path to ViT pretrained checkpoint.
        max_words (`int`): Max number of input words.
        model_ckpt (`str`): Path to the pretrained checkpoint.
    """

    batch_size: int = field(
        default=64,
        metadata={"help": "Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus)"},
    )
    llama_type: str = field(
        default="llama3",
        metadata={"help": "Type of LLaMA model"},
    )
    llama_path: str = field(
        default="/path/to/llama",
        metadata={"help": "Path to LLaMA pretrained checkpoint"},
    )
    vit_path: str = field(
        default="google/vit-base-patch16-224",
        metadata={"help": "Path to ViT pretrained checkpoint"},
    )
    max_words: int = field(
        default=512,
        metadata={"help": "Max number of input words"},
    )
    model_ckpt: str = field(
        default="./ckpts/all/ep8/checkpoint.pth",
        metadata={"help": "Path to the pretrained checkpoint"},
    )


# 标签映射表
label_mapping = {
    "E785": "Hyperlipidemia, Pure Hypercholesterolemia",
    "I2510": "Atherosclerotic Heart Disease Without Angina Pectoris",
    "N179": "Acute Kidney Failure, Unspecified",
    "I10": "Essential (Primary) Hypertension",
    "I4891": "Atrial Fibrillation",
    "E039": "Hypothyroidism, Unspecified",
    "I5033": "Chronic Systolic Heart Failure",
    "J189": "Pneumonia, Unspecified Organism",
    "J449": "Chronic Obstructive Pulmonary Disease (COPD), Unspecified",
    "I214": "Non-ST-Elevation Myocardial Infarction (NSTEMI)",
    "E119": "Type 2 diabetes mellitus without complications",
    "A419": "Sepsis, unspecified organism",
    "Other": "The conditions such as Hyperlipidemia, Atherosclerotic Heart Disease, Acute Kidney Failure, Essential Hypertension, Atrial Fibrillation, Hypothyroidism, Chronic Systolic Heart Failure, Pneumonia, COPD, NSTEMI, Type 2 Diabetes, and Sepsis were not identified. Please consider other diagnoses."
}

# 奖励函数
def Jaccard_reward(completions, ground_truths):
    """
    Compare model-generated completions to ground truth answers and calculate rewards.
    
    Args:
        completions (list): List of model-generated completions.
        ground_truths (list): List of ground truth answers.
    
    Returns:
        List of rewards for each completion.
    """
    rewards = []
    
    pattern = r"<answer>(.*?)</answer>"
    
    for completion, ground_truth in zip(completions, ground_truths):
        completion_match = re.search(pattern, completion, re.DOTALL)
        ground_truth_match = re.search(pattern, ground_truth, re.DOTALL)
        
        if completion_match and ground_truth_match:
            completion_answer = completion_match.group(1).strip()
            ground_truth_answer = ground_truth_match.group(1).strip()
            
            completion_labels = set(completion_answer.split(";"))
            ground_truth_labels = set(ground_truth_answer.split(";"))
            
            completion_labels = {label.strip() for label in completion_labels if label.strip()}
            ground_truth_labels = {label.strip() for label in ground_truth_labels if label.strip()}

            intersection = len(completion_labels & ground_truth_labels)
            union = len(completion_labels | ground_truth_labels)
            jaccard_similarity = intersection / union if union > 0 else 0.0
            rewards.append(jaccard_similarity)
        else:
            rewards.append(0.0)
    
    return rewards

def format_reward(completions, solution):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion.lstrip() for completion in completions]
    # matches = [re.match(pattern, content) for content in completion_contents]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

reward_funcs_registry = {
    "Jaccard": Jaccard_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def get_ecg(ecg_path):
    """
    Loads and preprocesses an ECG signal.
    """
    # signal = torch.from_numpy(wfdb.rdrecord(ecg_path).p_signal)

    # # normalize to be between -1 and 1
    # signal = 2 * (signal - signal.min()) / (signal.max() - signal.min()) - 1

    # return signal.unsqueeze(0).to(torch.float32)
    ecg_signal = wfdb.rdsamp(ecg_path)  # files/p1205/p12054137/s40989841/40989841
    ecg = ecg_signal[0].T
    ecg = torch.from_numpy(resample(ecg, int(ecg.shape[1] / 5), axis=1))
    if torch.any(torch.isnan(ecg)):
        isnan = torch.isnan(ecg)
        ecg = torch.where(isnan, torch.zeros_like(ecg), ecg)
    return ecg.to(torch.float32)

def load_all3qa_as_hf_dataset(
    json_path: str,
    root_path: str,
    tokenizer,
    max_words: int = 512,
    llama_type: str = "llama3",
    data_dir: str = '/remote-home/hao.lu/Code/Yuting/ECG/ECG_pretrain/Dataset/symile-mimic'
) -> Dataset:
    print('Loading ALL3QA dataset...')

    question_list = []
    answer_list = []
    ecg_path_list = []

    with open(json_path, 'r', encoding='utf-8') as f:
        res = json.load(f)

    for instance in tqdm(res, total=len(res)):
        user_message = instance['messages'][0]['content']
        assistant_message = instance['messages'][1]['content']
        question_list.append(user_message)
        answer_list.append(assistant_message)
        ecg_path_list.append(os.path.join(root_path, instance['ECG'][0]))

    # Load static tensors
    # cxr_tensors = torch.tensor(np.load(os.path.join(data_dir, "data_npy/train/cxr_train.npy")))
    # lab_percentiles_tensors = torch.tensor(np.load(os.path.join(data_dir, "data_npy/train/labs_percentiles_train.npy")))
    # labs_missingness_tensors = torch.tensor(np.load(os.path.join(data_dir, "data_npy/train/labs_missingness_train.npy")))
    # lab_tensors = torch.cat([lab_percentiles_tensors, labs_missingness_tensors], dim=1)
    ecg_root_path = '/remote-home/hao.lu/Code/Yuting/ECG/ECG_pretrain/Dataset/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0'
    cxr_root_path = '/remote-home/hao.lu/Code/Yuting/ECG/ECG_pretrain/Dataset/mimic-cxr-jpg-chest-radiographs-with-structured-labels-2.1.0'
    train_df = pd.read_csv("/remote-home/hao.lu/Code/Yuting/ECG/ECG_pretrain/Dataset/symile-mimic/train_filtered_7_catagory.csv")
    ecg_paths = train_df['ecg_path']
    cxr_paths = train_df['cxr_path']
    assert len(ecg_path_list)==len(ecg_paths)

    # Pack metadata into dict list
    raw_data = []
    for i in range(len(question_list)):
    # for i in range(8):
        raw_data.append({
            'question': question_list[i],
            'answer': answer_list[i],
            'ecg_path': ecg_path_list[i],
            'index': i  # index to access cxr/lab tensors
        })

    hf_dataset = Dataset.from_list(raw_data)

    # resize = transforms.Resize((224, 224))

    def preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
        idx = example['index']
        # ecg = get_ecg(example['ecg_path'])
        # cxr = resize(cxr_tensors[idx].unsqueeze(0)).squeeze(0)
        # lab = lab_tensors[idx]
        ecg = get_ecg(os.path.join(ecg_root_path, ecg_paths.iloc[idx]))
        cxr = get_cxr(os.path.join(cxr_root_path, cxr_paths.iloc[idx]))
        lab = get_lab(train_df.iloc[idx])

        input1 = example['question']
        answer = example['answer']
        input2 = input1 + " " + answer

        if llama_type == 'llama3':
            input1_tensor = torch.tensor(tokenizer.encode(input1, bos=True, eos=False, allowed_special={"<ecg>","<cxr>","<lab>"}), dtype=torch.int64)
            input2_tensor = torch.tensor(tokenizer.encode(input2, bos=True, eos=True, allowed_special={"<ecg>","<cxr>","<lab>"}), dtype=torch.int64)
        elif llama_type == '7B':
            input1_tensor = torch.tensor(tokenizer(input1).input_ids, dtype=torch.int64)
            input2_tensor = torch.tensor(tokenizer(input2).input_ids, dtype=torch.int64)
        else:
            raise ValueError("Unsupported llama_type")

        padding = max_words - input2_tensor.shape[0]
        if padding > 0:
            input2_tensor = F.pad(input2_tensor, (0, padding), "constant", tokenizer.pad_id)
        elif padding < 0:
            input2_tensor = input2_tensor[:max_words]

        labels = input2_tensor.clone()
        labels[:len(input1_tensor)] = -1
        completion_ids = input2_tensor[len(input1_tensor):]

        input2_mask = input2_tensor.ge(0)
        label_mask = labels.ge(0)
        input2_tensor[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()

        return {
            'input_ids': input2_tensor,
            'labels': labels,
            'input_mask': input2_mask,
            'question': input1,
            "answer": answer,
            "prompt_ids": input1_tensor,
            'ecg': ecg,
            'cxr': cxr,
            'lab': lab,
        }

    print('Applying preprocessing with .map() ...')
    hf_dataset = hf_dataset.map(preprocess)
    return hf_dataset


def main(training_args):
    # Get reward functions
    # import pdb; pdb.set_trace()
    reward_funcs = ['Jaccard','format']
    reward_funcs = [reward_funcs_registry[func] for func in reward_funcs]
    # import pdb; pdb.set_trace()

    llama_type = training_args.llama_type
    if llama_type == 'llama3':
        training_args.llama_ckpt_dir = os.path.join(training_args.llama_path)
        training_args.llama_tokenzier_path = os.path.join(training_args.llama_path,'tokenizer.model')
        
    model = Cardio_LLaMA(training_args.llama_ckpt_dir, training_args.llama_tokenzier_path, training_args, stage=3, load_llama=False)
    print(model)
    print("Loading Model Checkpoint")
    checkpoint = torch.load(training_args.model_ckpt, map_location='cpu')


    new_ckpt = {}
    for key, value in checkpoint['model'].items():
        key = key.replace("module.", "")
        new_ckpt[key] = value
    del checkpoint
    load_result = model.load_state_dict(new_ckpt, strict=False)
    model.to("cuda")
    tokenizer = model.tokenizer


    # Load the dataset
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    ### lzy modified
    dataset = load_all3qa_as_hf_dataset(json_path='/remote-home/hao.lu/Code/Yuting/ECG/ECG_pretrain/Dataset/symile-mimic/QA/train_dig_qa_dataset_gpt_7_category_refine_add_spec_token.json',
                                  root_path='/remote-home/hao.lu/Code/Yuting/ECG/ECG_pretrain/Dataset/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0',
                                  tokenizer=model.tokenizer, max_words=600, llama_type="llama3")

    
    trainer_cls = llama3GRPOVLLMTrainer
    print("using: ", trainer_cls)


    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    # if training_args.push_to_hub:
    #     trainer.push_to_hub("GRPO_llama3.2-1B")


if __name__ == "__main__":
    parser = TrlParser((GRPOConfig, MyArguments))
    training_args, args= parser.parse_args_and_config()
    for key, value in vars(args).items():
      setattr(training_args, key, value)
    main(training_args)
