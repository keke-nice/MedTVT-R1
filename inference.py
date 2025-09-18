import torch.cuda

import tempfile
from PIL import Image
import scipy
import argparse

from llama.Cardio_llama import Cardio_LLaMA
import llama
import numpy as np
import os
import torch
import torchaudio
import torchvision.transforms as transforms
import av
import subprocess
import librosa
import json
from tqdm import tqdm
import pandas as pd

from data.process_and_save_tensors import get_ecg, get_cxr, get_lab

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="./ckpts/checkpoint.pth", type=str,help="Name of or path to pretrained checkpoint",)
parser.add_argument("--llama_type", default="8B", type=str,help="Type of llama original weight",)
parser.add_argument("--train_type", default="GRPO", type=str,help="Type of training strategy",)
parser.add_argument("--model_type", default="cardio_llama", type=str,help="Type of sft model",)
parser.add_argument("--llama_dir", default="./CKPTS/LLaMA3.2-1B-Instruct", type=str, help="Path to LLaMA pretrained checkpoint",)
# Input Arguments
parser.add_argument("--json_path", default="./QA/test_dig_qa_dataset_gpt.json", type=str,help="QA path",)
parser.add_argument("--output_path", default="./results/infer_ep8_test_dig.json", type=str,help="QA path",)
parser.add_argument('--vit_path', default="google/vit-base-patch16-224", type=str,help='path to ViT pretrained checkpoint')
parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
parser.add_argument('--add_special_token', action='store_true')
args = parser.parse_args()

llama_type = args.llama_type

if llama_type == '8B' or llama_type == 'llama3':
    llama_ckpt_dir = os.path.join(args.llama_dir)
    llama_tokenzier_path = os.path.join(args.llama_dir,'tokenizer.model')   

elif llama_type == '7B':
    llama_ckpt_dir = os.path.join(args.llama_dir, llama_type)
    llama_tokenzier_path = args.llama_dir

if args.model_type == 'cardio_llama':
    model = Cardio_LLaMA(llama_ckpt_dir, llama_tokenzier_path, args, stage=3, load_llama=False)

print("Loading Model Checkpoint")
checkpoint = torch.load(args.model, map_location='cpu')

new_ckpt = {}
if args.train_type=='GRPO':
  for key, value in checkpoint.items():
      key = key.replace("module.", "")
      new_ckpt[key] = value
elif args.train_type=='sft':
    for key, value in checkpoint['model'].items():
      key = key.replace("module.", "")
      new_ckpt[key] = value
del checkpoint

load_result = model.load_state_dict(new_ckpt, strict=False)
model.eval()
model.to("cuda")


def predict(json_path, add_special_token):
    
    ecg_root_path='./Dataset/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0'
    cxr_root_path='./Dataset/mimic-cxr-jpg-chest-radiographs-with-structured-labels-2.1.0'
    test_df = pd.read_csv("./QA/test_dip.csv")
    ecg_paths = test_df['ecg_path']
    cxr_paths = test_df['cxr_path']

    with open(json_path, 'r', encoding='utf-8') as f:
            res = json.load(f)
    data=[]
    for i, instance in tqdm(enumerate(res), total=len(res)):
        question = [instance['messages'][0]['content']]
        answer = instance['messages'][1]['content']

        ecg = get_ecg(os.path.join(ecg_root_path, ecg_paths.iloc[i])).unsqueeze(0).to("cuda")
        cxr = get_cxr(os.path.join(cxr_root_path, cxr_paths.iloc[i]), split='test').unsqueeze(0).to("cuda")
        lab = get_lab(test_df.iloc[i]).unsqueeze(0).to("cuda")

        response = model.text_completion(question, ecg, cxr, lab, temperature=0.6, top_p=0.9,
                                  max_gen_len=600, add_special_token=add_special_token)
        # response_chat, response_outputs, filename = parse_reponse(response, audio_length_in_s)
        answer = response[0]['generation']
        print(f"Q. {question[0]}")
        print(f"A. {answer}")
        
        entry = {
        "messages": [
            {"role": "user", "content": question[0]},
            {"role": "assistant", "content": answer}
            ],
        "ECG": [f"{instance['ECG'][0]}"]
        }
        print(entry)
        data.append(entry)
    with open(args.output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


predict(args.json_path, args.add_special_token)

