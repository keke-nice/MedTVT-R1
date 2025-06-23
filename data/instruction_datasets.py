import torch
from torch.utils.data import Dataset
import json
import llama.utils
from transformers import LlamaTokenizer
import copy
import os
import numpy as np
import torchaudio
from PIL import Image
import av
from tqdm.auto import tqdm
from torchvision import transforms
from .process_and_save_tensors import get_cxr, get_ecg, get_lab
import pdb
import torch.nn.functional as F
import pandas as pd

resize = transforms.Resize((224, 224))
class CXRQADataset(Dataset):
    """Dataset for chest X-ray question answering."""

    def __init__(self, json_path: str, root_path: str, tokenizer: LlamaTokenizer, max_words: int, llama_type: str, add_special_token: bool):
        print('Loading CXRQA dataset...')
        self.max_words = max_words
        self.tokenizer = tokenizer
        self.root_path = root_path
        self.question_list = []
        self.answer_list = []
        self.cxr_path_list = []
        self.llama_type = llama_type
        self.add_special_token = add_special_token
        self.data_dir = '/remote-home/hao.lu/Code/Yuting/ECG/ECG_pretrain/Dataset/symile-mimic'

        with open(json_path, 'r', encoding='utf-8') as f:
            res = json.load(f)

        for instance in tqdm(res, total=len(res)):
            user_message = instance['messages'][0]['content']
            # parts = user_message.split("<image>")
            # user_message = parts[1].strip() if len(parts) > 1 else None
            assistant_message = instance['messages'][1]['content']
            self.question_list.append(user_message)
            self.answer_list.append(assistant_message) 
            self.cxr_path_list.append(os.path.join(root_path, instance['image'][0]))
        # self.cxr_tensors = torch.tensor(np.load(os.path.join(self.data_dir, "data_npy/train/cxr_train.npy")))
        # print('cxr_tensors:', self.cxr_tensors.shape)
        print("cxr_len:", len(self.cxr_path_list))

    def __len__(self):
        return len(self.cxr_path_list)

    def __getitem__(self, index):
        # cxr = self.cxr_tensors[index]
        # cxr = resize(cxr.unsqueeze(0)).squeeze(0)

        cxr = get_cxr(self.cxr_path_list[index])
        question = self.question_list[index]
        answer = self.answer_list[index]
        ecg = torch.zeros((12, 1000))
        lab = torch.zeros((100))
        # Tokenization
        input1 = question
        input2 = input1 + " " + answer  # Concatenate question and answer
        if self.llama_type=='llama3':
            if self.add_special_token:
                input1_tensor = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False, allowed_special={"<ecg>","<cxr>","<lab>"}), dtype=torch.int64)
                input2_tensor = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True, allowed_special={"<ecg>","<cxr>","<lab>"}), dtype=torch.int64)
            else:
                input1_tensor = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.int64)
                input2_tensor = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64)
        elif self.llama_type=='7B':
            input1_tensor = torch.tensor(self.tokenizer(input1).input_ids, dtype=torch.int64)
            input2_tensor = torch.tensor(self.tokenizer(input2).input_ids, dtype=torch.int64)
        # Padding
        padding = self.max_words - input2_tensor.shape[0]
        if padding > 0:
            input2_tensor = F.pad(input2_tensor, (0, padding), "constant", self.tokenizer.pad_id)  # Pad the tensor if its length is smaller than self.max_words
        elif padding < 0:
            input2_tensor = input2_tensor[:self.max_words]  # Truncate the tensor if its length is larger than self.max_words
        labels = input2_tensor.clone()
        labels[:len(input1_tensor)] = -1  # Ignore the question part in labels

        input2_mask = input2_tensor.ge(0)
        label_mask = labels.ge(0)
        input2_tensor[~input2_mask] = 0
        labels[~label_mask] = 0
        
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()
    
        return input2_tensor, labels, input2_mask, ecg, cxr, lab, 'cxr'

class ECGQADataset(Dataset):
    """Dataset for chest X-ray question answering."""

    def __init__(self, json_path: str, root_path: str, tokenizer: LlamaTokenizer, max_words: int, llama_type: str, add_special_token: bool):
        print('Loading ECGQA dataset...')
        self.max_words = max_words
        self.tokenizer = tokenizer
        self.root_path = root_path
        self.question_list = []
        self.answer_list = []
        self.ecg_path_list = []
        self.llama_type = llama_type
        self.add_special_token = add_special_token

        with open(json_path, 'r', encoding='utf-8') as f:
            res = json.load(f)

        for instance in tqdm(res, total=len(res)):
            user_message = instance['messages'][0]['content']
            # parts = user_message.split("<ECG>") 
            # user_message = parts[1].strip() if len(parts) > 1 else None
            assistant_message = instance['messages'][1]['content']
            self.question_list.append(user_message)
            self.answer_list.append(assistant_message)
            self.ecg_path_list.append(os.path.join(root_path, instance['ECG'][0]))
        print('ecg_len:', len(self.ecg_path_list))


    def __len__(self):
        return len(self.ecg_path_list)

    def __getitem__(self, index):
        image_path = self.ecg_path_list[index]
        ecg = get_ecg(image_path)
        cxr = torch.zeros((3,224,224))
        lab = torch.zeros((100))
        question = self.question_list[index]
        answer = self.answer_list[index]

        # Tokenization
        input1 = question
        input2 = input1 + " " + answer  # Concatenate question and answer
        if self.llama_type=='llama3':
            if self.add_special_token:
                input1_tensor = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False, allowed_special={"<ecg>","<cxr>","<lab>"}), dtype=torch.int64)
                input2_tensor = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True, allowed_special={"<ecg>","<cxr>","<lab>"}), dtype=torch.int64)
            else:
                input1_tensor = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.int64)
                input2_tensor = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64)
        elif self.llama_type=='7B':
            input1_tensor = torch.tensor(self.tokenizer(input1).input_ids, dtype=torch.int64)
            input2_tensor = torch.tensor(self.tokenizer(input2).input_ids, dtype=torch.int64)
        # Padding
        padding = self.max_words - input2_tensor.shape[0]
        if padding > 0:
            input2_tensor = F.pad(input2_tensor, (0, padding), "constant", self.tokenizer.pad_id)  # Pad the tensor if its length is smaller than self.max_words
        elif padding < 0:
            input2_tensor = input2_tensor[:self.max_words]  # Truncate the tensor if its length is larger than self.max_words
        labels = input2_tensor.clone()
        labels[:len(input1_tensor)] = -1  # Ignore the question part in labels

        input2_mask = input2_tensor.ge(0)
        label_mask = labels.ge(0)
        input2_tensor[~input2_mask] = 0
        labels[~label_mask] = 0
        
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()
        return input2_tensor, labels, input2_mask, ecg, cxr, lab, 'ecg'
    

class LABQADataset(Dataset):
    """Dataset for chest X-ray question answering."""

    def __init__(self, json_path: str, root_path: str, tokenizer: LlamaTokenizer, max_words: int, llama_type: str, add_special_token: bool):
        print('Loading LABQA dataset...')
        self.max_words = max_words
        self.tokenizer = tokenizer
        self.root_path = root_path
        self.question_list = []
        self.answer_list = []
        self.cxr_path_list = []
        self.llama_type = llama_type
        self.data_dir = '/remote-home/hao.lu/Code/Yuting/ECG/ECG_pretrain/Dataset/symile-mimic'
        self.add_special_token = add_special_token

        with open(json_path, 'r', encoding='utf-8') as f:
            res = json.load(f)

        for instance in tqdm(res, total=len(res)):
            user_message = instance['messages'][0]['content']
            assistant_message = instance['messages'][1]['content']
            self.question_list.append(user_message)
            self.answer_list.append(assistant_message) 
            self.cxr_path_list.append(os.path.join(root_path, instance['image'][0]))
        # self.lab_percentiles_tensors = torch.tensor(np.load(os.path.join(self.data_dir, "data_npy/train/labs_percentiles_train.npy")))
        # self.labs_missingness_tensors = torch.tensor(np.load(os.path.join(self.data_dir, "data_npy/train/labs_missingness_train.npy")))
        # self.lab_tensors = torch.cat([self.lab_percentiles_tensors, self.labs_missingness_tensors], dim=1)
        # print('lab_tensors:', self.lab_tensors.shape)
        self.train_df = pd.read_csv("/remote-home/hao.lu/Code/Yuting/ECG/ECG_pretrain/Dataset/symile-mimic/train.csv")
        print('lab_len:', len(self.train_df))

    def __len__(self):
        return len(self.cxr_path_list)

    def __getitem__(self, index):
        cxr = torch.zeros((3,224,224))
        ecg = torch.zeros((12, 1000))
        # lab = self.lab_tensors[index]
        lab = get_lab(self.train_df.iloc[index])
        question = self.question_list[index]
        answer = self.answer_list[index]

        # Tokenization
        input1 = question
        input2 = input1 + " " + answer  # Concatenate question and answer
        if self.llama_type=='llama3':
            if self.add_special_token:
                input1_tensor = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False, allowed_special={"<ecg>","<cxr>","<lab>"}), dtype=torch.int64)
                input2_tensor = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True, allowed_special={"<ecg>","<cxr>","<lab>"}), dtype=torch.int64)
            else:
                input1_tensor = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.int64)
                input2_tensor = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64)
        elif self.llama_type=='7B':
            input1_tensor = torch.tensor(self.tokenizer(input1).input_ids, dtype=torch.int64)
            input2_tensor = torch.tensor(self.tokenizer(input2).input_ids, dtype=torch.int64)
        # Padding
        padding = self.max_words - input2_tensor.shape[0]
        if padding > 0:
            input2_tensor = F.pad(input2_tensor, (0, padding), "constant", self.tokenizer.pad_id)  # Pad the tensor if its length is smaller than self.max_words
        elif padding < 0:
            input2_tensor = input2_tensor[:self.max_words]  # Truncate the tensor if its length is larger than self.max_words
        labels = input2_tensor.clone()
        labels[:len(input1_tensor)] = -1  # Ignore the question part in labels

        input2_mask = input2_tensor.ge(0)
        label_mask = labels.ge(0)
        input2_tensor[~input2_mask] = 0
        labels[~label_mask] = 0
        
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()
    
        return input2_tensor, labels, input2_mask, ecg, cxr, lab, 'lab'
  


class All3QADataset(Dataset):
    """Dataset for chest X-ray question answering."""

    def __init__(self, json_path: str, ecg_root_path: str, cxr_root_path: str, tokenizer: LlamaTokenizer, max_words: int, llama_type: str, add_special_token: bool):
        print('Loading ALL3QA dataset...')
        self.max_words = max_words
        self.tokenizer = tokenizer
        self.ecg_root_path = ecg_root_path
        self.cxr_root_path = cxr_root_path
        self.question_list = []
        self.answer_list = []
        self.ecg_path_list = []
        self.llama_type = llama_type
        self.data_dir = '/remote-home/hao.lu/Code/Yuting/ECG/ECG_pretrain/Dataset/symile-mimic'
        self.add_special_token = add_special_token

        with open(json_path, 'r', encoding='utf-8') as f:
            res = json.load(f)

        for instance in tqdm(res, total=len(res)):
            user_message = instance['messages'][0]['content']
            assistant_message = instance['messages'][1]['content']
            self.question_list.append(user_message)
            self.answer_list.append(assistant_message)
            self.ecg_path_list.append(instance['ECG'][0])
        
        print('ecg_path:', len(self.ecg_path_list))
        # self.cxr_tensors = torch.tensor(np.load(os.path.join(self.data_dir, "data_npy/train/cxr_train.npy")))
        # print('cxr_tensors:', self.cxr_tensors.shape)
        # self.lab_percentiles_tensors = torch.tensor(np.load(os.path.join(self.data_dir, "data_npy/train/labs_percentiles_train.npy")))
        # self.labs_missingness_tensors = torch.tensor(np.load(os.path.join(self.data_dir, "data_npy/train/labs_missingness_train.npy")))
        # self.lab_tensors = torch.cat([self.lab_percentiles_tensors, self.labs_missingness_tensors], dim=1)
        # print('lab_tensors:', self.lab_tensors.shape)
        self.train_df = pd.read_csv("/remote-home/hao.lu/Code/Yuting/ECG/ECG_pretrain/Dataset/symile-mimic/train_filtered_7_catagory.csv")
        self.ecg_paths = self.train_df['ecg_path']
        self.cxr_paths = self.train_df['cxr_path']
        assert len(self.ecg_path_list)==len(self.ecg_paths)



    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, index):
        # image_path = self.ecg_path_list[index]
        # ecg = get_ecg(image_path)
        ecg = get_ecg(os.path.join(self.ecg_root_path, self.ecg_paths.iloc[index]))
        question = self.question_list[index]
        answer = self.answer_list[index]
        # cxr = self.cxr_tensors[index]
        # cxr = resize(cxr.unsqueeze(0)).squeeze(0)
        cxr = get_cxr(os.path.join(self.cxr_root_path, self.cxr_paths.iloc[index]))
        # lab = self.lab_tensors[index]
        lab = get_lab(self.train_df.iloc[index])

        # Tokenization
        input1 = question
        input2 = input1 + " " + answer  # Concatenate question and answer
        if self.llama_type=='llama3':
            if self.add_special_token:
                input1_tensor = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False, allowed_special={"<ecg>","<cxr>","<lab>"}), dtype=torch.int64)
                input2_tensor = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True, allowed_special={"<ecg>","<cxr>","<lab>"}), dtype=torch.int64)
            else:
                input1_tensor = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.int64)
                input2_tensor = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64)
        elif self.llama_type=='7B':
            input1_tensor = torch.tensor(self.tokenizer(input1).input_ids, dtype=torch.int64)
            input2_tensor = torch.tensor(self.tokenizer(input2).input_ids, dtype=torch.int64)
        # Padding
        padding = self.max_words - input2_tensor.shape[0]
        if padding > 0:
            input2_tensor = F.pad(input2_tensor, (0, padding), "constant", self.tokenizer.pad_id)  # Pad the tensor if its length is smaller than self.max_words
        elif padding < 0:
            input2_tensor = input2_tensor[:self.max_words]  # Truncate the tensor if its length is larger than self.max_words
        labels = input2_tensor.clone()
        labels[:len(input1_tensor)] = -1  # Ignore the question part in labels

        input2_mask = input2_tensor.ge(0)
        label_mask = labels.ge(0)
        input2_tensor[~input2_mask] = 0
        labels[~label_mask] = 0
        
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()
        return input2_tensor, labels, input2_mask, ecg, cxr, lab, 'all'
       

        
