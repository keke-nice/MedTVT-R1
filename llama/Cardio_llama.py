import json
import os
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .llama import Transformer, ModelArgs, RMSNorm
from util.misc import download
from .utils import sample_top_p

# from .audioldm2 import AudioLDM2Pipeline
from .tokenizer_llama3 import ChatFormat, Dialog, Message, Tokenizer

from transformers import LlamaTokenizer
from transformers import ViTImageProcessor, ViTModel, ViTConfig


from .xresnet1d_101 import xresnet1d101
from .lab_encoder import LabsEncoder
from .MPL import CrossModalAttention, DynamicWeighting, CyclicCrossModalAttention
import pdb

import tiktoken
from tiktoken.load import load_tiktoken_bpe
import matplotlib.pyplot as plt
from collections import OrderedDict

from typing import List, Optional, Tuple, TypedDict
from deepspeed.runtime.zero.partition_parameters import GatheredParameters

class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required
from transformers import PretrainedConfig

class CustomConfig(PretrainedConfig):
    def __init__(self, hidden_size=None, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size  
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self):
        output = super().to_dict()
        output.update(self.__dict__)  
        return output

class Cardio_LLaMA(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, llama_ckpt_dir, llama_tokenizer, model_args, stage=1,
                 legacy_bridge=False, load_llama=True, device=None):
        super().__init__()

        self.args = model_args

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        # 4. llama
        with open(os.path.join(llama_ckpt_dir, "params.json"), "r") as f:
            params = json.loads(f.read())
        bias_lora = True
   
        self.model_args: ModelArgs = ModelArgs(
                max_seq_len=1024, max_batch_size=4, w_bias=bias_lora, w_lora=bias_lora,
                **params)  # max_batch_size only affects inference
        self.config = CustomConfig()
        self.config.hidden_size = self.model_args.dim
        print(f"model args: {self.model_args}")

        # ECG Encoder 加载ECG预训练模型
        print(f'Initialize ECG encoder ...')
        ecg_checkpoint = torch.load('./CKPTS/best_valid_all_increase_with_augment_epoch_3.pt', map_location='cpu')
        ecg_model_state_dict = ecg_checkpoint['ecg_model']
        # self.ecg_encoder = models.resnet18(weights="IMAGENET1K_V1")
        self.ecg_encoder = xresnet1d101(num_classes=5, input_channels=12, kernel_size=5,
                          ps_head=0.5, lin_ftrs_head=[768],
                         use_ecgNet_Diagnosis='other'
                         )
        self.ecg_encoder.load_state_dict(ecg_model_state_dict)
        # self.ecg_xresnet1d101_agg = nn.Conv1d(in_channels=768, out_channels=1, kernel_size=1)
        self.ecg_xresnet1d101_rnn = nn.RNN(input_size=768, hidden_size=1024, batch_first=True)
        self.ecg_xresnet1d101_attention = nn.Linear(1024, 1)
        self.ecg_xresnet1d101_softmax = nn.Softmax(dim=1)
        self.ecg_xresnet1d101_proj = nn.Linear(1024, self.model_args.dim)
        
        if legacy_bridge:
            bridge_norm_layer = nn.LayerNorm
            bridge_bias = True
        else:
            bridge_norm_layer = RMSNorm
            bridge_bias = False

        self.feature_scaler = 1

        self.ecg_xresnet1d101_norm_1 = bridge_norm_layer(self.model_args.dim)
        self.ecg_xresnet1d101_f1_1 = nn.Linear(self.model_args.dim, self.model_args.dim * self.feature_scaler, bias=bridge_bias)
        self.ecg_xresnet1d101_f2_1 = nn.Linear(self.model_args.dim * self.feature_scaler, self.model_args.dim, bias=bridge_bias)
        self.ecg_xresnet1d101_f3_1 = nn.Linear(self.model_args.dim, self.model_args.dim * self.feature_scaler, bias=bridge_bias)

        self.ecg_xresnet1d101_norm_2 = bridge_norm_layer(self.model_args.dim)
        self.ecg_xresnet1d101_f1_2 = nn.Linear(self.model_args.dim, self.model_args.dim * self.feature_scaler, bias=bridge_bias)
        self.ecg_xresnet1d101_f2_2 = nn.Linear(self.model_args.dim * self.feature_scaler, self.model_args.dim, bias=bridge_bias)
        self.ecg_xresnet1d101_f3_2 = nn.Linear(self.model_args.dim, self.model_args.dim * self.feature_scaler, bias=bridge_bias)

        self.ecg_xresnet1d101_norm_3 = bridge_norm_layer(self.model_args.dim)
        self.ecg_xresnet1d101_f1_3 = nn.Linear(self.model_args.dim, self.model_args.dim * self.feature_scaler, bias=bridge_bias)
        self.ecg_xresnet1d101_f2_3 = nn.Linear(self.model_args.dim * self.feature_scaler, self.model_args.dim, bias=bridge_bias)
        self.ecg_xresnet1d101_f3_3 = nn.Linear(self.model_args.dim, self.model_args.dim * self.feature_scaler, bias=bridge_bias)
        print(f'ECG Encoder initialized...')
      

        # 2. ViT Encoder
        # The model files for ViT can be downloaded here in case of network issues:
        # https://huggingface.co/google/vit-base-patch16-224-in21k
        # And set the vit_path argument to directory with the model files
        print(f'Initialize ViT...')
        # self.vit_model = ViTModel.from_pretrained(self.args.vit_path).to(self.device)  # .to(self.device)
        config_vit = ViTConfig.from_pretrained(self.args.vit_path) #注意transformer版本号！
        self.vit_model = ViTModel(config_vit)
        self.vit_model.eval()
        self.vit_processor = ViTImageProcessor.from_pretrained(self.args.vit_path, do_rescale=False)
        self.cxr_vit_agg = nn.Conv1d(in_channels=197, out_channels=1, kernel_size=1)
        self.cxr_vit_proj = nn.Linear(768, self.model_args.dim)

        self.cxr_vit_norm_1 = bridge_norm_layer(self.model_args.dim)
        self.cxr_vit_f1_1 = nn.Linear(self.model_args.dim, self.model_args.dim * self.feature_scaler, bias=bridge_bias)
        self.cxr_vit_f2_1 = nn.Linear(self.model_args.dim * self.feature_scaler, self.model_args.dim, bias=bridge_bias)
        self.cxr_vit_f3_1 = nn.Linear(self.model_args.dim, self.model_args.dim * self.feature_scaler, bias=bridge_bias)

        self.cxr_vit_norm_2 = bridge_norm_layer(self.model_args.dim)
        self.cxr_vit_f1_2 = nn.Linear(self.model_args.dim, self.model_args.dim * self.feature_scaler, bias=bridge_bias)
        self.cxr_vit_f2_2 = nn.Linear(self.model_args.dim * self.feature_scaler, self.model_args.dim, bias=bridge_bias)
        self.cxr_vit_f3_2 = nn.Linear(self.model_args.dim, self.model_args.dim * self.feature_scaler, bias=bridge_bias)

        self.cxr_vit_norm_3 = bridge_norm_layer(self.model_args.dim)
        self.cxr_vit_f1_3 = nn.Linear(self.model_args.dim, self.model_args.dim * self.feature_scaler, bias=bridge_bias)
        self.cxr_vit_f2_3 = nn.Linear(self.model_args.dim * self.feature_scaler, self.model_args.dim, bias=bridge_bias)
        self.cxr_vit_f3_3 = nn.Linear(self.model_args.dim, self.model_args.dim * self.feature_scaler, bias=bridge_bias)
        print(f'ViT initialized...')

        print(f'Initial lab encoder...')
        self.lab_encoder = LabsEncoder()
        lab_checkpoint = torch.load('./CKPTS/epoch=49-val_loss=0.4944.ckpt', map_location='cpu')
        model_state_dict = lab_checkpoint['state_dict']
        # 提取 labs_encoder 的参数并去掉前缀
        labs_encoder_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            if 'labs_encoder' in k:
                # 去掉 'labs_encoder.' 前缀
                new_key = k.replace('labs_encoder.', '')
                labs_encoder_state_dict[new_key] = v
        self.lab_encoder.load_state_dict(labs_encoder_state_dict)
        self.lab_fc_proj = nn.Linear(1024, self.model_args.dim)

        self.lab_fc_norm_1 = bridge_norm_layer(self.model_args.dim)
        self.lab_fc_f1_1 = nn.Linear(self.model_args.dim, self.model_args.dim * self.feature_scaler, bias=bridge_bias)
        self.lab_fc_f2_1 = nn.Linear(self.model_args.dim * self.feature_scaler, self.model_args.dim, bias=bridge_bias)
        self.lab_fc_f3_1 = nn.Linear(self.model_args.dim, self.model_args.dim * self.feature_scaler, bias=bridge_bias)

        self.lab_fc_norm_2 = bridge_norm_layer(self.model_args.dim)
        self.lab_fc_f1_2 = nn.Linear(self.model_args.dim, self.model_args.dim * self.feature_scaler, bias=bridge_bias)
        self.lab_fc_f2_2 = nn.Linear(self.model_args.dim * self.feature_scaler, self.model_args.dim, bias=bridge_bias)
        self.lab_fc_f3_2 = nn.Linear(self.model_args.dim, self.model_args.dim * self.feature_scaler, bias=bridge_bias)

        self.lab_fc_norm_3 = bridge_norm_layer(self.model_args.dim)
        self.lab_fc_f1_3 = nn.Linear(self.model_args.dim, self.model_args.dim * self.feature_scaler, bias=bridge_bias)
        self.lab_fc_f2_3 = nn.Linear(self.model_args.dim * self.feature_scaler, self.model_args.dim, bias=bridge_bias)
        self.lab_fc_f3_3 = nn.Linear(self.model_args.dim, self.model_args.dim * self.feature_scaler, bias=bridge_bias)
        print(f'lab encoder initialized...')


        if stage == 3:
          self.DynamicWeighting = DynamicWeighting(self.model_args.dim)
          self.CyclicCrossModalAttention = CyclicCrossModalAttention(self.model_args.dim)

        # 5. tokenizer
        if self.args.llama_type=='7B':
            self.tokenizer = LlamaTokenizer.from_pretrained(
                llama_tokenizer)  # Tokenizer(model_path=llama_tokenizer, num_aud_tokens=self.model_args.num_gen_audio_tokens)
            # self._add_audio_token()
            self.tokenizer.pad_id = self.tokenizer.eos_id
        elif self.args.llama_type=='llama3' or self.args.llama_type=='8B':
            self.tokenizer = Tokenizer(llama_tokenizer)
            self.tokenizer.pad_id = self.tokenizer.eos_id

        if torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        self.llama = Transformer(self.model_args)
        torch.set_default_tensor_type(torch.FloatTensor)

        if load_llama:
            print(f"Loading LLaMA Checkpoint...")
            ckpts = sorted(Path(llama_ckpt_dir).glob("*.pth"))

            """
            Adapted from https://github.com/cedrickchee/llama/blob/main/chattyllama/combined/inference.py
            """
            key_to_dim = {
                "w1": 0,
                "w2": -1,
                "w3": 0,
                "wo": -1,
                "wq": 0,
                "wk": 0,
                "wv": 0,
                "output": 0,
                "tok_embeddings": -1,
                "ffn_norm": None,
                "attention_norm": None,
                "norm": None,
                "rope": None,
            }
            for i, ckpt in enumerate(ckpts):
                checkpoint = torch.load(ckpt, map_location="cpu")
                for parameter_name, parameter in self.llama.named_parameters():
                    short_name = parameter_name.split(".")[-2] #'layers.0.attention.wk.weight'
                    if "gate" in parameter_name or "lora" in parameter_name or "bias" in parameter_name:
                        continue
                    if key_to_dim[short_name] is None and i == 0:
                        parameter.data = checkpoint[parameter_name]
                    elif key_to_dim[short_name] == 0: 
                        size = checkpoint[parameter_name].size(0) #[4096, 4096] 大小一样
                        parameter.data[size * i: size * (i + 1), :] = checkpoint[
                            parameter_name
                        ]
                    elif key_to_dim[short_name] == -1:
                        size = checkpoint[parameter_name].size(-1) #[32000,4096] /[4096,4096]/ [11008, 4096]大小一样
                        parameter.data[:, size * i: size * (i + 1)] = checkpoint[
                            parameter_name
                        ]
  
                del checkpoint
            print(f"LLaMA Checkpoint Loaded")

        

        # 6. training criterion
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.l2_loss = torch.nn.MSELoss()
        self.stage = stage
        self.set_default_trainability(self.stage)

    def get_trainable_params(self, stage=1):
        trainable = {}
        if stage == 1:
            for name, para in self.named_parameters():
                if "llama." in name:
                    if 'norm' in name or 'bias' in name or 'lora' in name:
                        trainable[name] = para
                if "tok_embeddings" in name:
                    trainable[name] = para
                if 'ecg_xresnet1d101_' in name:
                    trainable[name] = para
                if "cxr_vit_" in name:
                    trainable[name] = para
                if "lab_fc_" in  name:
                    trainable[name] = para

        elif stage == 3:
            for name, para in self.named_parameters():
                if "llama." in name:
                    if 'norm' in name or 'bias' in name or 'lora' in name:
                        trainable[name] = para
                if "tok_embeddings" in name:
                    trainable[name] = para
                if "CyclicCrossModalAttention" in name:
                    trainable[name] = para
                if "DynamicWeighting" in name:
                    trainable[name] = para
                # if 'ecg_xresnet1d101_' in name:
                #     trainable[name] = para
                # if "cxr_vit_" in name:
                #     trainable[name] = para
                # if "lab_fc_" in  name:
                #     trainable[name] = para

        return trainable

    def set_default_trainability(self, stage=1):
        for key, value in self.named_parameters():
            value.requires_grad = False
        trainable_params = self.get_trainable_params(stage)
        print(f"Trainable Params: {trainable_params.keys()}")
        for key, value in trainable_params.items():
            value.data = value.data.float()
            value.requires_grad = True
    
    def encode_ecg(self, x):
      with torch.no_grad():
            outputs = self.ecg_encoder(x)  # Outputs shape: [1, 768, T]
      outputs = outputs.permute(0, 2, 1)
    
      return outputs #[B, 32, 768]
    def encode_lab(self, x):
        with torch.no_grad():
            outputs = self.lab_encoder(x)  # Outputs shape: [B, 1024]
        return outputs

    def encode_image(self, x):
        xs = []
        for sub_x in x:
            # inputs = self.vit_processor(images=sub_x, return_tensors="pt").to(self.vit_model.device)
            with torch.no_grad():
                outputs = self.vit_model(sub_x.unsqueeze(0))
            last_hidden_states = outputs.last_hidden_state
            out_x = self.cxr_vit_agg(last_hidden_states.to(self.device)).squeeze()
            xs.append(out_x)
        return torch.stack(xs, dim=0)

    
    def forward_ecg(self, inputs, cache_size=10, cache_t=20, cache_weight=0.5):
        outputs = []
        outputs_weights = []
        for input_type, (input, input_weight) in inputs.items():
            outputs.append(F.normalize(self.encode_ecg(input), dim=-1))
            outputs_weights.append(input_weight)
        outputs_weights = [x / (sum(outputs_weights) + 1e-6) for x in outputs_weights]

        ecg_feats = sum([output * output_weight for output, output_weight in zip(outputs, outputs_weights)])
        device = ecg_feats.device

        ecg_feats, _ = self.ecg_xresnet1d101_rnn(ecg_feats)

        attention_weights = self.ecg_xresnet1d101_attention(ecg_feats).squeeze(-1)
        attention_scores = self.ecg_xresnet1d101_softmax(attention_weights)
        
        ecg_feats = torch.matmul(attention_scores.unsqueeze(1), ecg_feats).squeeze(1)

        ecg_feats = ecg_feats.unsqueeze(1)  # B, 1, D
        ecg_feats = self.ecg_xresnet1d101_proj(ecg_feats)
        ecg_feats_norm = self.ecg_xresnet1d101_norm_1(ecg_feats)
        ecg_feats = ecg_feats + self.ecg_xresnet1d101_f2_1(
            F.silu(self.ecg_xresnet1d101_f1_1(ecg_feats_norm)) * self.ecg_xresnet1d101_f3_1(ecg_feats_norm))

        ecg_feats_norm = self.ecg_xresnet1d101_norm_2(ecg_feats)
        ecg_feats = ecg_feats + self.ecg_xresnet1d101_f2_2(
            F.silu(self.ecg_xresnet1d101_f1_2(ecg_feats_norm)) * self.ecg_xresnet1d101_f3_2(ecg_feats_norm))

        ecg_feats_norm = self.ecg_xresnet1d101_norm_3(ecg_feats)
        ecg_feats = ecg_feats + self.ecg_xresnet1d101_f2_3(
            F.silu(self.ecg_xresnet1d101_f1_3(ecg_feats_norm)) * self.ecg_xresnet1d101_f3_3(ecg_feats_norm))
        return ecg_feats


    def forward_image(self, inputs, cache_size=10, cache_t=20, cache_weight=0.5):
        outputs = []
        outputs_weights = []
        for input_type, (input, input_weight) in inputs.items():
            outputs.append(F.normalize(self.encode_image(input), dim=-1))
            outputs_weights.append(input_weight)
        outputs_weights = [x / (sum(outputs_weights) + 1e-6) for x in outputs_weights]

        image_feats = sum([output * output_weight for output, output_weight in zip(outputs, outputs_weights)])
        device = image_feats.device

        image_feats = image_feats.unsqueeze(1)  # B, 1, D
        image_feats = self.cxr_vit_proj(image_feats)
        image_feats_norm = self.cxr_vit_norm_1(image_feats)
        image_feats = image_feats + self.cxr_vit_f2_1(
            F.silu(self.cxr_vit_f1_1(image_feats_norm)) * self.cxr_vit_f3_1(image_feats_norm))

        image_feats_norm = self.cxr_vit_norm_2(image_feats)
        image_feats = image_feats + self.cxr_vit_f2_2(
            F.silu(self.cxr_vit_f1_2(image_feats_norm)) * self.cxr_vit_f3_2(image_feats_norm))

        image_feats_norm = self.cxr_vit_norm_3(image_feats)
        image_feats = image_feats + self.cxr_vit_f2_3(
            F.silu(self.cxr_vit_f1_3(image_feats_norm)) * self.cxr_vit_f3_3(image_feats_norm))
        return image_feats
    
    def forward_lab(self, inputs, cache_size=10, cache_t=20, cache_weight=0.5):
        outputs = []
        outputs_weights = []
        for input_type, (input, input_weight) in inputs.items():
            outputs.append(F.normalize(self.encode_lab(input), dim=-1))
            outputs_weights.append(input_weight)
        outputs_weights = [x / (sum(outputs_weights) + 1e-6) for x in outputs_weights]

        lab_feats = sum([output * output_weight for output, output_weight in zip(outputs, outputs_weights)])
        device = lab_feats.device

        lab_feats = lab_feats.unsqueeze(1)  # B, 1, D
        lab_feats = self.lab_fc_proj(lab_feats)
        lab_feats_norm = self.lab_fc_norm_1(lab_feats)
        lab_feats = lab_feats + self.lab_fc_f2_1(
            F.silu(self.lab_fc_f1_1(lab_feats_norm)) * self.lab_fc_f3_1(lab_feats_norm))
        
        lab_feats_norm = self.lab_fc_norm_2(lab_feats)
        lab_feats = lab_feats + self.lab_fc_f2_2(
            F.silu(self.lab_fc_f1_2(lab_feats_norm)) * self.lab_fc_f3_2(lab_feats_norm))
        
        lab_feats_norm = self.lab_fc_norm_3(lab_feats)
        lab_feats = lab_feats + self.lab_fc_f2_3(
            F.silu(self.lab_fc_f1_3(lab_feats_norm)) * self.lab_fc_f3_3(lab_feats_norm))

        return lab_feats


    def forward(self, tokens, labels, ecgs=None, cxrs=None, labs=None):
        ecg_feats, cxr_feats, lab_feats = None, None, None

        _bsz, seqlen = tokens.shape
        
        model_dtype = next(self.parameters()).dtype
        h = self.llama.tok_embeddings(tokens.to(self.device))
      
        

        token_ecg = torch.tensor(self.tokenizer.encode('<ecg>', bos=False, eos=False, allowed_special={"<ecg>","<cxr>","<lab>"}), dtype=torch.int64, device=h.device)
        token_cxr = torch.tensor(self.tokenizer.encode('<cxr>', bos=False, eos=False, allowed_special={"<ecg>","<cxr>","<lab>"}), dtype=torch.int64, device=h.device)
        token_lab = torch.tensor(self.tokenizer.encode('<lab>', bos=False, eos=False, allowed_special={"<ecg>","<cxr>","<lab>"}), dtype=torch.int64, device=h.device)

        if self.stage==3:
            ecg_feats = self.forward_ecg({'Ecg': [ecgs, 1]})
            cxr_feats = self.forward_image({'Cxr': [cxrs, 1]})
            lab_feats = self.forward_lab({'Lab': [labs, 1]})
            cxr_feats, ecg_feats, lab_feats = self.CyclicCrossModalAttention(cxr_feats, ecg_feats, lab_feats)
            cxr_feats, ecg_feats, lab_feats = self.DynamicWeighting(cxr_feats, ecg_feats, lab_feats)

            ecg_positions = (tokens == token_ecg).nonzero(as_tuple=True)
            for batch_idx, pos in zip(ecg_positions[0], ecg_positions[1]): 
                    h[batch_idx, pos, :] = ecg_feats[batch_idx, 0, :]

            cxr_positions = (tokens == token_cxr).nonzero(as_tuple=True)
            for batch_idx, pos in zip(cxr_positions[0], cxr_positions[1]): 
                    h[batch_idx, pos, :] = cxr_feats[batch_idx, 0, :]

            lab_positions = (tokens == token_lab).nonzero(as_tuple=True)
            for batch_idx, pos in zip(lab_positions[0], lab_positions[1]): 
                    h[batch_idx, pos, :] = lab_feats[batch_idx, 0, :]
        
        elif self.stage==1:

            if ecgs is not None and not torch.all(ecgs == 0):
                ecg_feats = self.forward_ecg({'Ecg': [ecgs, 1]}) #[1,1,4096]
                if self.args.add_special_token:
                    ecg_positions = (tokens == token_ecg).nonzero(as_tuple=True)
                    for batch_idx, pos in zip(ecg_positions[0], ecg_positions[1]): 
                         h[batch_idx, pos, :] = ecg_feats[batch_idx, 0, :]
                else:
                    part1 = h[:, :1, :]  
                    part2 = h[:, 1:, :]  
                    h = torch.cat((part1, ecg_feats, part2), dim=1)
            
                for name, para in self.named_parameters():
                    if 'ecg_xresnet1d101_' in name:
                        para.data = para.data.float()
                        para.requires_grad = True
                    
            if cxrs is not None and not torch.all(cxrs == 0):
                cxr_feats = self.forward_image({'Cxr': [cxrs, 1]})
                if self.args.add_special_token:
                    cxr_positions = (tokens == token_cxr).nonzero(as_tuple=True)
                    for batch_idx, pos in zip(cxr_positions[0], cxr_positions[1]): 
                        h[batch_idx, pos, :] = cxr_feats[batch_idx, 0, :]
                else:
                    part1 = h[:, :1, :]  
                    part2 = h[:, 1:, :]  
                    h = torch.cat((part1, cxr_feats, part2), dim=1)

                for name, para in self.named_parameters():
                    if 'cxr_vit_' in name:
                        para.data = para.data.float()
                        para.requires_grad = True
                
            if labs is not None and not torch.all(labs == 0):
                lab_feats = self.forward_lab({'Lab': [labs, 1]})
                if self.args.add_special_token:
                    lab_positions = (tokens == token_lab).nonzero(as_tuple=True)
                    for batch_idx, pos in zip(lab_positions[0], lab_positions[1]): 
                        h[batch_idx, pos, :] = lab_feats[batch_idx, 0, :]
                else:
                    part1 = h[:, :1, :]  
                    part2 = h[:, 1:, :]  
                    h = torch.cat((part1, lab_feats, part2), dim=1)
                for name, para in self.named_parameters():
                    if 'lab_fc_' in name:
                        para.data = para.data.float()
                        para.requires_grad = True
            
        seqlen_now = h.shape[1]
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[:seqlen_now]
        mask = torch.full((1, 1, seqlen_now, seqlen_now), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=0 + 1).type_as(h)

        
        with torch.cuda.amp.autocast():
            for layer in self.llama.layers:
                h = layer(h, 0, freqs_cis, mask)
  
            h = self.llama.norm(h)
            output = self.llama.output(h)
            # _, output_seq_len, _ = output.size()
            # _, label_seq_len = labels.size()
        
            # if output_seq_len != label_seq_len:
            #   # 计算需要填充的数量
            #   padding_size = output_seq_len - label_seq_len
            #   if padding_size > 0:
            #       # 创建一个填充的张量
            #       padding = torch.zeros(labels.size(0), padding_size, dtype=labels.dtype, device=labels.device)
            #       # 在第二个维度前面拼接填充的张量
            #       labels = torch.cat((padding, labels), dim=1)
            output = output[:, :-1, :]
            labels = labels[:, 1:]

            if labels.sum() == 0:
                c_loss = output.mean() * 0
            else:
                assert self.llama.vocab_size == self.model_args.vocab_size
                c_loss = self.criterion(output.reshape(-1, self.llama.vocab_size), labels.flatten().to(self.device))

            return output, c_loss
    
    @torch.inference_mode()
    def forward_inference(self, tokens, start_pos: int, ecg_feats=None, cxr_feats=None, lab_feats=None):
        _bsz, seqlen = tokens.shape
        h = self.llama.tok_embeddings(tokens)
        # freqs_cis = self.llama.freqs_cis.to(h.device)
        # freqs_cis = freqs_cis[start_pos:start_pos + seqlen]

        # mask = None
        # mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        # mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        token_ecg = torch.tensor(self.tokenizer.encode('<ecg>', bos=False, eos=False, allowed_special={"<ecg>","<cxr>","<lab>"}), dtype=torch.int64, device=h.device)
        token_cxr = torch.tensor(self.tokenizer.encode('<cxr>', bos=False, eos=False, allowed_special={"<ecg>","<cxr>","<lab>"}), dtype=torch.int64, device=h.device)
        token_lab = torch.tensor(self.tokenizer.encode('<lab>', bos=False, eos=False, allowed_special={"<ecg>","<cxr>","<lab>"}), dtype=torch.int64, device=h.device)

        if self.stage==3:
            cxr_feats, ecg_feats, lab_feats = self.CyclicCrossModalAttention(cxr_feats, ecg_feats, lab_feats)
            cxr_feats, ecg_feats, lab_feats = self.DynamicWeighting(cxr_feats, ecg_feats, lab_feats)

            ecg_positions = (tokens == token_ecg).nonzero(as_tuple=True)
            for batch_idx, pos in zip(ecg_positions[0], ecg_positions[1]): 
                    h[batch_idx, pos, :] = ecg_feats[batch_idx, 0, :] 

            cxr_positions = (tokens == token_cxr).nonzero(as_tuple=True)
            for batch_idx, pos in zip(cxr_positions[0], cxr_positions[1]): 
                    h[batch_idx, pos, :] = cxr_feats[batch_idx, 0, :]

            lab_positions = (tokens == token_lab).nonzero(as_tuple=True)
            for batch_idx, pos in zip(lab_positions[0], lab_positions[1]): 
                    h[batch_idx, pos, :] = lab_feats[batch_idx, 0, :]
        
        elif self.stage==1:

            if ecg_feats is not None:
                if self.args.add_special_token:
                    ecg_positions = (tokens == token_ecg).nonzero(as_tuple=True)
                    for batch_idx, pos in zip(ecg_positions[0], ecg_positions[1]): 
                        h[batch_idx, pos, :] = ecg_feats[batch_idx, 0, :]
                else:
                    part1 = h[:, :1, :]  
                    part2 = h[:, 1:, :]  
                    h = torch.cat((part1, ecg_feats, part2), dim=1)

            if cxr_feats is not None:
                if self.args.add_special_token:
                    cxr_positions = (tokens == token_cxr).nonzero(as_tuple=True)
                    for batch_idx, pos in zip(cxr_positions[0], cxr_positions[1]): 
                        h[batch_idx, pos, :] = cxr_feats[batch_idx, 0, :]
                else:
                    part1 = h[:, :1, :]  
                    part2 = h[:, 1:, :]  
                    h = torch.cat((part1, cxr_feats, part2), dim=1)

            if lab_feats is not None:
                if self.args.add_special_token:
                    lab_positions = (tokens == token_lab).nonzero(as_tuple=True)
                    for batch_idx, pos in zip(lab_positions[0], lab_positions[1]): 
                        h[batch_idx, pos, :] = lab_feats[batch_idx, 0, :]
                else:
                    part1 = h[:, :1, :]  
                    part2 = h[:, 1:, :] 
                    h = torch.cat((part1, lab_feats, part2), dim=1)

        seqlen_now = h.shape[1]
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[start_pos:start_pos + seqlen_now]
        mask = None
        mask = torch.full((1, 1, seqlen_now, seqlen_now), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.llama.layers:
            h = layer(h, 0, freqs_cis, mask)

        h = self.llama.norm(h)
        output = self.llama.output(h[:, -1, :])

        return output.float()

    @torch.inference_mode()
    def generate(
            self,
            prompt_tokens,
            ecgs=None,
            cxrs=None,
            labs=None,
            max_gen_len: int = 512,
            cache_size=10,
            cache_t=20,
            cache_weight=0.5,
            temperature: float = 0.6,
            top_p: float = 0.9,
            logprobs: bool = False,
            echo: bool = False,
    ):
        bsz = len(prompt_tokens)
        params = self.llama.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        
        with torch.cuda.amp.autocast():
            if ecgs is not None:
                ecg_feats = self.forward_ecg({'Ecg': [ecgs, 1]})
            else:
                ecg_feats = None
            if cxrs is not None:
                cxr_feats = self.forward_image({'Cxr': [cxrs, 1]})
            else:
                cxr_feats = None
            if labs is not None:
                lab_feats = self.forward_lab({'Lab': [labs, 1]})
            else:
                lab_feats = None

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        
        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id
   
        stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens), device="cuda")

        for cur_pos in range(min_prompt_len, total_len):
            with torch.cuda.amp.autocast():
                
                logits = self.forward_inference(tokens[:, prev_pos:cur_pos], prev_pos,
                                                                          ecg_feats, cxr_feats, lab_feats) #[1,1,4096]
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)

            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
           
            # if bsz == 1 and self.tokenizer.decode(tokens[0, cur_pos - 2:cur_pos + 1]) == "\n###":
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                torch.isin(next_token, stop_tokens)
            )
            # prev_pos = cur_pos
            if all(eos_reached):
                break   
          
        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cut to after eos tok if any
            for stop_token in self.tokenizer.stop_tokens:
                try:
                    eos_idx = toks.index(stop_token)
                    toks = toks[:eos_idx]
                    probs = probs[:eos_idx] if logprobs else None
                except ValueError:
                    pass
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if logprobs else None)
 
    @torch.inference_mode()
    def text_completion(
        self,
        prompts: List[str],
        ecg,
        cxr,
        lab,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
        add_special_token: bool = False
    ) -> List[CompletionPrediction]:
        """
        Perform text completion for a list of prompts using the language generation model.

        Args:
            prompts (List[str]): List of text prompts for completion.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated completion sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            List[CompletionPrediction]: List of completion predictions, each containing the generated text completion.

        Note:
            This method generates text completions for the provided prompts, employing nucleus sampling to introduce controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        if add_special_token:
            prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False, allowed_special={"<ecg>","<cxr>","<lab>"}) for x in prompts]
        else:
            prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
            ecgs=ecg,
            cxrs=cxr,
            labs=lab,
            cache_size=10,
            cache_t=20,
            cache_weight=0.5,
        )
        if logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(t),
                    "tokens": [self.tokenizer.decode([x]) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [{"generation": self.tokenizer.decode(t), "completion_ids": t} for t in generation_tokens]


