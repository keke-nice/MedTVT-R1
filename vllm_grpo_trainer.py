# Copyright 2025 The HuggingFace Team. All rights reserved.
#
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
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
from accelerate.utils.other import is_compiled_module
from accelerate.utils import broadcast_object_list, gather, gather_object
import torch
import torch.utils.data
import transformers
import warnings
from unittest.mock import patch
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    maybe_apply_chat_template,
)
from trl.import_utils import is_vllm_available

from trl.models import (
    create_reference_model,
    prepare_deepspeed,
    unwrap_model_for_generation,
)
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, pad
from trl import GRPOTrainer

import copy
from llama.Cardio_llama import Cardio_LLaMA

if is_peft_available():
    from peft import PeftConfig, get_peft_model



if is_wandb_available():
    import wandb
import torch.nn as nn
from torch.utils.data import Sampler
import pdb
import deepspeed

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset N times.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        repeat_count (`int`):
            Number of times to repeat each index.

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d"], repeat_count=2)
    >>> list(sampler)
    [2, 2, 0, 0, 3, 3, 1, 1]
    ```
    """

    def __init__(self, data_source, repeat_count: int):
        self.data_source = data_source
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)

    def __iter__(self):
        indexes = [
            idx
            for idx in torch.randperm(self.num_samples).tolist()
            for _ in range(self.repeat_count)
        ]
        return iter(indexes)

    def __len__(self):
        return self.num_samples * self.repeat_count


class llama3GRPOVLLMTrainer(Trainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        # qwen2-vl related params
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
    ):

        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        ###load my model

        ########################################################################################
        model_init_kwargs = model.model_args
        # model_init_kwargs = args.model_init_kwargs or {}
        # model_init_kwargs["attn_implementation"] = attn_implementation

        # Reference model
        if is_deepspeed_zero3_enabled():
            # 如果启用了 DeepSpeed ZeRO Stage 3，加载参考模型
            print("Loading Reference Model with DeepSpeed ZeRO-3")
            self.ref_model = Cardio_LLaMA(args.llama_ckpt_dir, args.llama_tokenzier_path, args, stage=3, load_llama=False)
            
            print("Loading Reference Model Checkpoint")
            checkpoint = torch.load(args.model_ckpt, map_location='cpu')

            new_ckpt = {}
            for key, value in checkpoint['model'].items():
                key = key.replace("module.", "")
                new_ckpt[key] = value
            del checkpoint

            load_result = self.ref_model.load_state_dict(new_ckpt, strict=True)

        elif peft_config is None:
            # 如果没有使用 PEFT，直接复制当前模型作为参考模型
            print("Creating Reference Model from Existing Model")
            self.ref_model = create_reference_model(model)

        else:
            # 如果使用了 PEFT，不需要参考模型
            print("PEFT is enabled, no reference model needed")
            self.ref_model = None

        print("Reference Model Initialized")
        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError(
                    "The number of reward processing classes must match the number of reward functions."
                )

        for i, (reward_processing_class, reward_func) in enumerate(
            zip(reward_processing_classes, reward_funcs)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(
                        reward_func.config._name_or_path
                    )
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = (
                        reward_processing_class.eos_token
                    )
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = (
            args.max_completion_length
        )  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        pad_token_id = processing_class.pad_id
        # self.generation_config = GenerationConfig(
        #     max_new_tokens=self.max_completion_length,
        #     do_sample=True,
        #     temperature=1,  # HACK
        #     num_return_sequences=self.num_generations,
        #     pad_token_id=pad_token_id,
        # )
        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        # model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)
  

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False
        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [
            n_gen
            for n_gen in range(2, global_batch_size + 1)
            if (global_batch_size) % n_gen == 0
        ]

        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [
                n_gen
                for n_gen in range(2, global_batch_size + 1)
                if (global_batch_size) % n_gen == 0
            ]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )


        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True
                )

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(
                    reward_func, evaluation_mode=True
                )

    # We need a custom sampler that samples the same prompt multiple times
    def _get_train_sampler(self):
        return RepeatRandomSampler(self.train_dataset, self.num_generations)

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(
        self,
        model,
        input_ids,
        labels,
        ecg,
        cxr,
        lab,
        logits_to_keep,
    ):
        ecg = ecg.to(model.device)
        cxr = cxr.to(device=model.device)
        lab = lab.to(device=model.device)
        logits, _ = model(input_ids, labels, ecgs=ecg, cxrs=cxr, labs=lab)
        input_ids = input_ids[
            :, -logits_to_keep:
        ]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        logits = logits[:, -logits_to_keep:]
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(
                log_probs, dim=1, index=input_ids_row.unsqueeze(1)
            ).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        dtype = next(self.model.parameters()).dtype
        import pdb
        # pdb.set_trace()
        labels = torch.stack([torch.tensor(x["labels"]) if isinstance(x["labels"], list) else x["labels"] for x in inputs])
        questions = [x["question"] for x in inputs]#list套str
        ground_truths = [x["answer"] for x in inputs] #list套str
        # prompt_ids = [x["prompt_ids"] for x in inputs] #列表里套列表
        prompt_ids = torch.stack([torch.tensor(x["prompt_ids"]) if isinstance(x["prompt_ids"], list) else x["prompt_ids"] for x in inputs]).to(device)
        ecg = torch.stack([torch.tensor(x["ecg"], dtype=dtype) if isinstance(x["ecg"], list) else x["ecg"] for x in inputs]).to(device)#[B,12,1000]
        cxr = torch.stack([torch.tensor(x["cxr"], dtype=dtype) if isinstance(x["cxr"], list) else x["cxr"] for x in inputs]).to(device)#[B,3,224,224]
        lab = torch.stack([torch.tensor(x["lab"], dtype=dtype) if isinstance(x["lab"], list) else x["lab"] for x in inputs]).to(device)#[B,100]
      

        ###生成答案
        self.model.eval()
        with torch.no_grad(), torch.inference_mode():
          response = self.model.text_completion(questions, ecg, cxr, lab, temperature=0.6, top_p=0.9,
                                      max_gen_len=600, add_special_token=True)

        completions, completion_ids = [], []
        for i in range(len(response)):
            completion = response[i]['generation']
            completion_id = response[i]['completion_ids']
            completions.append(completion)
            if isinstance(completion, torch.Tensor):
                completion_ids.append(completion_id.to(device))
            else:
                completion_ids.append(torch.tensor(completion_id, device=device))
        

        eos_token_id = self.processing_class.eos_id  

        # 在每个 completion_ids 序列末尾拼接 eos_token_id
        completion_ids = [
            torch.cat([ids, torch.tensor([eos_token_id], device=ids.device)]) if isinstance(ids, torch.Tensor)
            else ids + [eos_token_id]
            for ids in completion_ids
        ]

        completion_ids = pad(completion_ids, padding_value=0) #取最长的长度[B, 420]
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1) #[B*G, P+C] [B,21+420]
        labels = prompt_completion_ids.clone()
        labels[:, :prompt_ids.shape[1]] = -1  # Ignore the question part in labels
        prompt_completion_ids_mask = prompt_completion_ids.ge(0)
        label_mask = labels.ge(0)
        completion_mask = completion_ids.ge(0)
        prompt_completion_ids[~prompt_completion_ids_mask] = 0
        labels[~label_mask] = 0 

        logits_to_keep = completion_ids.size(1)

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model,
                    prompt_completion_ids,
                    labels,
                    ecg,
                    cxr,
                    lab,
                    logits_to_keep,
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                      self.ref_model,
                      prompt_completion_ids,
                      labels,
                      ecg,
                      cxr,
                      lab,
                      logits_to_keep,
                    )

        # Compute the rewards
        rewards_per_func = torch.zeros(
            len(questions), len(self.reward_funcs), device=device
        )
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
           
            output_reward_func = reward_func(
                completions, ground_truths
            )
            rewards_per_func[:, i] = torch.tensor(
                output_reward_func, dtype=torch.float32, device=device
            )
        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        # print('advantages',advantages)

        # Log the metrics
        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(
                reward_per_func[i].item()
            )

        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())

        return {
            "prompt_ids": prompt_ids, #[B,21]
            "completion_ids": completion_ids, #[B,420]
            "labels": labels, #[B,441]
            "completion_mask": completion_mask, #[B, 420]
            'cxr': cxr, #[B,3,224,224]
            'ecg': ecg, #[B, 12,1000]
            'lab': lab,#[B,100]
            "ref_per_token_logps": ref_per_token_logps, #[B, 420]
            "advantages": advantages, #[B]
        }

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        prompt_ids = inputs["prompt_ids"]
        completion_ids = inputs["completion_ids"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        labels = inputs["labels"]
        ecg = inputs["ecg"]
        cxr = inputs["cxr"]
        lab = inputs["lab"]
        completion_mask = inputs["completion_mask"]
        logits_to_keep = completion_ids.size(1)
        self.model.train()
        per_token_logps = self._get_per_token_logps(
            model,
            input_ids,
            labels,
            ecg,
            cxr,
            lab,
            logits_to_keep,
        )
    
        # Compute the KL divergence between the model and the reference model
        ref_per_token_logps = inputs["ref_per_token_logps"]
        per_token_kl = (
            torch.exp(ref_per_token_logps - per_token_logps)
            - (ref_per_token_logps - per_token_logps)
            - 1
        )

        # x - x.detach() allows for preserving gradients from x
        advantages = inputs["advantages"]
        per_token_loss = torch.exp(
            per_token_logps - per_token_logps.detach()
        ) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = (
            (per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
        ).mean()

        # Log the metrics
        completion_length = (
            self.accelerator.gather_for_metrics(completion_mask.sum(1))
            .float()
            .mean()
            .item()
        )
        self._metrics["completion_length"].append(completion_length)

        mean_kl = (
            (per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
        ).mean()
        self._metrics["kl"].append(
            self.accelerator.gather_for_metrics(mean_kl).mean().item()
        )

        return loss

        
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()