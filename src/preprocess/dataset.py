import copy
import json
from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import Dataset
from torch.nn.utils import rnn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from src.constants import IGNORE_INDEX, TORCH_DTYPE

"""
Returns back array of data samples
[
    {
        "inputs" : "",
        "targets" : ""
    }
]
"""


def load_dataset(dataset_path, splits=["train"]):
    with open(dataset_path, "r") as f:
        data = json.load(f)

    combined_data = []
    for split in splits:
        combined_data.extend(data[split])

    return Dataset.from_list(combined_data)


def load_model_and_tokenizer(model_path, args):
    model = AutoModelForCausalLM.from_pretaiend(
        model_path,
        torch_dtype=TORCH_DTYPE[args.torch_dtype],
        cache_dir=args.cache_dir,
        token=args.hf_access_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, cache_dir=args.cache_dir, token=args.hf_access_token
    )
    return model, tokenizer


def preprocess_dataset(raw_dataset, tokenizer, args, max_seq_length=1024):
    raw_dataset_column_names = raw_dataset.column_names

    def preprocess_function(examples):
        prompts_responses = [
            p + " " + r for p, r in zip(examples["inputs"], examples["targets"])
        ]
        prompts_responses_tokenized = tokenizer(
            prompts_responses, truncation=True, max_length=max_seq_length
        )
        prompts_tokenized = tokenizer(
            examples["inputs"], truncation=True, max_length=max_seq_length
        )
        all_labels = copy.deepcopy(prompts_responses_tokenized["input_ids"])
        prompts_len = [len(prompt) for prompt in prompts_tokenized["input_ids"]]
        for labels, prompt_len in zip(all_labels, prompts_len):
            labels[:prompt_len] = [IGNORE_INDEX] * prompt_len
        result = {k: v for k, v in prompts_responses_tokenized.items()}
        result["labels"] = all_labels
        return result

    return raw_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=12,  # args.preprocessing_num_workers,
        load_from_cache_file=False,  # not args.overwrite_cache,
        remove_columns=raw_dataset_column_names,
        desc="Preprocessing the raw dataset",
    )


@dataclass
class DataCollatorForInstructionTuning:
    """Collate examples for instruction tuning."""

    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, attention_mask, labels = tuple(
            [torch.tensor(feature[key]) for feature in features]
            for key in ["input_ids", "attention_mask", "labels"]
        )
        input_ids = rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        labels = rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
