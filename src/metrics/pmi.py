import copy
import json
import sys

import torch
import torch.nn.functional as F
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from src.constants import IGNORE_INDEX

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


def load_model_amd_tokenizer(model_path, args):
    model = AutoModelForCausalLM.from_pretaiend(
        model_path,
        torch_dtype=TORCH_DTYPES[args.torch_dtype],
        cache_dir=args.cache_dir,
        token=args.hf_access_token,
    )
    tokenizer = AutoModelForCausalLM.from_pretrained(
        model_path, cache_dir=args.cache_dir, token=args.hf_access_token
    )
    return model, tokenizer


def preprocess_dataset(raw_dataset, tokenizer, args, max_seq_length=4096):
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


# test dataset name :  t0-task-sciq_Multiple_Choice.json

d = load_dataset("data/t0/t0-task-sciq_Multiple_Choice.json")
model, tokenizer = load_model_amd_tokenizer(
    "models_ft/result_gpt2_t0-task-sciq_Multiple_Choice"
)
print(preprocess_dataset(d, tokenizer, None))


def compute_label_probabilities(
    model,
    tokenizer,
    dataset,
    device,
    label_field="targets",
    text_field="inputs",
    epsilon=1e-12,
):
    """
    For a given dataset, compute for each sample the probability assigned by the model
    to the ground-truth label.
    """
    model.eval()
    label_probs = []

    with torch.no_grad():
        for sample in tqdm(dataset, desc="Computing label probabilities"):
            inputs = tokenizer(
                sample[text_field], return_tensors="pt", padding=True, truncation=True
            ).to(device)
            input_ids = inputs["input_ids"]
            prompt_ids = tokenizer(text_field, return_tensors="pt")["input_ids"].to(
                device
            )
            prompt_len = min(prompt_ids.shape[1], input_ids.shape[1])

            label_token_ids = input_ids[0, prompt_len:]

            logits = model(**inputs).logits

            label_logits = logits[0, prompt_len - 1 : -1]

            probs = F.softmax(logits, dim=-1)
            token_probs = probs[range(len(label_token_ids)), label_token_ids]

            clamped_probs = token_probs.clamp(min=epsilon)

            label = sample[label_field]

            log_prob = torch.sum(torch.log(clamped_probs))
            avg_log_prob = log_prob / len(label_token_ids)

            print(avg_log_prob)
            sys.exit(1)
            label_probs.append(avg_log_prob.exp().item())

    return label_probs


def compute_similarity_score(probs_A_on_B, probs_B_on_B, probs_B_on_A, probs_A_on_A):
    """
    Computes the PMI similarity score between tasks using the probabilities for the ground-truth labels.

    For samples from dataset D_Tj:
      term1 = (1/n) sum[ log( p_A(y|x) / p_B(y|x) ) ]

    For samples from dataset D_Ti:
      term2 = (1/m) sum[ log( p_B(y|x) / p_A(y|x) ) ]

    Finally:
      S_AB = 0.5 * (term1 + term2)
    """
    n = len(probs_A_on_B)
    m = len(probs_B_on_A)

    term1 = (
        sum(
            torch.log(torch.tensor(pA) / torch.tensor(pB))
            for pA, pB in zip(probs_A_on_B, probs_B_on_B)
        )
        / n
    )
    term2 = (
        sum(
            torch.log(torch.tensor(pB) / torch.tensor(pA))
            for pA, pB in zip(probs_A_on_A, probs_B_on_A)
        )
        / m
    )

    S_AB = 0.5 * (term1 + term2)
    return S_AB.item()
