import json
import os
import sys
from itertools import combinations

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Use the EOS token as pad if needed
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_dataset(jsonl_path):
    with open(jsonl_path) as f:
        return json.load(f)["train"]


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


device = "cuda" if torch.cuda.is_available() else "cpu"

# Generate all unique ordered pairs of tasks (task_A, task_B) where task_A != task_B.
tasks = []
dataset_paths = []
model_paths = []
base_model_dir = "models_ft"
base_data_dir = "data"
base_model_name = "gpt2"
# result_gpt2_t0-task-wiqa_does_the_supposed_perturbation_have_an_effect

for task in os.listdir("models_ft"):
    tasks.append(task)
    model_paths.append(os.path.join(base_model_dir, task))
    dataset_paths.append(
        os.path.join(
            base_data_dir,
            task[len(f"result_{base_model_name}_") : task.find("-")],
            task[len(f"result_{base_model_name}_") :],
        )
        + ".json"
    )

task_pairs = list(combinations([i for i in range(len(tasks))], 2))

for task_A, task_B in task_pairs:
    print(f"\nProcessing pair: {task_A} - {task_B}")

    # Retrieve corresponding paths for models and datasets.
    model_path_A = model_paths[task_A]
    model_path_B = model_paths[task_B]
    dataset_path_A = dataset_paths[task_A]
    dataset_path_B = dataset_paths[task_B]

    if model_path_A and model_path_B and dataset_path_A and dataset_path_B:
        # Load model A and its tokenizer, then its dataset.
        model_A, tokenizer_A = load_model_and_tokenizer(model_path_A)
        model_A.to(device)
        dataset_A = load_dataset(dataset_path_A)

        # Load model B and its tokenizer, then its dataset.
        model_B, tokenizer_B = load_model_and_tokenizer(model_path_B)
        model_B.to(device)
        dataset_B = load_dataset(dataset_path_B)

        # For dataset_B (task T_j), compute:
        #   p_A(y^Tj|x) using model A and p_B(y^Tj|x) using model B.
        probs_A_on_B = compute_label_probabilities(
            model_A, tokenizer_A, dataset_B, device
        )
        probs_B_on_B = compute_label_probabilities(
            model_B, tokenizer_B, dataset_B, device
        )

        # For dataset_A (task T_i), compute:
        #   p_A(y^Ti|x) using model A and p_B(y^Ti|x) using model B.
        probs_A_on_A = compute_label_probabilities(
            model_A, tokenizer_A, dataset_A, device
        )
        probs_B_on_A = compute_label_probabilities(
            model_B, tokenizer_B, dataset_A, device
        )

        # Compute the PMI similarity score following the defined equation.
        S_AB = compute_similarity_score(
            probs_A_on_B, probs_B_on_B, probs_B_on_A, probs_A_on_A
        )
        print(f"PMI Similarity Score (S_AB) for pair {task_A} - {task_B}: {S_AB:.4f}")
    else:
        print(f"Missing paths for pair {task_A} - {task_B}")
