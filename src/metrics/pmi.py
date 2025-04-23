from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.preprocess.dataset import (
    DataCollatorForInstructionTuning,
    load_dataset,
    load_model_and_tokenizer,
    preprocess_dataset,
)
from src.utils import list_all_models_and_datasets


# test dataset name :  t0-task-sciq_Multiple_Choice.json
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
            labels = sample["labels"]
            valid_mask = labels != -100
            output = model(**sample)
            logits = output.logits
            log_probs = F.log_softmax(logits, dim=-1)
            safe_labels = torch.where(valid_mask, labels, torch.zeros_like(labels))
            selected_log_probs = log_probs.gather(
                dim=2, index=safe_labels.unsqueeze(-1)
            ).squeeze(-1)
            selected_log_probs = selected_log_probs * valid_mask
            total_log_probs = selected_log_probs.sum(dim=1)
            label_probs.extend(total_log_probs.exp())

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


subtask_types, model_paths, dataset_paths = list_all_models_and_datasets(checkpoint_folder="checkpoints/full_ft")
map_task_dataset = defaultdict(lambda: defaultdict(list))

for task in range(len(model_paths)):
    model, tokenizer = load_model_and_tokenizer(model_paths[task], None)
    data_collator = DataCollatorForInstructionTuning(tokenizer)
    for dataset in range(len(dataset_paths)):
        dataset_path = dataset_paths[dataset]
        print("Model : ", model_paths[task], "Dataset : " , dataset_path, end  = " = ")
        d = load_dataset(dataset_path)
        pd = preprocess_dataset(d, tokenizer, None)
        train_dataloader = DataLoader(
            pd,
            shuffle=False,
            collate_fn=data_collator,
            batch_size=3,
            pin_memory=True,
            num_workers=8,
        )

        map_task_dataset[task][dataset] = compute_label_probabilities(
            model, tokenizer, train_dataloader, "cuda:2"
        )
        print("Done")

similarity_mat = np.ones((len(model_paths), len(model_paths)))

for task1 in range(len(model_paths)):
    for task2 in range(len(model_paths)):
        if task1 == task2:
            continue
        similarity_mat[task1, task2] = compute_similarity_score(
            map_task_dataset[task1][task2],
            map_task_dataset[task2][task2],
            map_task_dataset[task2][task1],
            map_task_dataset[task1][task1],
        )

np.save("similarity_mat.npy", similarity_mat)
