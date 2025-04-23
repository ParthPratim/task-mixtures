import multiprocessing
from collections import defaultdict
from multiprocessing import Pool, shared_memory

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


subtask_types, model_paths, dataset_paths = list_all_models_and_datasets(
    checkpoint_folder="checkpoints/full_ft"
)
NUM_WORKERS = 8
AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]

subtask_types, model_paths, dataset_paths = list_all_models_and_datasets()

# inter-process cache
ip_cache = shared_memory.SharedMemory(name="inter-worker-cache", create=True, size=100)
cache_buf = ip_cache.buf
cache_bug = {}
cache_buf["map_task_dataset"] = defaultdict(lambda: defaultdict(list))
cache_buf["model_tokenizer_cache"] = {}
cache_buf["dataset_cache"] = {}
cache_buf["data_collator"] = None
cache_buf["similarity_mat"] = np.ones((len(model_paths), len(model_paths)))

buffer_stores = {}


def workload_task_dataset(task, dataset):
    process_id = multiprocessing.current_process()._identity[0]
    assigned_gpu = AVAILABLE_GPUS[process_id % len(AVAILABLE_GPUS)]
    if process_id in buffer_stores:
        buffer_stores[process_id] = shared_memory.SharedMemory(
            name="inter-worker-cache"
        ).buf

    sh_cache = buffer_stores[process_id]

    if (task, dataset) not in sh_cache["model_tokenizer_cache"].keys():
        sh_cache["model_tokenizer_cache"][(task, dataset)] = load_model_and_tokenizer(
            model_paths[task], None
        )

    model, tokenizer = sh_cache["model_tokenizer_cache"][(task, dataset)]

    if sh_cache["data_collator"] is None:
        sh_cache["data_collator"] = DataCollatorForInstructionTuning(tokenizer)

    data_collator = sh_cache["data_collator"]

    if dataset not in sh_cache["dataset_cache"]:
        sh_cache["dataset_cache"][dataset] = load_dataset(dataset_paths[dataset])

    d = sh_cache["dataset_cache"][dataset]

    pd = preprocess_dataset(d, tokenizer, None)

    train_dataloader = DataLoader(
        pd,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=3,
        pin_memory=True,
        num_workers=8,
    )

    sh_cache["map_task_dataset"][task][dataset] = compute_label_probabilities(
        model, tokenizer, train_dataloader, f"cuda:{assigned_gpu}"
    )


def workload_pmi(task1, task2):
    if task1 == task2:
        return

    process_id = multiprocessing.current_process()._identity[0]

    if process_id in buffer_stores:
        buffer_stores[process_id] = shared_memory.SharedMemory(
            name="inter-worker-cache"
        ).buf

    sh_cache = buffer_stores[process_id]
    map_task_dataset = sh_cache["map_task_dataset"]
    sh_cache["similarity_mat"][task1, task2] = compute_similarity_score(
        map_task_dataset[task1][task2],
        map_task_dataset[task2][task2],
        map_task_dataset[task2][task1],
        map_task_dataset[task1][task1],
    )


"""
Data Parallelism : Run a pool of 8 processes to 
"""
with Pool(NUM_WORKERS) as pool:
    task_data_pair_loadgen = [
        (task, dataset)
        for task in range(len(model_paths))
        for dataset in range(len(dataset_paths))
    ]
    task_task_pair_loadgen = [
        (task1, task2)
        for task1 in range(len(model_paths))
        for task2 in range(len(model_paths))
    ]
    pool.map(workload_task_dataset, task_data_pair_loadgen)
    pool.map(workload_pmi, task_task_pair_loadgen)


np.save("similarity_mat.npy", cache_buf["similarity_mat"])

for cbuf in buffer_stores.values():
    cbuf.close()

cache_buf.close()
cache_buf.unlink()
