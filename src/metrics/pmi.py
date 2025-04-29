import gc
import os
from copy import deepcopy

import multiprocess as mp
import numpy as np
import torch
from multiprocess import Pool
from torch.utils.data import DataLoader

from src.preprocess.dataset import (
    DataCollatorForInstructionTuning,
    load_dataset,
    preprocess_dataset,
)
from src.utils import list_all_models_and_datasets
from src.vllm import load_vllm_model_and_tokenizer
from vllm import SamplingParams

# from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)
from vllm.inputs import TokensPrompt

IGNORE_INDEX = -100


# test dataset name :  t0-task-sciq_Multiple_Choice.json
def compute_label_probabilities(
    llm,
    tokenizer,
    dataset,
):
    """
    For a given dataset, compute for each sample the probability assigned by the model
    to the ground-truth label.
    """
    label_probs = []
    sampling_params = SamplingParams(prompt_logprobs=0, max_tokens=1)

    for sample in dataset:
        batch_input_ids = []
        batch_labels = sample["labels"]
        for input_ids in sample["input_ids"]:
            batch_input_ids.append(
                TokensPrompt(prompt_token_ids=input_ids)
            )  # shape = (batches, seq_length)

        outputs = llm.generate(
            batch_input_ids, sampling_params=sampling_params, use_tqdm=False
        )

        for i, output in enumerate(outputs):
            total_log_prob = 0
            for j, is_valid in enumerate(batch_labels[i] != IGNORE_INDEX):
                if is_valid:
                    if output.prompt_logprobs and output.prompt_logprobs[j]:
                        total_log_prob += list(output.prompt_logprobs[j].values())[
                            0
                        ].logprob
                    else:
                        continue

            label_probs.append(np.exp(total_log_prob))

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

    eps = 1e-12
    safe_log = lambda x: torch.log(torch.tensor(max(x, eps)))
    term1 = (
        sum(safe_log(pA) - safe_log(pB) for pA, pB in zip(probs_A_on_B, probs_B_on_B))
        / n
    )
    term2 = (
        sum(safe_log(pB) - safe_log(pA) for pA, pB in zip(probs_A_on_A, probs_B_on_A))
        / m
    )

    S_AB = 0.5 * (term1 + term2)
    return S_AB.item()


AVAILABLE_GPUS = [1, 2, 3, 4, 6, 7]


def worker_task_dataset(args):
    task, n, _dataset_loaders, model_paths, map_task_dataset = args
    process_id = mp.current_process()._identity[0]
    dataset_loaders = _dataset_loaders[process_id % len(AVAILABLE_GPUS)]
    device = AVAILABLE_GPUS[process_id % len(AVAILABLE_GPUS)]

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print(device)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    llm, tokenizer = load_vllm_model_and_tokenizer(model_paths[task])

    for dataset in range(n):
        if os.path.exists(f"pairwise_npy/prob_arr_{task}-{dataset}.npy"):
            map_task_dataset[(task, dataset)] = np.load(
                f"pairwise_npy/prob_arr_{task}-{dataset}.npy"
            )
            print(f"Loaded {task}-{dataset}")
            continue

        pd = dataset_loaders[dataset]
        tmp_probs = compute_label_probabilities(llm, tokenizer, pd)
        map_task_dataset[(task, dataset)] = tmp_probs
        np.save(f"pairwise_npy/prob_arr_{task}-{dataset}.npy", np.array(tmp_probs))
        print(f"Done {task}-{dataset}")

    destroy_model_parallel()
    destroy_distributed_environment()
    del llm.llm_engine.model_executor
    del llm
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    # torch.distributed.destroy_process_group()
    # import ray
    # ray.shutdown()

    return map_task_dataset


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    subtask_types, model_paths, dataset_paths = list_all_models_and_datasets(
        checkpoint_folder="checkpoints/full_ft"
    )

    n = len(model_paths)

    # model_paths=model_paths[:4]
    # dataset_paths=dataset_paths[:10]
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    dataset_loaders = {}
    llm, tokenizer = load_vllm_model_and_tokenizer(model_paths[0])
    data_collator = DataCollatorForInstructionTuning(tokenizer)
    # dataset_loaders = manager.dict()
    workers = len(AVAILABLE_GPUS)
    per_process_dataset_loaders = [{}] * workers
    for dataset in range(n):
        d = load_dataset(dataset_paths[dataset], splits=["validation"])
        pd1 = preprocess_dataset(d, tokenizer, None)
        for i in range(workers):
            per_process_dataset_loaders[i][dataset] = DataLoader(
                pd1,
                shuffle=False,
                pin_memory=True,
                collate_fn=data_collator,
                batch_size=1024,
                num_workers=8,
            )

    process_pool = []

    task_batch_size = n // workers

    for i in range(workers):
        per_process_dataset_loaders.append(deepcopy(dataset_loaders))

    with mp.Manager() as manager:
        map_task_dataset = manager.dict()

        del llm.llm_engine.model_executor.driver_worker
        del llm
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        torch.distributed.destroy_process_group()
        import ray

        ray.shutdown()

        workload_tasks = [
            (task, n, per_process_dataset_loaders, model_paths, map_task_dataset)
            for task in range(n)
        ]
        with Pool(workers) as pool:
            pool.imap_unordered(worker_task_dataset, workload_tasks)
            pool.close()
            pool.join()

        sim_mat = np.ones((n, n))

        for task1 in range(n):
            for task2 in range(task1, n):
                try:
                    sim_mat[task1, task2] = compute_similarity_score(
                        map_task_dataset[(task1, task2)],
                        map_task_dataset[(task2, task2)],
                        map_task_dataset[(task2, task1)],
                        map_task_dataset[(task1, task1)],
                    )
                except:
                    print(f"Skipping {task1}-{task2}")
        # print(sim_mat)
        np.save("mp_similarity_mat_pmi.py", sim_mat)
