import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.preprocess.dataset import (
    DataCollatorForInstructionTuningOnDevice,
    load_dataset,
    preprocess_dataset,
)
import os
import traceback
import multiprocess as mp
import gc
from copy import deepcopy
from multiprocess import Process, Pool
import numpy as np
from src.utils import list_all_models_and_datasets, load_model_and_tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

"""
def load_vllm_model_and_tokenizer(model_path, device=0):
    model = AutoModelForCausalLM.from_pretrained(model_path).to(f"cuda:{device}").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer
"""
IGNORE_INDEX = -100

def compute_label_probabilities(
    model,
    dataset,
    device,
):
    """
    For a given dataset, compute for each sample the probability assigned by the model
    to the ground-truth label.
    """
    label_probs = []

    llm  = model
    
    with torch.no_grad():
        for sample in dataset:
            #sample = {k: v.to(f"cuda:{device}") for k, v in sample.items()}
            try:
                if device == "cpu": 
                    model = torch.jit.optimize_for_inference(torch.jit.trace(llm, (sample['input_ids'], )))

            except Exception:
                traceback.print_exc()
    

            outputs = model(input_ids=sample['input_ids']) 
            log_probs = F.log_softmax(outputs.logits, dim=-1) # (batch_size, seq_length, vocab_size)
            labels = sample["labels"]
            valid_mask = labels != IGNORE_INDEX
            batch_indices = torch.arange(log_probs.shape[0], device=log_probs.device).unsqueeze(1)  # shape: [batch_size, 1]
            seq_indices = torch.arange(log_probs.shape[1], device=log_probs.device).unsqueeze(0)  # shape: [1, seq_length]
            log_probs_selected = log_probs[batch_indices, seq_indices, labels]
            log_probs_selected = log_probs_selected * valid_mask.float()  # shape: [batch_size, seq_length
            sum_log_probs_selected = torch.sum(log_probs_selected, dim=1)
            label_probs.extend(torch.exp(sum_log_probs_selected))
        


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

    eps = 1e-10
    safe_log = lambda x: torch.log(torch.tensor(max(x, eps)))
    term1 = (
        sum(
            safe_log(pA) - safe_log(pB)
            for pA, pB in zip(probs_A_on_B, probs_B_on_B)
        )
        / n
    )
    term2 = (
        sum(
            safe_log(pB) - safe_log(pA)
            for pA, pB in zip(probs_A_on_A, probs_B_on_A)
        )
        / m
    )

    S_AB = 0.5 * (term1 + term2)
    return S_AB.item()

AVAILABLE_GPUS = [0,1,2]
NUM_CPU_WORKERS = 0


def worker_task_dataset(task,i, j,_dataset_loaders, model_paths, map_task_dataset, named_devices):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    process_id = mp.current_process()._identity[0]
    dataset_loaders = _dataset_loaders[process_id % len(named_devices)] 
    device = named_devices[process_id % len(named_devices)]

    if device == "cpu":
        torch.set_num_threads(64)
    
    
    print("Loaded model ",model_paths[task], " on ", device)
    llm, _ = load_model_and_tokenizer(model_paths[task], device)



    for dataset in range(i,j):
        
        if os.path.exists(f"pairwise_npy/prob_arr_{task}-{dataset}.npy"):
            #map_task_dataset[(task,dataset)] = np.load(f"pairwise_npy/prob_arr_{task}-{dataset}.npy")
            print(f"Loaded {task}-{dataset}")
            continue
    
        pd = dataset_loaders[dataset]
        tmp_probs = torch.tensor(compute_label_probabilities(llm, pd, device)).cpu().numpy()
        np.save(f"pairwise_npy/prob_arr_{task}-{dataset}.npy", np.array(tmp_probs))
        print(f"Done {task}-{dataset} ", device)

    return map_task_dataset


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    subtask_types, model_paths, dataset_paths = list_all_models_and_datasets(
        checkpoint_folder="checkpoints/full_ft"
    )
    
    for i in range(6):
        model_paths.extend(model_paths)
        dataset_paths.extend(dataset_paths)
    
    n = len(model_paths)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # dataset_loaders = manager.dict()
    
    named_devices = [ f"cuda:{i}" for  i in AVAILABLE_GPUS ]
    named_devices += ["cpu"] * NUM_CPU_WORKERS

    workers = len(named_devices)

    print("Total Workers : ", workers)
   
    DATASET_BATCH_SIZE = 5
    for di in range(0,n,DATASET_BATCH_SIZE):
    
        llm, tokenizer = load_model_and_tokenizer(model_paths[0], named_devices[0])

        per_process_dataset_loaders = {}
        for dataset in range(di, di+DATASET_BATCH_SIZE):       
            try: 
                d = load_dataset(dataset_paths[dataset], splits=["validation"])
                pd1 = preprocess_dataset(d, tokenizer, None) 
                for i in range(workers):
                    data_collator = DataCollatorForInstructionTuningOnDevice(tokenizer, named_devices[i])
                    if i not in per_process_dataset_loaders:
                        per_process_dataset_loaders[i] = {}
                    per_process_dataset_loaders[i][dataset] = DataLoader(
                            pd1,
                            shuffle=False,
                            pin_memory=False,
                            collate_fn=data_collator,
                            batch_size=4056 if named_devices[i] ==  "cpu" else 256
                    )
            except : 
                print("Failed : ")

        del llm
        del tokenizer
        gc.collect()

        task_batch_size = n // workers
        with mp.Manager() as manager:
            per_process_dataset_loaders = manager.dict(per_process_dataset_loaders)
            map_task_dataset = manager.dict()
         
            workload_tasks = [ (task, di, di+DATASET_BATCH_SIZE, per_process_dataset_loaders, model_paths, map_task_dataset, named_devices)  for task in range(n) ] 
            
            pool = Pool(workers)

            #for w_task in workload_tasks:
                    #pool.imap_unordered(worker_task_dataset, workload_tasks)
            pool.starmap(worker_task_dataset, workload_tasks)

            pool.close()
            pool.join()
        
        del per_process_dataset_loaders
        gc.collect()

       
    sim_mat = np.ones((n,n))

    for task1 in range(n):
        for task2 in range(task1, n):
            
            sim_mat[task1,task2] = compute_similarity_score(map_task_dataset[(task1,task2)], 
                        map_task_dataset[(task2,task2)],
                        map_task_dataset[(task2,task1)], 
                        map_task_dataset[(task1,task1)])

            #except:
            #    print(f"Skipping {task1}-{task2}")
    #print(sim_mat)
    np.save("mp_similarity_mat_pmi", sim_mat)