import multiprocessing as mp
from  multiprocessing import Pool
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import pickle as pkl
#import gc
from src.mixtures.base import TaskMeta
from src.mixtures.uniform import Multinomial, Uniform, GlobalUniform
from src.opti.belief_prop import BeliefPropagation
from src.opti.quad_cvx import QuadraticConvexOptimization, GraphLaplacianOptimization
# from src.opti.search import GridSearch
from src.opti.cluster import ClusteredOptimization
#from src.opti.search import GridSearch
from src.preprocess.dataset import load_dataset
from functools import cmp_to_key
import re
import json


def process_job(args):
    prefix, subtask, idx = args
    print(subtask) 
    train_instances = list(
        load_dataset(os.path.join(prefix, subtask), splits=["train"])
    )
    print(f"Done Loading {subtask}")
    return TaskMeta(subtask, train_split=train_instances)


def load_general_tasks(
    use_orderring=True,
    orderring_file="artifacts/task-index-maps/3epochs-t0-flan2021-cot-tulu-sglue.csv",
    num_proc=64,
):
    mixture_folders = ["data/t0", "data/flan2021",
                       "data/cot", "data/tulu", "data/sglue"]
    subtask_jsons = []
    subtask_metas = []
    order_index = {}

    subtasks_list = []
    for data_folder in mixture_folders:
        subtask_jsons.append((data_folder, os.listdir(data_folder)))

    workload = []
    i = 0
    for prefix, subtasks in subtask_jsons:
        for subtask in subtasks:
            if subtask[-1] != 'n':
                continue
            workload.append((prefix, subtask, i))
            subtasks_list.append(subtask)
            if use_orderring:
                order_index[subtask] = i
            i += 1

    with Pool(num_proc) as pool:
        subtask_metas = list(pool.imap(process_job, workload))
        pool.close()
        pool.join()


    if use_orderring:
        order = pd.read_csv(orderring_file)
        prefix = "result_gpt2_"
        for index, row in order.iterrows():
            task_name = row["Task-Name"][len(prefix):]
            if task_name not in order_index or order_index[task_name] == index:
                continue

            tmp = subtask_metas[index]
            subtask_metas[index] = subtask_metas[order_index[task_name]]
            subtask_metas[order_index[task_name]] = tmp
            order_index[tmp.task_name] = order_index[task_name]
            order_index[task_name] = index

    return subtasks_list, subtask_metas


"""
Experiment 1
--------------------
Corresponds to using the following listed submixtures :
    1. flan2021
    2. T0
    3. CoT

To create a uniform submixture with 25K final submixture instances

Output : JSON with 25K instances uniformly sampled, with equal budget split among tasks
"""


def experiment_1(NUM_INSTANCES=25000):
    subtasks_list, subtask_metas = load_general_tasks()
    uniform = Uniform(
        subtasks_list,
        subtask_metas,
        NUM_INSTANCES,
        "artifacts/final-submixtures/25K-uniform-t0-flan2021-cot-all-full",
    )

    uniform.create_mixture()
    uniform.dump_mixture(f"{uniform.mixture_name}.json")

def experiment_1_1(NUM_INSTANCES=25000):
    subtasks_list, subtask_metas = load_general_tasks()
    uniform = GlobalUniform(
        subtasks_list,
        subtask_metas,
        NUM_INSTANCES,
        "artifacts/final-submixtures/25K-random-t0-flan2021-cot-tulu-sglue",
    )

    uniform.create_mixture()
    uniform.dump_mixture(f"{uniform.mixture_name}.json")



"""
Experiment 2
------------------------
1. Create task probability distribution vector from PMI matrix
2. Select instances using Mutinomial Distribution
"""


def experiment_2(
    sim_npy="artifacts/similarity-matrix/3epochs-t0-flan2021-cot-tulu-sglue.npy",
    NUM_INSTANCES=25000,
):
    #subtasks_list, subtask_metas = load_general_tasks()

    # Load PMI Matrix
    S = np.load(sim_npy)

    quad_cvx = QuadraticConvexOptimization(S)
    task_prob = quad_cvx.closed_form_task_probs(1, 50)
    print(np.sum(task_prob))
    print(np.count_nonzero(task_prob))
    plot_prob_dist(task_prob, 'test.png')
    return 
    multinomial = Multinomial(
        subtasks_list,
        subtask_metas,
        NUM_INSTANCES,
        "artifacts/final-submixtures/25K-opti-closed-10-20-to-flan2021-cot-tulu-sglue",
        task_prob=task_prob,
    )

    multinomial.create_mixture()

    multinomial.dump_mixture(f"{multinomial.mixture_name}.json")


"""
Experiment 3
------------------------
1. Create task probability distribution vector from PMI matrix
2. Select instances using Mutinomial Distribution
"""


def experiment_3(
    sim_npy="artifacts/similarity-matrix/2epochs-t0-flan2021-cot.npy",
    NUM_INSTANCES=25000,
):
    subtasks_list, subtask_metas = load_general_tasks()

    # Load PMI Matrix
    S = np.exp(np.load(sim_npy))

    bp = BeliefPropagation(S)
    task_prob = bp.compute_task_probability(_beta=10.0, _lambda=15.0, T=1000)

    multinomial = Multinomial(
        subtasks_list,
        subtask_metas,
        NUM_INSTANCES,
        "artifacts/final-submixtures/25K-bp1",
        task_prob=task_prob,
    )

    multinomial.create_mixture()

    multinomial.dump_mixture(f"{multinomial.mixture_name}.json")


"""
Optimization Function Test
"""

def plot_prob_dist(n, filename):
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(n)), n, tick_label=[f"P{i}" for i in range(len(n))])
    plt.ylabel("Probability")
    plt.title("Probability Distribution")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(filename)


def experiment_4(sim_npy="artifacts/similarity-matrix/3epochs-t0-flan2021-cot-tulu-sglue.npy"):
    S = np.load(sim_npy)

    
    # S = (S - np.mean(S) ) / (np.std(S) + 1e-8)
    # S = (S - S.min()) / (S.max() - S.min() + 1e-8)
    S = 0.5 * (S + S.T)

    dim = S.shape[0]
    print("DImension of S is: ",dim)
    bp = QuadraticConvexOptimization(S)
    n = np.array(bp.compute_task_probability(beta=dim, lambda_=1.0))
    print(n)

    # n[n <= 1e-8] = 0

    print("Total probability:", np.sum(n))
    print("NZ : ", np.count_nonzero(n))
    print(n)
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(n)), n, tick_label=[f"P{i}" for i in range(len(n))])
    plt.ylabel("Probability")
    plt.title("Probability Distribution")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

def experiment_5_1(sim_npy="artifacts/similarity-matrix/3epochs-t0-flan2021-cot-tulu-sglue.npy"):
    S = np.load(sim_npy)
    
    # Ensure S is symmetric
    #S = 0.5 * (S + S.T)
    dim = S.shape[0]

    #np.random.seed(42)
    #S = np.random.rand(dim, dim)  # Random values between 0 and 1
    S = 0.5 * (S + S.T)  # Ensure symmetry
    #S += np.eye(dim)  # Add identity to make it diagonally dominant (PSD)

    
    print("Dimension of S is: ", dim)
    # Define expanded ranges for beta and lambda
    beta_values = [1.0, 5.0, 10.0, 20.0, 50.0, 100, 150 , dim, 500, 1000]
    lambda_values = [0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 200, 500, 800,1000]


    plt.figure(figsize=(14, 8))

    for beta in beta_values:
        for lambda_ in lambda_values:
            print(beta, lambda_)
            try:
            # Instantiate the optimizer and compute probabilities
                bp = QuadraticConvexOptimization(S)
                n = np.array(bp.compute_task_probability(_beta=beta, _lambda=lambda_))
                print(np.count_nonzero(n))
            
                # Plot the probability distribution as a line plot
                plt.plot(range(len(n)), n, label=f"β={beta}, λ={lambda_}")
            except:
                pass

    # Add plot details
    plt.xlabel("Task Index")
    plt.ylabel("Probability")
    plt.title("Probability Distributions for Different β and λ Combinations")
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("test.png")
    
"""
def experiment_5(sim_npy="artifacts/similarity-matrix/2epochs-t0-flan2021-cot.npy"):
    S  = np.exp(np.load(sim_npy))

    # p is the probability distribution
    def evaluator(p):
        p = np.array(p)
        eps = 1e-10
        total_excl = np.sum(p[p <= eps])
        p[p <= eps] = 0 
        non_zero_count = np.count_nonzero(p)
        return non_zero_count 

    return print(GridSearch(GraphLaplacianOptimization, S, beta_range=(-50,50),lambda_range=(-50,50),evaluator=evaluator))

"""

def experiment_5_2(
        sim_npy="artifacts/similarity-matrix/3epochs-t0-flan2021-cot-tulu-sglue.npy",
        orderring_file="artifacts/task-index-maps/3epochs-t0-flan2021-cot-tulu-sglue.csv",
        cluster_size=32,
        _beta = 0.25, _lambda = 1.25, NUM_INSTANCES=25000
        ):
    
    subtasks_list, subtask_metas = load_general_tasks()
    task_meta_file = "tasks.pkl"
    task_embs_file = "tasks_embeddings.npy"
    emb_folder_map = {
        "t0" : "artifacts/task_embeddings/t0_embed",
        "flan2021" : "artifacts/task_embeddings/flan2021_embed",
        "cot" : "artifacts/task_embeddings/cot_embed",
        "tulu" : "artifacts/task_embeddings/tulu_embed",
        "sglue" : "artifacts/task_embeddings/sglue_embed"
    }
    order = pd.read_csv(orderring_file)
    reverse_map = {}
    for _, row in order.iterrows():
        subtask = row["Task-Name"][len("result_gpt2_"):]
        if "t0" in subtask or "flan2021" in subtask or "cot" in subtask:
            subtask = subtask[subtask.index('-')+1:]
        subtask = subtask[subtask.index('-')+1:]
        reverse_map[subtask] = int(row['Task-ID'])

    S = np.load(sim_npy)
    task_embeddings = [ None ] * S.shape[0]

    embed_size = 0
    for task in emb_folder_map.keys():
        with open(os.path.join(emb_folder_map[task], task_meta_file), "rb") as f:
            task_meta = list(pkl.load(f))

        task_embs = np.load(os.path.join(emb_folder_map[task], task_embs_file))
 
        for i, subtask in enumerate(task_meta):
            print(task, subtask)
            _subtask = re.sub(r'[:/-]', '_',subtask)
            task_embeddings[reverse_map[_subtask]] = task_embs[i]
            embed_size = task_embs[i].shape[0]
        

    for idx, val in enumerate(task_embeddings):
        if val is None:
            task_embeddings[idx] = np.zeros((embed_size, )) 

    obj = ClusteredOptimization(S)
    
    cluster_task_map, cluster_prob = obj.compute_task_probability(task_embeddings=task_embeddings, 
                                 cluster_size=cluster_size,
                                 _beta=_beta,
                                 _lambda=_lambda)
    
    for i, p in enumerate(cluster_prob):
        if len(p) == 0:
            continue
        print(f"Non-Zero in cluster {i} = ", np.count_nonzero(p))
        plot_prob_dist(p, f"cluster_prob_dist_{i}.png")
    

    final_submixture = []
    submixture_filename = "artifacts/final-submixtures/25K-cluster-opti-cvx-32clusters-0_25-1_25-to-flan2021-cot"
    for cluster_idx, cluster_task_indices in cluster_task_map.items():
        task_budget = int(math.ceil(NUM_INSTANCES * ((len(cluster_task_indices)*1.0)  / S.shape[0])))
        multinomial = Multinomial(
            [subtasks_list[i] for i in cluster_task_indices],
            [subtask_metas[i] for i in cluster_task_indices],
            task_budget,
            submixture_filename,
            task_prob=cluster_prob[cluster_idx],
        )
        final_submixture.extend(multinomial.create_mixture().train_split)
    
    with open(submixture_filename+".json", "w") as f:
        json.dump(
            {
                "train": final_submixture,
                "validation": [],
            },
            f,
        )
    

if __name__ == "__main__":
    mp.set_start_method("spawn")
    # experiment_2()
    experiment_2()
