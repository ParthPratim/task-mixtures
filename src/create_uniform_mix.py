import multiprocessing as mp
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import gc
from src.mixtures.base import TaskMeta
from src.mixtures.uniform import Multinomial, Uniform
from src.opti.belief_prop import BeliefPropagation
from src.opti.quad_cvx import QuadraticConvexOptimization, GraphLaplacianOptimization
# from src.opti.search import GridSearch
from src.preprocess.dataset import load_dataset


def process_job(args):
    prefix, subtask, idx = args
    train_instances = list(
        load_dataset(os.path.join(prefix, subtask), splits=["train"])
    )
    print(f"Done Loading {subtask}")
    return TaskMeta(subtask, train_split=train_instances)


def load_general_tasks(
    use_orderring=True,
    orderring_file="artifacts/task-index-maps/3epochs-t0-flan2021-cot-tulu-sglue.csv",
    num_proc=2,
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
            workload.append((prefix, subtask, i))
            subtasks_list.append(subtask)
            if use_orderring:
                order_index[subtask] = i
            i += 1

    # with Pool(num_proc) as pool:
    subtask_metas = list(map(process_job, workload))

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


"""
Experiment 2
------------------------
1. Create task probability distribution vector from PMI matrix
2. Select instances using Mutinomial Distribution
"""


def experiment_2(
    sim_npy="artifacts/similarity-matrix/2epochs-t0-flan2021-cot.npy",
    NUM_INSTANCES=25000,
):
    subtasks_list, subtask_metas = load_general_tasks()

    # Load PMI Matrix
    S = np.load(sim_npy)

    quad_cvx = QuadraticConvexOptimization(S)
    task_prob = quad_cvx.compute_task_probability(10.0, 15.0)

    multinomial = Multinomial(
        subtasks_list,
        subtask_metas,
        NUM_INSTANCES,
        "artifacts/final-submixtures/25K-closed-form-multinomial-to-flan2021-cot",
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


def experiment_4(sim_npy="artifacts/similarity-matrix/3epochs-t0-flan2021-cot-tulu-sglue.npy"):
    S = np.load(sim_npy)
    
    
    # S = (S - np.mean(S) ) / (np.std(S) + 1e-8)
    # S = (S - S.min()) / (S.max() - S.min() + 1e-8)
    S = 0.5 * (S + S.T)

    dim = S.shape[0]
    print("DImension of S is: ",dim)
    bp = QuadraticConvexOptimization(S)
    n = np.array(bp.closed_form_task_probs(beta=dim, lambda_=1.0))
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

def experiment_5(sim_npy="artifacts/similarity-matrix/3epochs-t0-flan2021-cot-tulu-sglue.npy"):
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
            # Instantiate the optimizer and compute probabilities
            bp = QuadraticConvexOptimization(S)
            n = np.array(bp.closed_form_task_probs(beta=beta, lambda_=lambda_))
            
            # Plot the probability distribution as a line plot
            plt.plot(range(len(n)), n, label=f"β={beta}, λ={lambda_}")

    # Add plot details
    plt.xlabel("Task Index")
    plt.ylabel("Probability")
    plt.title("Probability Distributions for Different β and λ Combinations")
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
    
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

if __name__ == "__main__":
    mp.set_start_method("spawn")
    # experiment_2()
    experiment_5()
