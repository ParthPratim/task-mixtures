import multiprocessing as mp
import os

import numpy as np
import pandas as pd

from src.mixtures.base import TaskMeta
from src.mixtures.uniform import Multinomial, Uniform
from src.opti.belief_prop import BeliefPropagation
from src.opti.quad_cvx import QuadraticConvexOptimization
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
    orderring_file="artifacts/task-index-maps/2epochs-t0-flan2021-cot.csv",
    num_proc=2,
):
    mixture_folders = ["data/t0", "data/flan2021", "data/cot"]
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
            task_name = row["Task-Name"][len(prefix) :]
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
    S = np.load(sim_npy)

    bp = BeliefPropagation(S)
    task_prob = bp.compute_task_probability(_beta=10.0, _lambda=15.0)

    multinomial = Multinomial(
        subtasks_list,
        subtask_metas,
        NUM_INSTANCES,
        "artifacts/final-submixtures/25K-belief-prop-multinomial-to-flan2021-cot",
        task_prob=task_prob,
    )

    multinomial.create_mixture()

    multinomial.dump_mixture(f"{multinomial.mixture_name}.json")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    # experiment_2()
    experiment_3()
