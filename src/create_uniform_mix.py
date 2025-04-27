import os

from src.mixtures.base import TaskMeta
from src.mixtures.uniform import Uniform
from src.preprocess.dataset import load_dataset

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
    mixture_folders = ["data/t0", "data/flan2021", "data/cot"]
    subtask_jsons = []
    subtask_metas = []

    subtasks_list = []
    for data_folder in mixture_folders:
        subtask_jsons.append((data_folder, os.listdir(data_folder)))

    for prefix, subtasks in subtask_jsons:
        for subtask in subtasks:
            subtasks_list.append(subtask)
            print(subtask)
            train_instances = list(
                load_dataset(os.path.join(prefix, subtask), splits=["train"])
            )
            subtask_metas.append(TaskMeta(subtask, train_split=train_instances))

    uniform = Uniform(
        subtasks_list,
        subtask_metas,
        NUM_INSTANCES,
        "25K-uniform-t0-flan2021-cot-all-full",
    )

    uniform.create_mixture()
    print("hel")
    uniform.dump_mixture(f"{uniform.mixture_name}.json")


experiment_1()
