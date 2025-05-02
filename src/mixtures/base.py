import json
from abc import ABC, abstractmethod
from typing import List


class TaskMeta:
    def __init__(self, task_name, train_split=[], validation_split=[]):
        self.task_name = task_name
        self.train_split = train_split
        self.validation_split = validation_split
        # if self.train_split:
        self.num_train_instances = len(self.train_split)
        # if self.vaidation_split:
        self.num_validation_instances = len(self.validation_split)


class DataMixture(ABC):
    def __init__(
        self,
        subtask_list: list,
        subtask_metas: List[TaskMeta],
        num_instances: int,
        mixture_name=None,
        task_prob=None,
    ):
        if mixture_name is None:
            #  Add code to use date-time combination to create a name
            pass

        self.mixture_name = mixture_name
        self.subtask_list = subtask_list
        self.subtask_metas = subtask_metas
        self.num_tasks = len(subtask_list) if subtask_list else 0
        self.mixture_ready = False
        self.final_mixture = None
        self.num_instances = num_instances
        self.task_prob = task_prob  # shape : (self.num_tasks, )

    @abstractmethod
    def create_mixture(self) -> bool:
        pass

    def get_mixture(self) -> TaskMeta:
        return self.final_mixture

    def dump_mixture(self, filename):
        with open(filename, "w") as f:
            json.dump(
                {
                    "train": list(self.final_mixture.train_split),
                    "validation": list(self.final_mixture.validation_split),
                },
                f,
            )
