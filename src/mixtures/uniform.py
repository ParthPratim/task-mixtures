import numpy as np

from src.mixtures.base import DataMixture, TaskMeta


class Uniform(DataMixture):
    def create_mixture(self) -> bool:
        self.task_prob = np.full((self.num_tasks,), float(1.0 / self.num_tasks))
        task_budget = self.num_instances * self.task_prob
        train_split = np.empty((0,))
        print(self.task_prob)
        for i, subtask_meta in enumerate(self.subtask_metas):
            sel_instances = np.random.uniform(
                low=0,
                high=subtask_meta.num_train_instances,
                size=int(task_budget[i]),
            ).astype(int)

            train_split = np.append(
                train_split, np.array(subtask_meta.train_split)[sel_instances]
            )

        print(train_split)

        self.final_mixture = TaskMeta(self.mixture_name, train_split=train_split)
