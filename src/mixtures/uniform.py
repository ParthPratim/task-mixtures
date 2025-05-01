import numpy as np

from src.mixtures.base import DataMixture, TaskMeta

"""
All tasks have budget equally split among themselves
"""


class Uniform(DataMixture):
    def create_mixture(self) -> bool:
        self.task_prob = np.full((self.num_tasks,), float(1.0 / self.num_tasks))
        task_budget = self.num_instances * self.task_prob
        train_split = np.empty((0,))
        #        print(self.task_prob)
        for i, subtask_meta in enumerate(self.subtask_metas):
            sel_instances = np.random.uniform(
                low=0,
                high=subtask_meta.num_train_instances,
                size=int(task_budget[i]),
            ).astype(int)

            train_split = np.append(
                train_split, np.array(subtask_meta.train_split)[sel_instances]
            )

        #        print(train_split)

        self.final_mixture = TaskMeta(self.mixture_name, train_split=train_split)

        return True


"""
All tasks are split based on the number of instances they have
Instances are sampled uniformly given the budget
"""


class TaskInstanceProportional(DataMixture):
    def create_mixture(self) -> bool:
        def num_sample(i):
            return len(self.subtask_metas[i].train_split)

        self.task_prob = np.fromfunction(lambda i: num_sample(i), (self.num_tasks,))
        self.task_prob = self.task_prob / np.sum(self.task_prob)

        task_budget = self.num_instances * self.task_prob
        train_split = np.empty((0,))
        #        print(self.task_prob)
        for i, subtask_meta in enumerate(self.subtask_metas):
            sel_instances = np.random.uniform(
                low=0,
                high=subtask_meta.num_train_instances,
                size=int(task_budget[i]),
            ).astype(int)

            train_split = np.append(
                train_split, np.array(subtask_meta.train_split)[sel_instances]
            )

        #        print(train_split)

        self.final_mixture = TaskMeta(self.mixture_name, train_split=train_split)

        return True


"""
Tasks are sampled from a multinomial disribution 
Budget is not specified explicitly
Uniform across each subtask, budget is from multinomial
"""


class Multinomial(DataMixture):
    def create_mixture(self) -> bool:
        if self.task_prob is None:
            raise ("This mixture requires setting task probabilities")

        sel_subtasks = np.argmax(
            np.random.multinomial(
                self.num_tasks, self.task_prob, (self.num_instances,)
            ),
            axis=1,
        )

        print(sel_subtasks)
        task_budget = np.zeros((self.num_tasks,))

        def update_budget(x):
            task_budget[x] += 1

        list(map(update_budget, sel_subtasks))
        train_split = np.empty((0,))

        for i, subtask_meta in enumerate(self.subtask_metas):
            sel_instances = np.random.uniform(
                low=0,
                high=subtask_meta.num_train_instances,
                size=int(task_budget[i]),
            ).astype(int)

            train_split = np.append(
                train_split, np.array(subtask_meta.train_split)[sel_instances]
            )

        #        print(train_split)

        self.final_mixture = TaskMeta(self.mixture_name, train_split=train_split)

        return True
