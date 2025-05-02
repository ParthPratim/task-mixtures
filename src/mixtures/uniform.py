import bisect

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

        self.task_prob = np.array(self.task_prob)
        sel_inst_prob = []
        inst_cnt = [0]
        for i in range(self.num_tasks):
            num_instances = self.subtask_metas[i].num_train_instances
            inst_cnt.append(num_instances)
            sel_inst_prob.extend([self.task_prob[i] / num_instances] * num_instances)

        sel_inst_prob = np.array(sel_inst_prob)
        nz_probs = np.count_nonzero(self.task_prob)

        if nz_probs >= self.num_instances:
            sel_inst = np.random.choice(
                len(sel_inst_prob),
                p=sel_inst_prob,
                size=self.num_instances,
                replace=False,
            )
        else:
            sel_inst = np.random.choice(
                len(sel_inst_prob), p=sel_inst_prob, size=self.num_instances
            )

        inst_cnt = np.cumsum(inst_cnt)

        def get_mapped_instance(index):
            idx = bisect.bisect_right(inst_cnt, index)  # actual index I am looking for
            if idx == len(inst_cnt) or inst_cnt[idx] > index:
                idx -= 1

            true_index = index - inst_cnt[idx]
            assert true_index >= 0 and true_index < len(
                self.subtask_metas[idx].train_split
            )
            return self.subtask_metas[idx].train_split[true_index]

        train_split = list(map(get_mapped_instance, sel_inst))
        self.final_mixture = TaskMeta(self.mixture_name, train_split=train_split)

        return True
