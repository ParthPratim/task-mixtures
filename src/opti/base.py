from abc import ABC, abstractmethod

import numpy as np

"""
Abstract class representing what any Task Probability optimization problem for data mixtures should look like

S : Similarity Matrix (nxn

"""


class TaskProbabilityOptimization(ABC):
    def __init__(self, S):
        assert len(S.shape) == 2 and S.shape[0] == S.shape[1]
        self.S = S
        self.n = S.shape[0]
        self.task_prob = np.array((self.n,))

    @abstractmethod
    def compute_task_probability(self, *args, **kwargs):
        pass
