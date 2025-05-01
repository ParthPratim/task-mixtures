import numpy as np
from scipy.special import softmax

from src.opti.base import TaskProbabilityOptimization


class QuadraticConvexOptimization(TaskProbabilityOptimization):
    """
    1. Compute Minimum EignenValues for the matrix
    2. Make Matrix Positive Semi-Definite
    3. Impose Regularizer
    4. Compute optimization solution
    """

    def compute_task_probability(self, _beta, _lambda):
        A = _lambda * self.S
        gamma = _beta * np.sum(self.S, axis=0)
        abs_min_e_value = np.abs(np.min(np.linalg.eigvals(A)))

        self.task_prob = softmax(
            -np.matmul(
                np.linalg.inv(A + 2 * _lambda * abs_min_e_value * np.eye(A.shape[0])),
                gamma,
            )
        )

        return self.task_prob
