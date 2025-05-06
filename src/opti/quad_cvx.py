import cvxpy as cp
import numpy as np
from scipy.special import softmax

from src.opti.base import TaskProbabilityOptimization


def check_matrix(S):
    all_positive = np.all(S >= 0)
    is_symmetric = np.allclose(S, S.T, atol=1e-8)
    return all_positive, is_symmetric


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
            (2.0 / _lambda)
            * np.matmul(
                np.linalg.inv(A + abs_min_e_value * np.eye(A.shape[0])),
                gamma,
            )
        )

        return self.task_prob


class GraphLaplacianOptimization(TaskProbabilityOptimization):
    def compute_task_probability(self, _beta, _lambda, _sigma):
        S = self.S
        row_sum = np.sum(S, axis=1)
        np.random.seed(42)
        n = S.shape[0]
        uni_pot = _beta * row_sum
        epsilon = 1e-4
        pair_pot = np.diag(row_sum) - S
        pair_pot += epsilon * np.eye(n)
        p = cp.Variable(n, nonneg=True)
        Beta = cp.Parameter()
        Lambda = cp.Parameter()
        Beta.value = _beta
        Lambda.value = _lambda
        objective = cp.Minimize(
            0.5 * _lambda * cp.quad_form(p, cp.psd_wrap(pair_pot)) - _beta * uni_pot @ p
        )
        constraints = [cp.sum(p) == 1]
        prob = cp.Problem(objective, constraints)
        prob.solve()

        self.task_prob = p.value
        return p.value
