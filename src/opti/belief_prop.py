import numpy as np
from scipy.special import softmax

from src.opti.base import TaskProbabilityOptimization


class BeliefPropagation(TaskProbabilityOptimization):
    def compute_task_probability(self, _beta=1.0, _lambda=1.0, mu=2.0, T=50):
        S = self.S
        n = S.shape[0]

        # Unary potentials: Ψ_i = β * sum_j S_ij
        unary_pot = _beta * S.sum(axis=1)

        # Pairwise potentials with regularization
        pairwise_pot = _lambda * S + mu * np.eye(n)

        # Initialize messages: m_{i → j}
        messages = np.zeros((n, n))

        for t in range(T):
            new_messages = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        continue
                    # Compute the incoming messages to node i excluding the one from j
                    incoming = sum(messages[k, i] for k in range(n) if k != j)
                    # Update the message from i to j
                    new_messages[i, j] = -unary_pot[i] + incoming + pairwise_pot[i, j]
            messages = new_messages

        # Compute beliefs
        beliefs = np.zeros(n)
        for i in range(n):
            beliefs[i] = -unary_pot[i] + sum(messages[j, i] for j in range(n) if j != i)

        # Project beliefs onto the probability simplex
        p = softmax(-beliefs)

        return p
