import cvxpy as cp
import numpy as np
from scipy.special import softmax
import numpy as np

from src.opti.base import TaskProbabilityOptimization


def check_matrix(S):
    all_positive = np.all(S >= 0)
    is_symmetric = np.allclose(S, S.T, atol=1e-8)
    return all_positive, is_symmetric


def project_to_simplex_chenn_et_al_2024(y):
    sorted_indices = np.argsort(y)
    y = y[sorted_indices]

    csum = np.cumsum(y)
    n = y.shape[0]
    t = np.zeros(y.shape)
    t_cap = -1
    for i in range(n-2,-1,-1):
        t[i] = (csum[n-1] - csum[i] - 1) / (n - i - 1)
        if t[i] >= y[i]:
            t_cap = t[i]
            break
    
    if t_cap == -1:
        t_cap = (csum[n-1] - 1 ) / n 
    
    inverse_indices = np.argsort(sorted_indices) 

    return np.maximum(y - t_cap, 0)[inverse_indices]


class QuadraticConvexOptimization(TaskProbabilityOptimization):
    """
    1. Compute Minimum EignenValues for the matrix
    2. Make Matrix Positive Semi-Definite
    3. Impose Regularizer
    4. Compute optimization solution
    """

    def closed_form_task_probs(self, beta=1.0, lambda_=1.0, epsilon=1e-5, project_to_simplex=True):
        """
        Computes task probability vector p* using closed-form expression with
        Laplacian-based pairwise potentials and unary potentials from similarity matrix S.

        Args:
            S: Similarity matrix (n x n), can have negative entries (e.g., PMI)
            beta: Unary potential coefficient
            lambda_: Pairwise potential coefficient
            epsilon: Regularization term added to Laplacian for numerical stability
            project_to_simplex: Whether to project result back to probability simplex

        Returns:
            p_star: Optimal probability vector (n,)
        """
        S = self.S
        n = S.shape[0]
        S = (S + S.T) / 2  # Ensure symmetry

        use_laplace = False
        if(use_laplace == False):
            # Check if S is positive semi-definite
            all_positive, is_symmetric = check_matrix(S)
            if not all_positive or not is_symmetric:
                print("Matrix is not positive semi-definite or not symmetric")
                #return None
            min_eigenvalue = np.min(np.linalg.eigvals(S))
            if min_eigenvalue < 0:
                #print("Min eigen val is neg")
                S += np.abs(min_eigenvalue) * np.eye(n)
            
            S = (S + S.T) / 2  # Ensure symmetry
            all_positive, is_symmetric = check_matrix(S)
            if not all_positive or not is_symmetric:
                print("Matrix is not positive semi-definite or not symmetric")
                #return None
            else:
                print("Matrix is positive semi-definite and symmetric")
                
            S_pos = S
        else:
            sigma = np.std(S)
            S = np.exp(S / sigma)
            D = np.diag(S.sum(axis=1))
            L = D - S + epsilon * np.eye(n)
        
            S_pos = L

        # Construct Laplacian L
        #print("S pos", S_pos)
        #sigma = np.std(S)
        #S_pos = np.exp(S / sigma)  # Convert PMI to positive similarities
        #D = np.diag(S_pos.sum(axis=1))
        #L = D - S_pos + epsilon * np.eye(n)
        
#        row_sums = S.sum(axis=1, keepdims=True)
 #       S = S / row_sums  #
        
        #print("S1_n:::",S @ np.ones(n))
        
        # Unary and pairwise potentials
        Psi_un = beta * S_pos @ np.ones(n)
        Psi_pair = lambda_ * S_pos

        # Inverse of pairwise potential
        Psi_pair_inv = np.linalg.inv(Psi_pair)
        one = np.ones(n)

        a = one @ (Psi_pair_inv @ Psi_un)
        b = one @ (Psi_pair_inv @ one)
        nu = (a - 1) / b

        p_star = Psi_pair_inv @ (Psi_un - nu * one)

        if project_to_simplex:
            # Project to probability simplex
            p_star = project_to_simplex_chenn_et_al_2024(p_star)

        return p_star

    def compute_task_probability_single_param(self, _ratio):
        S = self.S
        row_sum = np.sum(S, axis=1)
        np.random.seed(42)
        n = S.shape[0]
        uni_pot = _ratio * row_sum  # _ratio = beta / lambda
        epsilon = 1e-4
        abs_min_e_value = np.abs(np.min(np.linalg.eigvals(S)))
        pair_pot = S + abs_min_e_value * np.eye(n)

        p = cp.Variable(n, nonneg=True)
        Ratio = cp.Parameter()
        Ratio.value = _ratio
        objective = cp.Minimize(
            0.5 * cp.quad_form(p, cp.psd_wrap(pair_pot)) - uni_pot @ p
        )
        constraints = [cp.sum(p) == 1]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP)

        self.task_prob = p.value
        return p.value

    def compute_task_probability(self, _beta, _lambda):
        S = self.S 
        row_sum = np.sum(S, axis=1)
        np.random.seed(42)
        n = S.shape[0]
        uni_pot = _beta * row_sum
        epsilon = 1e-4
        abs_min_e_value = np.abs(np.min(np.linalg.eigvals(S)))
        pair_pot = S + abs_min_e_value * np.eye(n)

        p = cp.Variable(n, nonneg=True)
        Beta = cp.Parameter()
        Lambda = cp.Parameter()
        Beta.value = _beta
        Lambda.value = _lambda
        objective = cp.Minimize(
            0.5 *
            _lambda * cp.quad_form(p, cp.psd_wrap(pair_pot)
                                    ) - _beta * uni_pot @ p
        )
        constraints = [cp.sum(p) == 1]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP)

        self.task_prob = p.value
        return p.value


class GraphLaplacianOptimization(TaskProbabilityOptimization):
    def compute_task_probability(self, _beta, _lambda):
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
            0.5 *
            _lambda * cp.quad_form(p, cp.psd_wrap(pair_pot)
                                   ) - _beta * uni_pot @ p
        )
        constraints = [cp.sum(p) == 1]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)

        self.task_prob = p.value
        return p.value
