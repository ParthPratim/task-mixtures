# Load the similarity matrix
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
import cvxpy as cp
import numpy as np

from src.opti.quad_cvx import QuadraticConvexOptimization


matrix_path = "/Users/prateekchanda/task-mixtures/artifacts/similarity-matrix/2epochs-t0-flan2021-cot.npy"
similarity_matrix = np.load(matrix_path)

# Instantiate the QuadraticConvexOptimization class
# optimizer = QuadraticConvexOptimization(S=similarity_matrix)

# Define _beta and _lambda values
# _beta = 0.5  # Example value, adjust as needed
# _lambda = 0.1  # Example value, adjust as needed

# Compute task probability
# task_probability = optimizer.compute_task_probability(_beta, _lambda)

# Print the result
# print("Task Probability:", task_probability)


# Number of tasks
n = similarity_matrix.shape[0]
np.random.seed(42)


num_positive = np.sum(similarity_matrix > 0)
num_negative = np.sum(similarity_matrix < 0)
num_zero = np.sum(similarity_matrix == 0)

print(f"Number of positive values: {num_positive}")
print(f"Number of negative values: {num_negative}")
print(f"Number of zero values: {num_zero}")
# Similarity matrix S (symmetric)


# S = similarity_matrix
# print("Similarity matrix S:\n", S)

gamma = 1.0 / similarity_matrix.shape[0]  # Adjust gamma as needed
S = rbf_kernel(similarity_matrix, gamma=gamma)
print("Similarity matrix S:\n", S)
# Degree matrix D
# D = np.diag(S.sum(axis=1))

# Graph Laplacian
# L = D - S

eigenvalues = np.linalg.eigvals(S)
print("Eigenvalues of L:", eigenvalues)
# Regularization for numerical PSD guarantee
# epsilon = 1e-5
# L += epsilon * np.eye(n)

S
# Hyperparameters
beta = 2.0
lambda_ = 60.0

# Unary potential
Psi_un = S @ np.ones(n)

# CVXPY variable
p = cp.Variable(n)

# Objective
objective = cp.Minimize(-beta * Psi_un @ p +
                        (lambda_ / 2) * cp.quad_form(p, S))

# Constraints
constraints = [p >= 1e-16, cp.sum(p) == 1]

# Problem
problem = cp.Problem(objective, constraints)
problem.solve()


# Output
print("Optimal probability vector p*:\n", p.value)

# Plot the probability distribution
plt.figure(figsize=(10, 6))
plt.bar(range(len(p.value)), p.value, color='skyblue', edgecolor='black')
plt.xlabel('Task Index')
plt.ylabel('Probability')
plt.title('Optimal Probability Distribution')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
