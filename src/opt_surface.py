import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the similarity matrix S from the provided file
sim_npy = "artifacts/similarity-matrix/3epochs-t0-flan2021-cot-tulu-sglue.npy"
S = np.load(sim_npy)

# Regularize S to ensure positive semidefiniteness
def make_psd(S, epsilon=1e-6):
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals = np.maximum(eigvals, epsilon)  # Clip negative eigenvalues
    return eigvecs @ np.diag(eigvals) @ eigvecs.T

# Define the objective function f(p) = 0.5 * p^T S p - u^T p
def objective_function(p, S, u):
    return 0.5 * np.dot(p.T, np.dot(S, p)) - np.dot(u, p)

# Ensure S is positive semidefinite
S_psd = make_psd(S, epsilon=1e-6)

# Set up the grid for visualization
grid_size = 50
p_values = np.linspace(0, 1, grid_size)
P1, P2 = np.meshgrid(p_values, p_values)
Z = np.zeros_like(P1)

# Here, we compute the objective function over just two probabilities (p1, p2)
# For the full 316-dimensional optimization, you would need a more advanced method
# This just provides an example of how you can reduce the dimensionality
for i in range(grid_size):
    for j in range(grid_size):
        # Creating a 316-dimensional probability vector where only p1 and p2 are varied
        p = np.ones(316) * (1 / 316)  # Initialize uniform distribution
        p[0] = P1[i, j]  # Set p1
        p[1] = P2[i, j]  # Set p2
        u = np.ones(316)  # Uniform utility vector
        Z[i, j] = objective_function(p, S_psd, u)

# Plot the optimization surface (only for p1, p2)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Surface plot
ax.plot_surface(P1, P2, Z, cmap='viridis', edgecolor='none')

# Labels and title
ax.set_xlabel('p1 (probability of task 1)')
ax.set_ylabel('p2 (probability of task 2)')
ax.set_zlabel('Objective function value')
ax.set_title('Optimization Surface: Objective Function Over Two Probabilities')

plt.show()
