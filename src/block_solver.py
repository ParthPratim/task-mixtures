import numpy as np

def generate_similarity_block(idx):
    # Example: Gaussian RBF from random task embeddings
    X = np.random.randn(len(idx), 128)  # replace 128 with task embedding dim
    dists = np.sum((X[:, None, :] - X[None, :, :])**2, axis=-1)
    S_block = np.exp(-dists / (2 * np.std(dists)))
    return S_block

def blockwise_closed_form_lazy(n, beta=1.0, lambda_=1.0, block_size=1000, num_iters=10, project_to_simplex=True, seed=42):
    np.random.seed(seed)
    p = np.ones(n) / n  # initialize uniform

    for it in range(num_iters):
        # Random block
        idx = np.random.choice(n, size=block_size, replace=False)
        S_block = generate_similarity_block(idx)
        one_block = np.ones(block_size)

        # Closed-form quantities
        Psi_un = beta * S_block @ one_block
        Psi_pair = lambda_ * S_block
        try:
            Psi_pair_inv = np.linalg.inv(Psi_pair + 1e-5 * np.eye(block_size))  # regularization
        except np.linalg.LinAlgError:
            continue  # skip if singular

        a = one_block @ (Psi_pair_inv @ Psi_un)
        b = one_block @ (Psi_pair_inv @ one_block)
        nu = (a - 1) / b
        p_block = Psi_pair_inv @ (Psi_un - nu * one_block)

        if project_to_simplex:
            p_block = np.clip(p_block, 0, None)
            p_block = p_block / (p_block.sum() + 1e-8)

        p[idx] = p_block  # update global prob vector

    return p


n = 100_000
p_opt = blockwise_closed_form_lazy(n, beta=1.0, lambda_=1.0, block_size=1000, num_iters=20)
print("First 10 probabilities:", p_opt[:10])
