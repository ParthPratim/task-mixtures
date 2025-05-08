import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed


"""
Hyperparameter Search Tool
"""

"""
GridSearch over parameter values in a given step size 
"""

def _evaluate_config(model_class, S, beta, lam, evaluator):
    try:
        p = model_class(S).compute_task_probability(_beta=beta, _lambda=lam)
        rank = evaluator(p)
        return (rank, beta, lam)
    except Exception:
        return None

def GridSearch(model, S, beta_range=(0,1), lambda_range=(0,1), step_size=0.01, evaluator=None):
    if evaluator is None:
        raise Exception("Implement ranking function used for evaluator")

    beta_vals = np.arange(beta_range[0], beta_range[1], step_size)
    lambda_vals = np.arange(lambda_range[0], lambda_range[1], step_size)

    tasks = [(model, S, b, l, evaluator) for b in beta_vals for l in lambda_vals]

    curr_best_value = -1
    curr_best_config = (-1, -1)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(_evaluate_config, *task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                rank, b, l = result
                print(f"beta = {b}, lambda = {l}, rank = {rank}")
                if rank > curr_best_value:
                    curr_best_value = rank
                    curr_best_config = (b, l)

    return curr_best_value, curr_best_config
