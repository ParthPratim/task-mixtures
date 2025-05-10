import torch
from src.opti.base import TaskProbabilityOptimization

class ProjectedGradientDescent(TaskProbabilityOptimization):

    def compute_task_probability(self, _beta , _lambda):
        S = self.S
        n = S.shape[0]
        p = torch.zeros((n,))
        uni_pot = _beta * torch.sum(S, dim=1)
        with torch.no_grad():
            p_new = 0.5 * _lambda *(p @ S @ p) - uni_pot @ p 
            


