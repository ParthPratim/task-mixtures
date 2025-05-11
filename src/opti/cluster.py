from src.opti.base import TaskProbabilityOptimization
from src.opti.quad_cvx import QuadraticConvexOptimization
from sklearn.cluster import KMeans
import numpy as np
from multiprocessing import Pool

def compute_cluster_uniform_prob(S, _beta = 0.25, _lambda=1.25):
    if S is None:
        return []
    print("Started QuadCvx On Similarity Matrix shape : " , S.shape)
    obj = QuadraticConvexOptimization(S)        
    return obj.closed_form_task_probs(_beta, _lambda)


"""
Clusterring over task embeddings 
"""
class ClusteredOptimization(TaskProbabilityOptimization):

    def compute_task_probability(self,  task_embeddings, cluster_size = 10, _beta = 0.25, _lambda = 1.25, use_sim_mat=False):
        kmeans = KMeans(n_clusters=cluster_size, 
               init='k-means++',
               n_init=10, 
               max_iter=500,
               random_state=42,
               algorithm='lloyd').fit(task_embeddings if not use_sim_mat else  self.S)


        labels = kmeans.labels_
        print(labels)
        cluster_task_map = {} 
        cluster_sim_mat_params = [ None ] * cluster_size
        cluster_probs = []
        """ 
        IMPORTANT TO NOTE
        -------------------
        For cluster k,
        ith row in cluster_sim_mat[k] refers cluster_task_map[k][i] task
        """
        active_clusters = 0

        print(cluster_task_map)
        S = self.S
        def extract_matrix(task_indices):
            n = len(task_indices)
            new_mat = np.zeros((n,n))
            for i,index1 in enumerate(task_indices):
                for j,index2 in enumerate(task_indices):
                    new_mat[i, j] = S[index1, index2]
            return new_mat

        for idx, l in enumerate(labels):
            if not l in cluster_task_map:
                cluster_task_map[l] = []
            cluster_task_map[l].append(idx) 

        for i in range(cluster_size):
            if  i in cluster_task_map:
                active_clusters += 1
                #print(cluster_task_map[i])
                cluster_sim_mat_params[i] = (extract_matrix(cluster_task_map[i]), _beta, _lambda)
        
        with Pool(active_clusters) as pool:
            cluster_probs = list(pool.starmap(compute_cluster_uniform_prob, cluster_sim_mat_params))
            # cluster_probs has shape : (clusters, each_cluster_size)
            pool.close()
            pool.join()        

        return cluster_task_map, cluster_probs