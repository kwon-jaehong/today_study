import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math
import matplotlib.pyplot as plt

def eucliDist(A, B):
    return np.sqrt(np.sum((A-B)**2))



class kMeans():
    
    def __init__(self,k=5,max_iters=100,plt_steps=False):
        self.k=k
        self.max_iters =max_iters
        self.clusters = [[] for _ in range(self.k)]
        self.centroids = []
        self.plt_steps = plt_steps
        
        
    def predict(self,X):
        self.X = X
        self.n_samples,self.n_features = X.shape
        
        random_sample_idxs = np.random.choice(self.n_samples,self.k,replace=False)
        
        ## 센터 좌표
        self.centroids = [self.X[idx] for idx in random_sample_idxs]
        
        for _ in range(self.max_iters):
            self.clustres = self._create_clusters(self.centroids)
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

                
            if self._is_converged(centroids_old,self.centroids):
                break
        return self._get_cluster_labels(self.clusters)
            
    
    def _create_clusters(self,centorids):
        clustres = [ [] for _ in range(self.k)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample,centorids)
            clustres[centroid_idx].append(idx)
        return clustres
    
    
    def _closest_centroid(self,sample,centroids):
        distances = []
        for point in centroids:
            distances.append(eucliDist(sample,point))            
        closest_idx = np.argmin(distances)
        return closest_idx
            
    def _get_centroids(self,clusters):
        centroids = np.zeros((self.k,self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster],axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self,centroids_old,centroids):
        distances = [eucliDist(centroids_old[i],centroids[i]) for i in range(self.k)]
        return sum(distances) == 0

    def _get_cluster_labels(self,clusters):
        labels = np.empty(self.n_samples)
        for cluster_idx,cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels





x = [25, 25, 25, 25, 25, 25, 25, 25, 25, 317, 317, 317, 317, 317, 317, 317, 317, 317, 610, 610, 610, 610, 610, 610, 610, 610, 610, 902, 902, 902, 902, 902, 902, 902, 902, 902, 1192, 1192, 1192, 1192, 1192, 1192, 1192, 1192, 1192, 1483, 1483, 1483, 1483, 1483, 1483, 1483, 1483, 1483, 1775, 1775, 1775, 1775, 1775, 1775, 1775, 1775, 1775, 2066, 2066, 2066, 2066, 2066, 2066, 2066, 2066, 2066, 2358, 2358, 2358, 2358, 2358, 2358, 2358, 2358, 2358, 2650, 2650, 2650, 2650, 2650, 2650, 2650, 2650, 2650, 2943, 2943, 2943, 2943, 2943, 2943, 2943, 2943, 2943, 3237, 3237, 3237, 3237, 3237, 3237, 3237, 3237, 3237, 3527, 3527, 3527, 3527, 3527, 3527, 3527, 3527, 3527, 3821, 3821, 3821, 3821, 3821, 3821, 3821, 3821, 3821, 0]
np_arr = np.array(x)
k = 4
k = kMeans(k=k,max_iters=150,plt_steps=False)

y_pred = k.predict(np_arr.reshape(np_arr.shape[0],1))



# print(np.reshape(np.array(x),np.array(x).shape[0],1))
# print()





































