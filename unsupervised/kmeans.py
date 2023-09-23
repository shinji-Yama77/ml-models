import numpy as np



class KMeans():


    def __init__(self, max_iters=100, clusters=None):
        self.max_iters = max_iters

        if clusters is None:
            raise ValueError("Number of clusters must be provided")
        
        self.clusters = clusters



    # returns the indices of centroid where it has the least cost
    
    def compute_indices(self, X, centroids):
        # assign points to cluster centroids
        m = X.shape[0]

        for i in range(m):
            





