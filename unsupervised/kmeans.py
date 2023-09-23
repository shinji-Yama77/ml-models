import numpy as np



class KMeans():


    def __init__(self, max_iters=10, clusters=None):
        self.max_iters = max_iters

        if clusters is None:
            raise ValueError("Number of clusters must be provided")
        
        self.clusters = clusters



    # returns the indices of centroid where it has the least cost
    def compute_indices(self, X, centroids):
        # assign points to cluster centroids
        m = X.shape[0]
        idx = np.zeros(m, dtype=int)
        for i in range(m):
            # this code subtracts each array in centroids from 
            # the array in X[i]. Each array in centroids has the same
            # number of feature values (dimensions) as X[i]
            distances = np.sum((X[i] - centroids) ** 2, axis=1)
            idx[i] = np.argmin(distances)

        return idx
    
    # calculates the new centroid by taking the average
    def new_centroids(self, X, indices):

        m, n = X.shape
        centroids = np.zeros((self.clusters, n))
        all = np.zeros(self.clusters) # save num examples for each cluster

        for i in range(m):
            centroid_val = indices[i]
            all[centroid_val] += 1
            for j in range(n):
                centroids[centroid_val][j] += X[i][j]

        for i in range(self.clusters):
            if (all[i] != 0):
                for j in range(n):
                    centroids[i][j] = centroids[i][j] / all[i]

        return centroids

    def fit(self, X):
        # assuming X is a numpy array 
        rands = np.random.permutation(X.shape[0])
        centroids = X[rands[:self.clusters]]

        for i in range(self.max_iters):
            idx = self.compute_indices(X, centroids)

            centroids = self.new_centroids(X, idx)


        return centroids, idx








