import numpy as np


class LogisticRegression:

    def __init__(self, lr=0.01, iterations=1000):
        self.lr = lr
        self.iterations = iterations


    # outputs a value between 0 and 1
    def sigmoid(self, z):

        g = 1 / (1 + np.exp(-z))

        return g
    


    
    


