import numpy as np


class LogisticRegression:

    def __init__(self, lr=0.01, iterations=1000):
        self.lr = lr
        self.iterations = iterations
        self.final_w = None
        self.final_b = None


    # outputs a value between 0 and 1
    def sigmoid(self, z):

        g = 1 / (1 + np.exp(-z))

        return g
    
    def single_cost(self, actual, out):

        cost = actual*np.log(out)+(1-actual)*np.log(1-out)

        return cost
    
    def single_descent(self, X, y, b, w):


        # receives b 
        # receives w




        for i in range(m):





        pass

    def gradient_descent(self, X, y):

        w = np.zeros(X.shape[1])
        for i in range(self.iterations):
            self.single_descent(X, y)

        pass

    

    def fit(self, X, y):

        self.gradient_descent(X, y)




        

    
    



    
    


