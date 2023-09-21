import numpy as np


class LogisticRegression:

    def __init__(self, lr=0.001, iterations=1000):
        self.lr = lr
        self.iterations = iterations
        self.final_w = None
        self.final_b = None


    # outputs a value between 0 and 1
    def sigmoid(self, z):

        g = 1 / (1 + np.exp(-z))

        return g
    
    def single_cost(self, actual, out):

        cost = -actual*np.log(out)-(1-actual)*np.log(1-out)
        return cost

    
    def single_descent(self, X, y, b, w):


        m = X.shape[0]
        num_features = X.shape[1]
        total_cost = 0

        for i in range(m):
            z = np.dot(X[i], w) + b
            out_sig = self.sigmoid(z)
            err = out_sig - y[i]
            total_cost += self.single_cost(y[i], out_sig)
            for j in range(num_features):
                w[j] += err * X[i][j]
            b += err

        b = b / m
        w = w / m # numpy broadcasting, each number in w is divided by m
        total_cost = total_cost / m
            
        return w, b, total_cost

    def gradient_descent(self, X, y):

        initial_w = np.zeros(X.shape[1])
        initial_b = 0
        all_costs = []

        for i in range(self.iterations):
            curr_w, curr_b, curr_cost = self.single_descent(X, y, initial_b, initial_w)
 
            initial_w = initial_w - (self.lr * curr_w)
            initial_b = initial_b - (self.lr * curr_b)

            if (i%10 == 0):
                all_costs.append(curr_cost)
        self.final_w = initial_w
        self.final_b = initial_b

        return self.final_w, self.final_b, all_costs

    def fit(self, X, y):

        final_w, final_b, all_costs = self.gradient_descent(X, y)
        
        return final_w, final_b, all_costs
    
    def predict(self, X):
        final_preds = []

        for i in range(len(X)):
            out = np.dot(X[i], self.final_w) + self.final_b
            pred = self.sigmoid(out)
            if pred < 0.5:
                final_preds.append(0)
            else:
                final_preds.append(1)

        return np.array(final_preds)



        

    
    



    
    


