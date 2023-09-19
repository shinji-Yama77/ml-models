import numpy as np
from node import Node
from collections import Counter

class ID3:

    # parameters: stores the max depth for tree
    def __init__(self, mx_depth=None):
        self.mx_depth = mx_depth
        self.rootNode = None # set rootnode later to fit when building tree


    # calculates entropy for each node
    def cal_entropy(self, y, node_indices):
        entropy = 0
        if (len(node_indices) != 0):
            filter_first = y[node_indices]  # filter indices 
            p_1 = len(filter_first[filter_first==1]) / len(node_indices) # calculating the probability that y=1
            if (p_1 == 1) or (p_1 == 0):
                entropy = 0
            else:
                entropy = -p_1*np.log2(p_1)-(1-p_1)*np.log2(1-p_1)
        return entropy
        

    # computes the gain for specific feature and threshold value
    def compute_gain(self, left_indices, right_indices, y):

        total_examples = len(y)

        w_left = len(left_indices) / total_examples
        w_right = len(right_indices) / total_examples

        left = w_left * self.cal_entropy(y, left_indices)
        right = w_right * self.cal_entropy(y, right_indices)

        all_indices = np.arange(len(y))

        total_gain = self.cal_entropy(y, all_indices) - (left + right)
        return total_gain
        
    # find the best_feature and best_threshold by calculating the gain 
    # for all the possible features and thresholds for every feature
    def find_split(self, X, y):
        best_gain = -1
        best_thresh = 0
        best_feature = 0

        for i in range(X.shape[1]): # number of features in dataset
            total = np.unique(X[:, i]) # getting all unique values
            vals = X[:, i] 
            for j, threshold in enumerate(total): # loop through each unique threshold value
                curr_gain = 0
                left_indices = np.where(vals <= threshold)[0]
                right_indices = np.where(vals > threshold)[0]
                curr_gain = self.compute_gain(left_indices, right_indices, y)
                if (curr_gain > best_gain):
                    best_gain = curr_gain
                    best_thresh = threshold
                    best_feature = i

        
        return best_feature, best_thresh

    def split_dataset(self, X, best_feature, best_thresh):
    
        left_indices = np.where(X[:, best_feature] <= best_thresh)[0]
        right_indices = np.where(X[:, best_feature] > best_thresh)[0]
    
        return left_indices, right_indices
    

    def convert(self, X, y, indices): # converts into numpy arrays with all feature values
        new_X = X[indices]
        new_Y = y[indices]

        return new_X, new_Y


    # build_tree: method stores the best_feature and threshold for each node

    def build_tree(self, X, y, current_depth): 

        if (current_depth == self.mx_depth) or (len(np.unique(y)) == 1):
            # return the most common value
            val_label = self.classify(y)
            return Node(label=val_label)
        else:
            # convert x to indices first
            best_feature, best_thresh = self.find_split(X, y)
            left_indices, right_indices = self.split_dataset(X, best_feature, best_thresh)
            left_X, left_Y = self.convert(X, y, left_indices)
            right_X, right_Y = self.convert(X, y, right_indices)
            
            # recursion to build the tree
            
            left_n = self.build_tree(left_X, left_Y, current_depth+1)
            right_n = self.build_tree(right_X, right_Y, current_depth+1)
            return Node(left=left_n, right=right_n, threshold=best_thresh, feature=best_feature)

    # find the most common occurence and classifies the node label
    def classify(self, y): 
        data = Counter(y)
        return data.most_common(1)[0][0] # find the most common occurence


    def fit(self, X, y):
        self.rootNode = self.build_tree(X,y, 0) # receives X, y in numpy array format

        return self.rootNode # could be none too
    

    def predict(self, X):
        pred_vals = []
        for i in range(len(X)):
            pred_val = self.traverse(X[i], self.rootNode)
            pred_vals.append(pred_val)

        return pred_vals

    # returns the node label for each data point x
    def traverse(self, row, node):

        if (node.left == None) and (node.right == None):
            return node.label
        else:
            if (row[node.feature] <= node.threshold):
                return self.traverse(row, node.left)
            else:
                return self.traverse(row, node.right)
            

    




        
        
            




    





    


