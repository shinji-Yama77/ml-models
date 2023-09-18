import numpy as np
import node

class ID3:

    def __init__(self, depth=None, num_features=None):
        self.depth = depth
        self.num_features = num_features
        self.rootNode = None # set rootnode later to fit when building tree



    def cal_entropy(self, y, node_indices):
        # calculate cross_entropy for that node
        entropy = 0
        if (len(node_indices) != 0):
            filter_first = y[y==node_indices] # filter indices 
            p_1 = len(filter_first[filter_first==1]) / len(node_indices) # calculating the probability that y=1
            if (p_1 == 1) or (p_1 == 0):
                entropy = 0
            else:
                entropy = -p_1*np.log2(p_1)-(1-p_1)*np.log2(1-p_1)
        return entropy
        
    def compute_gain(self, left_indices, right_indices, num_examples, all_indices):
        # calculate info   rmation gain for that specific split


        # weighted averages 

        w_left = len(left_indices) / num_examples
        w_right = len(right_indices) / num_examples

        left = w_left * self.cal_entropy(left_indices)
        right = w_right * self.cal_entropy(right_indices)

        total_gain = self.cal_entropy(all_indices) - (left + right)

        return total_gain
        

    def find_split(self, X, y, feature_names):
        # split the data on the best information gain
        # for i in feature:
        # calculate crossentropy for each
        best_gain = 0
        best_thresh = 0
        best_feature = ""

        for i, feature in enumerate(feature_names):
            total = np.unique(X[:, i]) # getting all unique values
            vals = X[:, i] 
            for j, threshold in enumerate(total): # loop through each unique threshold value
                curr_gain = 0
                left_indices = np.where(vals <= threshold)[0]
                right_indices = np.where(vals > threshold)[0]
                curr_gain = compute_gain(left_indices, right_indices)
                if (curr_gain > best_gain):
                    best_gain = curr_gain
                    best_thresh = threshold
                    best_feature = i

        
        return best_feature, best_thresh

        pass


    def split_dataset(self, X, best_feature, best_thresh):
        

        left_indices = np.where(X[:, best_feature] <= best_thresh)[0]
        right_indices = np.where(X[:, best_feature] > best_thresh)[0]
        

        return left_indices, right_indices

    def build_tree(self, X, y, current_depth):
        # start off by root node

        if # number of features == 0
            # return node()

        else:
            
            best_feature, best_thresh = self.find_split(X, y)
            left_indices, right_indices = self.split_dataset(X, best_feature, best_thresh)

            
            left = build_tree()
            right = build_tree()

        return node
        # base case = if training examples equal to zero

        #

    def fit(X, y):
        self.rootNode = self.build_tree(X,y)

        return self.rootNode

        
        
            




    





    


