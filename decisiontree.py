import numpy as np
import node

class ID3:

    def __init__(self, depth=None, num_features=None):
        self.depth = depth
        self.num_features = num_features
        



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
        
    def cal_information_gain(self):
        # calculate info   rmation gain for that specific split
        pass

    def find_split(self):
        # split the data on the best informaiton gain
        # for i in feature:
        # calculate crossentropy for each 
        pass
    def build_tree(self):
        # start off by root node
        return node
        # base case = if training examples equal to zero

        #

        pass
            




    





    


