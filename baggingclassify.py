import numpy as np
from decisiontree import ID3
from collections import Counter



class BaggingClassifier:

    def __init__(self, num_trees=10, max_depth=8):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.trees = None
        


    def create_trees(self, X, y):
        save_trees = []


        for i in range(self.num_trees):
            new = ID3(mx_depth=self.max_depth)
            rans = np.random.choice(X.shape[0], X.shape[0], replace=True)
            sub_x = X[rans]
            sub_y = y[rans]
            new.fit(sub_x, sub_y)
            save_trees.append(new) 

        return save_trees
    
    def fit(self, X, y):

        all_trees = self.create_trees(X, y)

        self.trees = all_trees

        return all_trees
    
    def predict(self, X):
        curr = []
        all_pred = []
        for obj in self.trees:
            pred = obj.predicts(X)
            all_pred.append(pred)

        grouped_predictions = list(zip(*all_pred))
        for i in range(len(grouped_predictions)):
            pre = self.classify(np.array(grouped_predictions[i]))
            curr.append(pre)
        return curr
 

    def classify(self, preds): 
        data = Counter(preds)
        return data.most_common(1)[0][0]
        
    def view_trees(self):
        for i, obj in self.trees:
            print(obj)
        
        

