import unittest
from decisiontree import ID3
import numpy as np
from logisticregression import LogisticRegression

class TestTree(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
        self.y = np.array([1,1,0,0,1,0,0,1,1,0])
        self.x = np.array([5, 6, 7, 8, 9, 10])
        self.y1 = np.array([0, 1, 0, 1, 0, 1])

    def test_cal_entropy(self):
        list = np.array([0])
        list1 = np.array([1, 2, 3, 4, 5])
        id31 = ID3()
        entropy = id31.cal_entropy(self.y1, list)
        entropy1 = id31.cal_entropy(self.y1, list1)




    def test_find_split(self):
        # Test find_split function
        id3 = ID3()
        best_feature, best_thresh = id3.find_split(self.X, self.y)
        print(best_feature)
        self.assertEqual(best_feature, 2)  # Update with your expected value

    

class TestLogistic(unittest.TestCase):


    def setUp(self):
        pass
    
        
    def test_sigmoid(self):
        Logisticregression = LogisticRegression()
        output = Logisticregression.sigmoid(0)
        self.assertEqual(output, 0.5)
        






if __name__ == '__main__':
    unittest.main()

