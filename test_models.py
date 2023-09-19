import unittest
from decisiontree import ID3
import numpy as np
from logisticregression import LogisticRegression

class TestTree(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
        self.y = np.array([1,1,0,0,1,0,0,1,1,0])


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

