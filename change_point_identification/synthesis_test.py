import unittest
from synthesis import PerformanceHistorySynthesizer
from sklearn.model_selection import ParameterGrid
import numpy as np

class Test(unittest.TestCase):


    def setUp(self):
        pass

    #def tearDown(self):
    #    pass


    def testPerformanceHistorySynthesizer(self):
        '''
        Testing different parameter configurations, sanity check
        '''
        
        pgrid = {
            "n_revisions": np.arange(1, 2000, 500),
            "n_features": np.arange(10, 1200, 500),
            "p_influential": np.linspace(0.001, 0.1, 5),
            "n_changepoints": np.arange(1, 50, 10),
            "p_geom": np.linspace(0.7, 0.99, 6),
            "noise": np.linspace(0.001, 1, 5),
            "seed": np.arange(0, 100, 25)
        }
        
        pgrid = ParameterGrid(pgrid)
        for i, param in enumerate(pgrid):
            PerformanceHistorySynthesizer(
                param["n_revisions"],
                param["n_features"],
                param["p_influential"],
                param["n_changepoints"],
                param["p_geom"],
                param["noise"],
                param["seed"],
            )

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()