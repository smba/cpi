import numpy as np
import random
import collections
import itertools
import matplotlib.pyplot as plt

# test
class PerformanceHistorySynthesizer:

    def __init__(self,
                 n_revisions: int = 1000,
                 n_features: int = 50,
                 p_influential: int = 0.08,
                 n_changepoints: int = 5,
                 p_geom: float = 0.85,
                 noise: float = 0.1,
                 seed: int = 1234):
        '''
        Class to synthesize performance histories with variable parameters.
        
        :param n_revisions: Total number of revisions in the performance history
        :param n_features: Total number of features in the variability model
        :param p_influential: Percentage of features with influence on performance
        :param n_changepoints: Number of occasions performance changes wrt. configurations
        :param p_geom: Parameter for geometric distribution of feature interaction degree
        :param seed: Seed to set on random number generators (random and numpy.random )
        '''
        
        self.n_revisions = n_revisions
        self.n_features = n_features
        self.__n_influential = int(p_influential * n_features)
        self.__n_changepoints = min(n_changepoints, n_revisions)
        self.__p_geom = p_geom
        self.__noise = noise

        # set seeds
        np.random.seed(seed)
        random.seed(seed)
        
        # Each term of the performance model has a degree, i.e., the number of features interacting.
        # We synthesize n_influential terms from the given $n_features.
        termsizes = collections.Counter(np.random.geometric(p=p_geom, size=int(n_features* p_influential)))
        
        # list of factors
        factors = []
        for degree in termsizes:
            number = termsizes[degree]
            for j in range(number):
                factor = list(np.random.choice(np.arange(n_features), size=degree, replace=False))
                factors.append(factor)
        
        factors = np.array(factors).reshape(1, -1)[0]
        factors.sort()
        factors = list(k for k,_ in itertools.groupby(factors))
        
        # Associate influences to the different factors of the performance model
        terms = []
        for factor in factors:
            
            # initial influence is between 25 and 75
            influence = np.random.uniform(25, 75)
            sign = -1 if random.random() < 0.5 else 1
            terms.append([[factor], sign * influence])
        
        # Synthesize and distribute change points among influential factors
        self.change_points = []

        for revision in sorted(np.random.choice(np.arange(n_revisions), size=self.__n_changepoints, replace=False)):
            
            # initial change rate is between 0.0 and 0.5
            change = np.random.uniform(0.0, 0.50)
            direction = -1 if random.random() < 0.5 else 1
            
            if len(terms) > 0:
                factor = np.random.choice(factors, size=1)
                self.change_points.append([revision, factor, 1 + direction*change])
        
        # keep list of terms (initial model) and respective generate function
        self.__terms = terms
       
        self.generate = lambda x: np.sum([np.prod([x[i] for i in term[0]]) * term[1] for term in terms])
        
    def synthesize(self, x: np.ndarray, revision: int = None):
        '''
        Synthesize performance observations for an individual configuration. If no revision is specified, the 
        entire history is returned.
        
        :param x: Binary vector of the configuration
        :param revision: Revision count (int) of which performance should be synthesized (default: None)
        '''
        y = self.generate(x)
        ys = np.full(self.n_revisions, y)
        for cp in self.change_points:
            do = np.prod(x[np.array(cp[1][0])])
            if do == 1:
                ys[cp[0]:] *= cp[2]
        result = ys[revision] if revision == None else ys
        return result.reshape(1, -1)[0]
        
