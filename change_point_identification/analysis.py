import numpy as np
import itertools
from change_point_identification.synthesis import PerformanceHistorySynthesizer

def find_runs(value, a):
    # Create an array that is 1 where a is `value`, and pad each end with an extra 0.
    isvalue = np.concatenate(([0], np.equal(a, value).view(np.int8), [0]))
    absdiff = np.abs(np.diff(isvalue))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

class Transformation:
    
    def __init__(self):
        pass
    
    @staticmethod
    def transform_single(n_revisions: int, obs: dict, epsilon = 0.01):
        '''
        
        :param n_revisions:
        :param obs:
        :param epsilon:
        '''
        observed = sorted(obs.keys())
        
        # tune window size delta
        yss = {}
        dd = []
        for delta in range(n_revisions // 4):        
            ys = np.zeros(n_revisions)
            for a, b in itertools.combinations(observed, 2):
                if np.abs(b - a) < delta and np.abs(obs[b] - obs[a]) > epsilon:
                    ys[a:b] += 1.0
            yss[delta] = ys
            
            dd.append(len(find_runs(0, ys)))
        
        # optimal window size (trade-off between information used and number of segments)
        opt = np.argsort(dd, kind="stable")[-1]
        
        return yss[opt]
            
class GroupSamplingEstimator():
    
    def __init__(self, synthesizer: PerformanceHistorySynthesizer):
        self.synthesizer = synthesizer
        
    def initialize(self, sample_size: int, sample_rate: float = 0.05, p_select: float = 0.5):
        n_features = self.synthesizer.__n_features
        
        
        
dd = np.zeros(1000)
dd[400:] += 1
dd[800:] += 2
s = Transformation().transform_single(1000, {i: dd[i] for i in np.random.choice(np.arange(1000), size=50, replace=False)})

