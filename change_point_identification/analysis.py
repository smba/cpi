import numpy as np
import itertools
import matplotlib.pyplot as plt


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
        observed = sorted(obs.keys())
        ys = np.zeros(n_revisions)
        
        # Tuning of Delta
        yss = {}
        dd = []
        for delta in range(n_revisions // 3):        
            for a, b in itertools.combinations(observed, 2):
                if np.abs(b - a) < delta and np.abs(obs[b] - obs[a]) > epsilon:
                    ys[a:b] += 1.0
            yss[delta] = ys
            
            dd.append(len(find_runs(0, ys)))
        
        plt.plot(range(n_revisions // 3), dd)
        plt.axvline(np.argsort(dd, kind="stable")[-1], color="dodgerblue")
        plt.show()
            

dd = np.zeros(1000)
dd[400:] += 1
dd[700:] += 2
s = Transformation().transform_single(1000, {i: dd[i] for i in np.random.choice(np.arange(1000), size=50, replace=False)})
