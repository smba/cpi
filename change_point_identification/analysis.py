import numpy as np
import itertools
import matplotlib.pyplot as plt
import scipy.signal as signal
import synthesis
import pandas as pd

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
        
        # bias correction
        min_x = min(obs.keys())
        max_x = max(obs.keys())
        half_n_revisions = n_revisions // 2
        n_sample = len(obs)
        obs_corrected = {x: obs[min_x] for x in np.random.choice(np.arange(half_n_revisions), size=n_sample)}
        obs_corrected.update(
            {n_revisions +half_n_revisions + x: obs[max_x] for x in np.random.choice(np.arange(half_n_revisions), size=n_sample)}
        )
        obs_corrected.update(
            {x + half_n_revisions: obs[x] for x in obs}
        )
                
        observed = sorted(obs_corrected.keys())
                  
        # tune window size delta
        yss = {}
        dd = []
        for delta in range(n_revisions // 10):        
            ys = np.zeros(2*n_revisions)
            #print(observed)
            for a, b in itertools.combinations(observed, 2):
                if np.abs(b - a) < delta and np.abs(obs_corrected[b] - obs_corrected[a]) > epsilon:
                    ys[a:b] += 1.0
            yss[delta] = ys
            
            dd.append(len(find_runs(0, ys)))
        
        # optimal window size (trade-off between information used and number of segments)
        opt = np.argsort(dd, kind="stable")[-1]
        
        # remove bias correction
        return yss[opt][half_n_revisions:n_revisions + half_n_revisions]
            
class GroupSamplingEstimator():
    
    def __init__(self, synthesizer: synthesis.PerformanceHistorySynthesizer):
        self.synthesizer = synthesizer
        
    def initialize(self, sample_size: int, sample_rate: float = 0.05, p_select: float = 0.5):

        n_features = self.synthesizer.n_features
        n_revisions = self.synthesizer.n_revisions
        
        # Sample initial configurations
        # Remove duplicates from configurations and add new ones if necessary
        configs = np.random.choice([0, 1], p = [1-p_select, p_select], size=(sample_size, n_features))
        configs = np.unique(configs, axis=0)
        while configs.shape[0] < sample_size:
            n_missing = sample_size - configs.shape[0]
            config = np.random.choice([0, 1], p = [1-p_select, p_select], size=(n_missing, n_features))
            configs = np.vstack([configs, config])
            configs = np.unique(configs, axis=0)
            
        # Sample revisions to observe for each configuration
        n_sample_revisions = int(sample_rate * n_revisions)
        change_map = []
        i = 0
        for config in configs:
            print(i)
            performance = self.synthesizer.synthesize(config)
            revisions = sorted(np.random.choice(np.arange(n_revisions), size=n_sample_revisions, replace=False))
            observations = {rev: performance[rev] for rev in revisions}
            change = Transformation.transform_single(n_revisions, observations, epsilon = 0.01) # TODO tune epsilon
            
            # normalize change
            change = np.divide(change - np.nanmin(change),np.nanmax(change) - np.nanmin(change))
            change = np.nan_to_num(change)
            change_map.append(change)
            i += 1
        change_map = np.vstack(change_map)
        
        feature_weights1 = np.zeros((configs.shape[1], n_revisions))
        feature_weights0 = np.zeros((configs.shape[1], n_revisions))
        for f in range(configs.shape[1]):
            print(f)
            indexes1 = np.argwhere(configs[:,f] == 1)
            indexes0 = np.argwhere(configs[:,f] == 0)
            mean1 = np.mean(change_map[indexes1,:], axis=0)
            mean0 = np.mean(change_map[indexes0,:], axis=0)
            feature_weights1[f] = mean1
            feature_weights0[f] = mean0
            
        feature_weights = feature_weights1 - feature_weights0
                
        #smoothed = np.sum(change_map, axis=0)#pd.DataFrame(np.sum(change_map, axis=0)).rolling(window=30, center=True).mean().values.reshape(1, -1)[0]
        #fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        
        #ax1.pcolormesh(change_map, cmap="binary")
        #ax1.set_title("Change footprint")
        #ax1.set_xlabel("time")
        #ax1.set_ylabel("configuration")
        #for cm in change_map:
        #    ax2.plot(np.arange(n_revisions), cm)
        #ax2.plot(smoothed / len(configs))
        #ax2.fill_between(np.arange(n_revisions), np.zeros(n_revisions), smoothed.reshape(1, -1)[0] / len(configs), alpha=0.5)
        #ax2.set_title("Sum of change point probability")
        #ax2.set_xlabel("time")
        #ax2.set_ylabel("$\sum_{i \in C} p_i(t)$")
        cps = self.synthesizer.change_points
        for cp in cps:
            plt.axvline(cp[0], color="black")
            print(cp)
        #for cp in signal.find_peaks(smoothed, prominence=0.2)[0]:
        #    ax2.axvline(cp, color="lime", alpha=0.7)
        # 
        #pix = [len(signal.find_peaks(smoothed, prominence=promi)[0]) for promi in np.linspace(0,0.9,100)]
        #print(pix)
        #plt.plot(np.linspace(0,0.9,100), pix)
        for ys in range(configs.shape[1]):
            plt.plot(feature_weights[ys,:])
        #plt.colorbar()
        plt.show()
        
#dd = np.zeros(1000)
#dd[400:] += 1
#dd[800:] += 2
#s = Transformation().transform_single(1000, {i: dd[i] for i in np.random.choice(np.arange(1000), size=50, replace=False)})


gse = GroupSamplingEstimator(synthesis.PerformanceHistorySynthesizer(seed=42))
gse.initialize(1000, 0.01, 0.5)

