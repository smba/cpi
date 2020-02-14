import numpy as np
import itertools
import matplotlib.pyplot as plt
import scipy.stats as stats
import synthesis
import pandas as pd
import sklearn.utils as utils
import sklearn.linear_model as lm
import seaborn as sns

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
            
class Estimator():
    
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
            #print(i)
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
        
        """
        # Variant 1: Step-wise correction with bootstrappingxtr tracer
        feature_weights = np.zeros((change_map.shape[1], configs.shape[1]))
        feature_iqr = np.zeros((change_map.shape[1], configs.shape[1]))
        for rev in range(change_map.shape[1]):
            xs = configs
            ys = change_map[:,rev]
            boot_xs = [utils.resample(xs, replace=True, n_samples=len(xs) // 2, random_state=i) for i in range(50)]
            boot_ys = [utils.resample(ys, replace=True, n_samples=len(ys) // 2, random_state=i) for i in range(50)]
            
            model = lm.LinearRegression()
            weights = []
            for i in range(50):
                print("fit", rev)
                model.fit(boot_xs[i], boot_ys[i])
                weights.append(model.coef_)
                
            weights = np.array(weights)
            mean = np.median(weights, axis=0)
            iqr = stats.iqr(weights, axis=0)
            
            feature_weights[rev,:] = mean
            feature_iqr[rev,:] = iqr
        
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.pcolormesh(feature_weights)
        #ax2.pcolormesh(feature_iqr)    print(i)
        #plt.show()
        #return change_map
        """
        feature_weights1 = np.zeros((configs.shape[1], n_revisions))
        feature_weights0 = np.zeros((configs.shape[1], n_revisions))
        for f in range(configs.shape[1]):
            indexes1 = np.argwhere(configs[:,f] == 1)
            indexes0 = np.argwhere(configs[:,f] == 0)
            mean1 = np.mean(change_map[indexes1,:], axis=0)
            mean0 = np.mean(change_map[indexes0,:], axis=0)
            feature_weights1[f] = mean1
            feature_weights0[f] = mean0#feature_weights
            
        feature_weights = feature_weights1 - feature_weights0
        
        """for fw in feature_weights:
            plt.plot(fw)
            
        for cp in self.synthesizer.change_points:
            plt.axvline(cp[0])"""
        return feature_weights

