import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm

# FLAGS
SIGN_VARIABLES = False

np.random.seed(629)

class Target:
    def __init__(self, influential):
        self.f = lambda xs: np.sum([np.prod(xs[factors]) for factors in influential])
      
def flip(matrix: np.ndarray, col: int):
    matrix[:, col] = np.abs( matrix[:, col] - 1 ) 
    return matrix
      
def ofat(k: int):
    xs = np.zeros((2*k, k))
    for i in range(k):
        xs[i][0:i+1] = 1
    xs[k:][:] = 1
    for i in range(k):
        xs[i+k][1:i+1] = 0
    return(np.unique(xs, axis=0))
   
target = Target([[300,600], 900])
tcount = 0

# make groupings123
f = 16
groupings = [np.array_split(np.random.choice(np.arange(1024), size=1024, replace=False), f) for i in range(f)]

# make 
weights = []

if SIGN_VARIABLES:
    signs = []

for g in range(f):
    xs = []
    
    if SIGN_VARIABLES:
        sign = np.random.choice([0,1], size=1024)
        
    for i, r in enumerate(ofat(f)):
        x = np.zeros(1024)
        for group in np.where(r == 1)[0]:
            x[groupings[g][group]] = 1
        xs.append(x)
    xs = np.vstack(xs)

    # do the flip
    if SIGN_VARIABLES:
        for i in range(1024):
            if sign[i] == 0:
                xs = flip(xs, i)
    
    if SIGN_VARIABLES:
        signs.append(sign)

    ys = [target.f(x) for x in xs]
    tcount += len(ys)
    X = ofat(f)
    
    linmod = lm.LinearRegression()
    linmod.fit(X, ys)  
    
    weight = np.zeros(1024)
    for i in range(linmod.coef_.shape[0]):
        weight[groupings[g][i]] = linmod.coef_[i]

    weights.append(weight)

weights = np.vstack(weights)

if SIGN_VARIABLES:
    signs = np.vstack(signs)
    weights = np.multiply(signs, weights)

weights_mean = np.mean(weights, axis=0)

influential = list(reversed(np.argsort(weights_mean)[-3:]))

for i in influential:
    print(i, weights_mean[i])
    weights[:, i] =  weights[:, i] - weights_mean[i]
    for m, grouping in enumerate(groupings):
        g = None
        for group in grouping:
            if i in group:
                g = group
        g = np.delete(g, np.where(g == i))
        weights[m, g] = weights[m, g] - weights_mean[i] 
    
    weightz_mean = np.mean(weights, axis=0)
    #plt.pcolormesh(weights)
    plt.fill_between(np.arange(1024), np.zeros(1024), weightz_mean, alpha=0.7)
    

#plt.fill_between(np.arange(1024), np.full(1024, -0.2), np.full(1024, 0.2), color="red", alpha=0.1)
#plt.fill_between(np.arange(1024), np.full(1024, 0.2), np.full(1024, 0.5), color="yellow", alpha=0.1)
#plt.fill_between(np.arange(1024), np.full(1024, 0.5), np.full(1024, 1), color="green", alpha=0.1)
#plt.axhline(0.5, color="black", linewidth=0.1)
plt.show()
            
            
            