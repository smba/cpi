import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm

# FLAGS
SIGN_VARIABLES = False
SEQUENTIAL_ANALAYSIS = False

np.random.seed(1852)

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
   
target = Target([[300,600,520]])
tcount = 0

# make groupings123
f = 64
groupings = [np.array_split(np.random.choice(np.arange(1024), size=1024, replace=False), f) for i in range(f)]



if SIGN_VARIABLES:
    signs = []

# make 
weights = []
coefs = []
for g in range(f):
    
    if SEQUENTIAL_ANALAYSIS and len(weights) > 0:
        max_influential = list(reversed(np.argsort(coefs[-1])))[-1]
        # divide group max influential from previous grouping to G_1... G_{f//2}
        
        grouping = []
        for i, group in enumerate(np.array_split(groupings[g][max_influential], f//2)):
            grouping.append(group)
        
        non_influential = []
        for i in range(f):
            if i != max_influential:
                non_influential += list(groupings[g][i])
                
        non_influential = np.array(non_influential)
        np.random.shuffle(non_influential)
        
        for i in range(f//2):
            add = 1024 // f - 1024 // 16 // (16//2)
            grouping[i] = np.append(grouping[i], non_influential[:add])
            non_influential = non_influential[add:]
            
        for i in np.array_split(non_influential, f//2):
            grouping.append(i)
            
        groupings[g] = grouping
        
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

    coefs.append(linmod.coef_)
    weights.append(weight)

weights = np.vstack(weights)

if SIGN_VARIABLES:
    signs = np.vstack(signs)
    weights = np.multiply(signs, weights)

weights_mean = np.mean(weights, axis=0)

influential = list(reversed(np.argsort(weights_mean)[-3:]))

#plt.plot(np.arange(1024), weights_mean, alpha=0.8)  
for i in influential:
    weights[:, i] =  weights[:, i] - weights_mean[i]
    for m, grouping in enumerate(groupings):
        g = None
        for group in grouping:
            if i in group:
                g = group
        g = np.delete(g, np.where(g == i))
        weights[m, g] = weights[m, g] - weights_mean[i] 
    
    weightz_mean = np.mean(weights, axis=0)
print(tcount)
plt.plot(np.arange(1024), weights_mean, alpha=0.8)   
#plt.fill_between(np.arange(1024), np.full(1024, -0.2), np.full(1024, 0.2), color="red", alpha=0.7)
#plt.fill_between(np.arange(1024), np.full(1024, 0.2), np.full(1024, 0.5), color="yellow", alpha=0.7)
#plt.fill_between(np.arange(1024), np.full(1024, 0.5), np.full(1024, 1), color="green", alpha=0.7)
#plt.axhline(0.5, color="black", linewidth=0.1)
plt.show()
            
            
            