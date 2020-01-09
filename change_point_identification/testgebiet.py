import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm

np.random.seed(1224)

class Target:
    def __init__(self, influential):
        self.f = lambda xs: np.sum([np.prod(xs[factors]) for factors in influential])
      
      
def ofat(k: int):
    xs = np.zeros((2*k, k))
    for i in range(k):
        xs[i][0:i+1] = 1
    xs[k:][:] = 1
    for i in range(k):
        xs[i+k][1:i+1] = 0
    return(np.unique(xs, axis=0))
   
target = Target([[450, 460]])
tcount = 0

# make groupings
f = 16
groupings = [np.array_split(np.random.choice(np.arange(1024), size=1024, replace=False), f) for i in range(f)]

# make 
weights = []

print(ofat(f).shape)
for g in range(f):
    xs = []
    for i, r in enumerate(ofat(f)):
        x = np.zeros(1024)
        for group in np.where(r == 1)[0]:
            x[groupings[g][group]] = 1
        xs.append(x)
    xs = np.vstack(xs)


    ys = [target.f(x) for x in xs]
    tcount += len(ys)
    X = ofat(f)
    
    linmod = lm.LinearRegression()
    linmod.fit(X, ys)  
    
    weight = np.zeros(1024)
    for i in range(linmod.coef_.shape[0]):
        weight[groupings[g][i]] = linmod.coef_[i]
    #plt.bar(np.arange(1024), weight)
    #plt.show()
    weights.append(weight)

weights = np.vstack(weights)

#signs = np.random.choice([-1,1], size=weights.shape)
#weights = np.multiply(signs, weights)
weights = np.mean(weights, axis=0)
plt.fill_between(np.arange(1024), np.zeros(1024), weights / f, color="midnightblue")
#plt.axhline(1.0/f)
#plt.axhline(0.5)
print(tcount)
plt.show()
            
            
            