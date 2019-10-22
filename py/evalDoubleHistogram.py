import gzip

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import parameters
from parameters import parameters as ps

for p in ps:
    el = p["tolerance_lower"]
    eu = p["tolerance_upper"]
    name = parameters.outname.format(**p).replace(".", "")
    bincounts = []
    with gzip.open(name + ".cluster.dat.gz") as f:
        while(True):
            # read two lines, first (beginning with #) is data, second is weight
            l1 = f.readline().decode()
            l2 = f.readline().decode()
            if not l1:
                break
            data = list(map(float, l1.strip("#").strip().split()))
            weight = list(map(float, l2.strip().split()))
            
            # fill it into a new histogram and 
            bins, borders = np.histogram(data, bins=9, range=(0, 1), weights=weight)
            
            # then fill the bincounts into a persistent histogram, which will be output
            bincounts.extend(bins)
            
    print(p["num_agents"], p["eta"])
    plt.xlabel("cluster size $S$")
    
    bincounts = np.array(bincounts)
    
    if(False):
        plt.ylabel("probability of realization to contain a cluster of size $S$")
        data, borders, _ = plt.hist(bincounts, weights=np.ones(bincounts.shape)/p["samples"], bins=32, range=(1e-6, p["num_agents"]))
    else:
        plt.ylabel("probability for an agent to end in a cluster of size $S$")
        data, borders, _ = plt.hist(bincounts, weights=bincounts/np.sum(bincounts), bins=32, range=(1e-6, p["num_agents"]))
        
    plt.savefig("n{}_eta{}.png".format(p["num_agents"], p["eta"]))
    plt.clf()
    #plt.show()
    
    print(sum(data))
    
    centers = (borders[1:]+borders[:-1]) / 2
    with open("n{}_eta{}.dat".format(p["num_agents"], p["eta"]), "w") as f:
        for d, c in zip(data, centers):
            f.write("{} {}\n".format(c, d))
    
