import os
import gzip

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('text', usetex=True)

import parameters
from parameters import parameters as ps

os.makedirs("plots", exist_ok=True)

for p in ps:
    el = p["tolerance_lower"]
    eu = p["tolerance_upper"]
    name = parameters.outname.format(**p).replace(".", "")
    bincounts = []
    num_clusters = []
    aggregate_dists = np.zeros(100)
    entropies = []
    with gzip.open(name + ".cluster.dat.gz") as f:
        while(True):
            # read two lines, first (beginning with #) is data, second is weight
            l1 = f.readline().decode()
            if "sweep" in l1:
                l1 = f.readline().decode()
            l2 = f.readline().decode()
            if not l1:
                break
            data = list(map(float, l1.strip("#").strip().split()))
            weight = list(map(float, l2.strip().split()))
            
            ent_data, ent_borders = np.histogram(data, bins=100, range=(0,1), weights=weight, density=False)
            ent_data /= sum(weight)
            centers = (ent_borders[1:] + ent_borders[:-1])

            S = 0
            for tmp in data:
                if tmp > 0:
                    S += -tmp * np.log(tmp)
            entropies.append(S)
            
            aggregate_dist, borders2 = np.histogram(data, bins=100, range=(0, 1), weights=weight)
            aggregate_dists += aggregate_dist

            if(True):
                # fill it into a new histogram and
                cluster_sizes, borders = np.histogram(data, bins=9, range=(0, 1), weights=weight)
            else:
                # cluster the date with an adaptive thresholding method
                # sort all opinions, measure all distances
                s = sorted(list(zip(data, weight)))
                s = [[0, 0]] + s + [[1, 0]]
                s_data = [i[0] for i in s]
                distances = np.abs(np.diff(s_data))
                threshold = np.max(distances) / 2
                # threshold = 10 * np.mean(distances)

                # threshold = 0
                # max_factor = 0
                # sdist = sorted(distances)
                # for n in range(len(distances)-1):
                #     i = sdist[n]
                #     j = sdist[n+1]
                #     factor = j/i
                #     if factor > max_factor:
                #         threshold = j
                #         max_factor = factor
                # threshold *= 2

                cluster_sizes = []
                size = s[0][1]
                for n, i in enumerate(distances, start=1):
                    if i > threshold:
                        cluster_sizes.append(size)
                        size = 0
                    size += s[n][1]

            cluster_sizes = [i for i in cluster_sizes if i > 0]
            if len(cluster_sizes) == 0:
                cluster_sizes = [sum(weight)]
            # then fill the bincounts into a persistent histogram, which will be output
            bincounts.extend(cluster_sizes)
            num_clusters.append(len(cluster_sizes))

#            print(len(cluster_sizes), threshold)
#            plt.hist(data, 100, weights=weight, range=(0, 1))
#            plt.show()
#            plt.clf()

    print(p["num_agents"], p["eta"])
    print("clusternumber", np.mean(num_clusters), np.std(num_clusters)/np.sqrt(len(num_clusters)))
    plt.xlabel("cluster size $S$")

    bincounts = np.array(bincounts)

    if(False):
        plt.ylabel("probability of realization to contain a cluster of size $S$")
        data, borders, _ = plt.hist(bincounts, weights=np.ones(bincounts.shape)/p["samples"], bins=32, range=(1e-6, p["num_agents"]))
    else:
        plt.ylabel("probability for an agent to end in a cluster of size $S$")
        data, borders, _ = plt.hist(bincounts, weights=bincounts/np.sum(bincounts), bins=32, range=(1e-6, p["num_agents"]))

    plt.savefig("plots/n{}_eta{}.png".format(p["num_agents"], p["eta"]))
    #plt.show()
    plt.clf()
    
    plt.gcf().set_size_inches(3+3/8, 2.5)
    
    plt.xlabel("$x$")
    plt.ylabel("$P(x)$")
    
    plt.plot(aggregate_dists / p["num_agents"] / p["samples"])
    plt.savefig("plots/agg_n{}_eta{}.png".format(p["num_agents"], p["eta"]))
    plt.tight_layout()
    #plt.show()
    plt.clf()
    
    
    plt.gcf().set_size_inches(3+3/8, 2.5)
    
    plt.xlabel("$S$")
    plt.ylabel("$P(S)$")
    
    plt.hist(entropies)
    plt.savefig("plots/entropy_dist_n{}_eta{}.png".format(p["num_agents"], p["eta"]))
    plt.tight_layout()
    #plt.show()
    plt.clf()

    print(sum(data))

    centers = (borders[1:]+borders[:-1]) / 2
    with open("plots/n{}_eta{}.dat".format(p["num_agents"], p["eta"]), "w") as f:
        for d, c in zip(data, centers):
            f.write("{} {}\n".format(c, d))
            
    centers = (borders2[1:]+borders2[:-1]) / 2
    with open("plots/agg_n{}_eta{}.dat".format(p["num_agents"], p["eta"]), "w") as f:
        for d, c in zip(aggregate_dists, centers):
            f.write("{} {}\n".format(c, d))
