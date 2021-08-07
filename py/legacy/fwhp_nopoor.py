import os
import gzip

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import interp1d

matplotlib.rc('text', usetex=True)

import parameters
from parameters import parameters as ps

os.makedirs("plots", exist_ok=True)

BINS = 100

for p in ps:
    it = p["samples"]
    el = p["tolerance_lower"]
    eu = p["tolerance_upper"]
    name = parameters.outname.format(**p).replace(".", "")
    bincounts = []
    num_clusters = []
    aggregate_dists = np.zeros(BINS)
    all_data = np.zeros(BINS)
    all_data_p = np.zeros(BINS)
    all_data_np = np.zeros(BINS)
    entropies = []
    ctr = 0
    all_cluster_sizes = []
    all_cluster_sizes_np = []
    all_cluster_widths = []
    all_num_p = []
    try:
        with gzip.open(name + ".cluster.dat.gz") as f:
            g = gzip.open(name + ".nopoor.dat.gz")
            while(True):
                ctr += 1
                # read two lines, first (beginning with #) is data, second is weight
                l1 = f.readline().decode()
                if "sweep" in l1:
                    l1 = f.readline().decode()
                l2 = f.readline().decode()
                if not l1:
                    break
                k1 = g.readline().decode()
                if "sweep" in k1:
                    k1 = g.readline().decode()
                k2 = g.readline().decode()
                if not k1:
                    break
                raw_data = list(map(float, l1.strip("#").strip().split()))
                weight = list(map(float, l2.strip().split()))
                
                raw_data_np = list(map(float, k1.strip("#").strip().split()))
                weight_np = list(map(float, k2.strip().split()))
                
                data, borders = np.histogram(raw_data, bins=BINS, range=(0,1), weights=weight, density=False)
                centers = (borders[1:] + borders[:-1])/2
                data_np, borders_np = np.histogram(raw_data_np, bins=BINS, range=(0,1), weights=weight_np, density=False)
                data_p = data - data_np
                
                w = sum(weight)
                w_np = sum(weight_np)
                data /= w
                data_np /= w
                data_p /= w
                
                num_p = w - w_np
                
                # interpolate the data linearly
                peak_height = np.max(data)
                fn = interp1d(centers, data - peak_height / 3.)

                changed_sign = [centers[0]]
                last_sign = np.sign(data[0] - peak_height / 3.)
                for x, y in zip(centers, data - peak_height / 3.):
                    if last_sign != np.sign(y) and np.sign(y) != 0:
                        changed_sign.append(x)
                        last_sign = np.sign(y)

                all_borders = []
                for (a, b) in zip(changed_sign[0:], changed_sign[1:]):
                    r = brentq(fn, a, b)
                    all_borders.append(r)

                all_borders.sort()
                if len(all_borders) % 2 != 0:
                    print(changed_sign)
                    print(all_borders)

                assert(len(all_borders) % 2 == 0)
                cluster_borders = [(r1, r2) for (r1, r2) in zip(all_borders[0::2], all_borders[1::2])]
                cluster_widths = [r2 - r1 for (r1, r2) in cluster_borders]
                cluster_sizes = [sum([weight if r1 < pos < r2 else 0 for (pos, weight) in zip(raw_data, weight)]) for (r1, r2) in cluster_borders]
                cluster_sizes_np = [sum([weight if r1 < pos < r2 else 0 for (pos, weight) in zip(raw_data_np, weight_np)]) / w_np for (r1, r2) in cluster_borders]
                
                all_cluster_sizes.append(cluster_sizes)
                all_cluster_sizes_np.append(cluster_sizes_np)
                all_cluster_widths.append(cluster_widths)
                all_num_p.append(num_p)
                
                all_data += data
                all_data_p += data_p
                all_data_np += data_np

                if ctr < -1:
                    plt.plot(centers, data)
                    plt.plot(centers, data_np)
                    plt.plot(centers, data_p)
                    plt.plot(centers, [peak_height/3. for _ in centers])
                    plt.plot(all_borders, [peak_height/3. for _ in all_borders], "o")
                    plt.savefig("plots/fwhp_n{}_eta{}_ctr{}.png".format(p["num_agents"], p["eta"], ctr))
                    plt.tight_layout()
                    #plt.show()
                    plt.clf()
    except KeyboardInterrupt: 
        raise
    except Exception as e:
        print(e, " -- error in:", name) 


    # TODO: collect statistics of the clustersizes
    # TODO: plot the statistics

    mean_num_clusters = np.mean([len(x) for x in all_cluster_sizes])
    mean_num_clusters_err = np.std([len(x) for x in all_cluster_sizes]) / it**0.5
    mean_largest_cluster_size = np.mean([max(x) for x in all_cluster_sizes])
    mean_largest_cluster_size_err = np.std([max(x) for x in all_cluster_sizes]) / it**0.5
    mean_largest_cluster_size_np = np.mean([max(x) for x in all_cluster_sizes_np])
    mean_largest_cluster_size_np_err = np.std([max(x) for x in all_cluster_sizes_np]) / it**0.5
    mean_largest_cluster_size_relative = np.mean([max(x) / sum(x) for x in all_cluster_sizes])
    mean_largest_cluster_size_relative_err = np.std([max(x) / sum(x) for x in all_cluster_sizes]) / it**0.5
    mean_largest_cluster_width = np.mean([y[np.argmax(x)] for x, y in zip(all_cluster_sizes, all_cluster_widths)])
    mean_largest_cluster_width_err = np.std([y[np.argmax(x)] for x, y in zip(all_cluster_sizes, all_cluster_widths)]) / it**0.5
    mean_num_p = np.mean(all_num_p)
    mean_num_p_err = np.std(all_num_p) / it**0.5

    print(
        p["num_agents"], p["eta"],
        mean_num_clusters, mean_num_clusters_err,
        mean_largest_cluster_size, mean_largest_cluster_size_err,
        mean_largest_cluster_size_np, mean_largest_cluster_size_np_err,
        mean_largest_cluster_size_relative, mean_largest_cluster_size_relative_err,
        mean_largest_cluster_width, mean_largest_cluster_width_err, 
        mean_num_p, mean_num_p_err,
    )

    with open("plots/fwhp_np_n{}.dat".format(p["num_agents"]), "a") as f:
        f.write("{} {} {} {} {} {} {} {} {} {} {}\n".format(
            p["eta"],
            mean_num_clusters, mean_num_clusters_err,
            mean_largest_cluster_size, mean_largest_cluster_size_err,
            mean_largest_cluster_size_relative, mean_largest_cluster_size_relative_err,
            mean_largest_cluster_width, mean_largest_cluster_width_err,
            mean_num_p, mean_num_p_err,
        ))
    
    a = np.trapz(all_data, centers)
    with open("plots/hist_n{}_eta{}.dat".format(p["num_agents"], p["eta"]), "w") as f:
        f.write("# center all_count all_normsum all_normint with_count with_normsum with_normint poor_count poor_normsum poor_normint\n") 
        for (c, d1, d2, d3) in zip(centers, all_data, all_data_np, all_data_p):
            f.write("{} {} {} {} {} {} {} {} {} {}\n".format(
                c, d1, d1/ctr, d1/a, d2, d2/ctr, d2/a, d3, d3/ctr, d3/a,
            ))
            
    continue

    plt.gcf().set_size_inches(3+3/8, 2.5)

    plt.xlabel("$x$")
    plt.ylabel("$P(x)$")
    
        
    plt.plot(centers, all_data/ctr, label="all")
    plt.plot(centers, all_data_np/ctr, label="with")
    plt.plot(centers, all_data_p/ctr, label="without")
    
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/fwhp_n{}_eta{}.png".format(p["num_agents"], p["eta"], ctr), dpi=300)
    
    #plt.show()
    plt.clf()

#    with open("plots/fwhp_n{}_eta{}.dat".format(p["num_agents"], p["eta"]), "w") as f:
#        for d, c in zip(data, centers):
#            f.write("{} {}\n".format(c, d))
