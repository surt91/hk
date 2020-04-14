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


def bootstrap(xRaw, f=np.mean, n_resample=100):
    """Bootstrap resampling, returns mean and stderror"""
    if not len(xRaw):
        return float("NaN"), float("NaN")
    bootstrapSample = [f(np.random.choice(xRaw, len(xRaw), replace=True)) for i in range(n_resample)]
    return np.mean(bootstrapSample), np.std(bootstrapSample)


os.makedirs("plots", exist_ok=True)

for p in ps:
    it = p["samples"]
    el = p["tolerance_lower"]
    eu = p["tolerance_upper"]
    name = parameters.outname.format(**p).replace(".", "")
    bincounts = []
    num_clusters = []
    aggregate_dists = np.zeros(100)
    entropies = []
    ctr = 0
    all_cluster_sizes = []
    all_cluster_widths = []
    try:
        with gzip.open(name + ".cluster.dat.gz") as f:
            while(True):
                ctr += 1
                # read two lines, first (beginning with #) is data, second is weight
                l1 = f.readline().decode()
                if "sweep" in l1:
                    l1 = f.readline().decode()
                l2 = f.readline().decode()
                if not l1:
                    break
                raw_data = list(map(float, l1.strip("#").strip().split()))
                weight = list(map(float, l2.strip().split()))

                # ensure to start with 0 bins, by going beyond the range (otherwise we might start above the cut and our detection will not work, and trigger the assert)
                data, borders = np.histogram(raw_data, bins=102, range=(-0.01,1.01), weights=weight, density=False)
                data /= sum(weight)
                centers = (borders[1:] + borders[:-1])/2

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

                all_cluster_sizes.append(cluster_sizes)
                all_cluster_widths.append(cluster_widths)

                if ctr < 5:
                    plt.plot(centers, data)
                    plt.plot(centers, [peak_height/3. for _ in centers])
                    plt.plot(all_borders, [peak_height/3. for _ in all_borders], "o")
                    plt.savefig("plots/fwhp_n{}_eta{}_ctr{}.png".format(p["num_agents"], p["eta"], ctr))
                    plt.tight_layout()
                    #plt.show()
                    plt.clf()

    except:
        print("error for:", name)

    # TODO: collect statistics of the clustersizes
    # TODO: plot the statistics

    mean_num_clusters, mean_num_clusters_err = bootstrap([len(x) for x in all_cluster_sizes])
    mean_largest_cluster_size, mean_largest_cluster_size_err = bootstrap([max(x) for x in all_cluster_sizes])
    mean_largest_cluster_size_relative, mean_largest_cluster_size_relative_err = bootstrap([max(x) / sum(x) for x in all_cluster_sizes])
    mean_largest_cluster_width, mean_largest_cluster_width_err = bootstrap([y[np.argmax(x)] for x, y in zip(all_cluster_sizes, all_cluster_widths)])

    var_num_clusters, var_num_clusters_err = bootstrap([len(x) for x in all_cluster_sizes], np.var)
    var_largest_cluster_size, var_largest_cluster_size_err = bootstrap([max(x) for x in all_cluster_sizes], np.var)
    var_largest_cluster_size_relative, var_largest_cluster_size_relative_err = bootstrap([max(x) / sum(x) for x in all_cluster_sizes], np.var)
    var_largest_cluster_width, var_largest_cluster_width_err = bootstrap([y[np.argmax(x)] for x, y in zip(all_cluster_sizes, all_cluster_widths)], np.var)

    # print(
    #     p["num_agents"], p["eta"],
    #     mean_num_clusters, mean_num_clusters_err,
    #     mean_largest_cluster_size, mean_largest_cluster_size_err,
    #     mean_largest_cluster_size_relative, mean_largest_cluster_size_relative_err,
    #     mean_largest_cluster_width, mean_largest_cluster_width_err
    # )

    with open("plots/fwhp_n{}.dat".format(p["num_agents"]), "a") as f:
        f.write("{} {} {} {} {} {} {} {} {}\n".format(
            p["eta"],
            mean_num_clusters, mean_num_clusters_err,
            mean_largest_cluster_size, mean_largest_cluster_size_err,
            mean_largest_cluster_size_relative, mean_largest_cluster_size_relative_err,
            mean_largest_cluster_width, mean_largest_cluster_width_err,
            var_num_clusters, var_num_clusters_err,
            var_largest_cluster_size, var_largest_cluster_size_err,
            var_largest_cluster_size_relative, var_largest_cluster_size_relative_err,
            var_largest_cluster_width, var_largest_cluster_width_err,
        ))

#    plt.gcf().set_size_inches(3+3/8, 2.5)
#
#    plt.xlabel("$x$")
#    plt.ylabel("$P(x)$")
#
#    plt.plot(aggregate_dists / p["num_agents"] / p["samples"])
#    plt.savefig("plots/fwhp_n{}_eta{}.pdf".format(p["num_agents"], p["eta"]))
#    plt.tight_layout()
#    #plt.show()
#    plt.clf()
#
#    with open("plots/fwhp_n{}_eta{}.dat".format(p["num_agents"], p["eta"]), "w") as f:
#        for d, c in zip(data, centers):
#            f.write("{} {}\n".format(c, d))
