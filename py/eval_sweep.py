import gzip

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import parameters
from parameters import parameters as ps


for p in ps:
    n = p["num_agents"]
    eu = p["tolerance_upper"]

    name = parameters.outname.format(**p).replace(".", "")

    try:
        with gzip.open(name + ".cluster.dat.gz") as f:
            sizes = []
            speeds = []
            for line in f:
                if line.startswith(b"#"):
                    if b"sweep" in line:
                        speeds.append(int(line.split()[-1]))
                    continue
                a = list(map(float, line.split()))

                try:
                    len(a)
                except:
                    val = a
                else:
                    val = max(a)

                sizes.append(val/n)
        with open("d{}.dat".format(n), "a") as f:
            f.write("{} {} {}\n".format(eu, np.mean(sizes), np.std(sizes)/len(sizes)**0.5))
        with open("s{}.dat".format(n), "a") as f:
            f.write("{} {} {}\n".format(eu, np.mean(speeds), np.std(speeds)/len(sizes)**0.5))
    except OSError:
        print("# {} missing".format(name))
