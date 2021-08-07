from glob import glob
import gzip
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt


N = 65536

def parse_file_generator(filename):
    with gzip.open(filename) as f:
        for line in f:
            if line.startswith(b"# sweeps:"):
                speed = float(line[9:])
            elif line.startswith(b"#"):
                positions = list(map(float, line[2:].split()))
            else:
                sizes = list(map(float, line.split()))

                # for every dataset (consisting of 3 lines) return one result
                yield positions, sizes, speed


def plot(filename, basename):
    sizes = Counter()
    ns = []
    for c in parse_file_generator(filename):
        for cluster in c[1]:
            #sizes[cluster // 128] += cluster
            sizes[cluster // 128] += 1
            ns.append(cluster / N)
            
#    data, borders = np.histogram(ns, bins=np.logspace(np.log10(0.001), np.log10(1.0), 50), density=True)
    data, borders = np.histogram(ns, bins=np.linspace(0, 1, 50), density=True)
    bins = (borders[1:] + borders[:-1])/2

    x = []
    y = []
    for i in range(0, 16384//128):
        x.append(i / (16384//128))
        y.append(sizes[i] / 1000)
    
    x = []
    y = []
    for c, d in zip(bins, data):
        x.append(c)
        y.append(d)
    
    plt.xlabel("S")
    plt.ylabel("n_S")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim([1e-3, 1e3])
    plt.xlim([1e-3, 1])
    plt.plot(x, y)
    plt.savefig(f"out/{basename}.png")
    plt.show()

    with open(basename, "w") as f:
        for i, j in zip(x, y):
            f. write(f"{i} {j}\n")


#plot(f"data/out_n{N}_e035_s13.cluster.dat.gz", "0350")
#plot(f"data/out_n{N}_e015_s13.cluster.dat.gz", "0150")
#plot(f"data/out_n{N}_e026_s13.cluster.dat.gz", "0260")
#plot(f"data/out_n{N}_e027_s13.cluster.dat.gz", "0270")
#plot(f"data/out_n{N}_e028_s13.cluster.dat.gz", "0280")
#plot(f"data/out_n{N}_e029_s13.cluster.dat.gz", "0290")
#plot(f"data/out_n{N}_e03_s13.cluster.dat.gz", "0300")
#plot(f"data/out_n{N}_e021_s13.cluster.dat.gz", "0210")
#plot(f"data/out_n{N}_e02_s13.cluster.dat.gz", "0200")
#plot(f"data/out_n{N}_e019_s13.cluster.dat.gz", "0190")
#plot(f"data/out_n{N}_e0202_s13.cluster.dat.gz", "0202")
plot(f"data/out_n{N}_e025_s13.cluster.dat.gz", "lin_0250.dat")
plot(f"data/out_n{N}_e026_s13.cluster.dat.gz", "lin_0260.dat")
plot(f"data/out_n{N}_e027_s13.cluster.dat.gz", "lin_0270.dat")
