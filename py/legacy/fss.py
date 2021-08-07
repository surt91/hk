import gzip

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import parameters
from parameters import parameters as ps


for p in ps:
    n = p["num_agents"]

    name = parameters.outname.format(**p).replace(".", "")

    with gzip.open(name + ".cluster.dat.gz") as f:
        sizes = []
        for line in f:
            if line.startswith(b"#"):
                continue
            a = list(map(float, line.split()))

            try:
                len(a)
            except:
                val = a
            else:
                val = max(a)

            sizes.append(val/n)

    print(n, np.mean(sizes), np.std(sizes)/n**0.5)
