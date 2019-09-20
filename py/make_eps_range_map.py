import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import parameters
from parameters import parameters as ps

xs = []
ys = []
zs = []

for p in ps:
    el = p["tolerance_lower"]
    eu = p["tolerance_upper"]
    name = parameters.outname.format(**p).replace(".", "")
    try:
        b = np.loadtxt(name + ".cluster.dat.gz")
    except:
        print("file not found:", name + ".cluster.dat.gz")
        continue

    for a in b:
        try:
            len(a)
        except:
            val = a
        else:
            val = max(a)
        xs.append(el)
        ys.append(eu)
        zs.append(val)

plt.hist2d(
    xs,
    ys,
    weights=zs,
    bins=(len(set(xs)), len(set(ys))),
    cmap=plt.cm.viridis_r,
    norm=matplotlib.colors.LogNorm(),
    density=True
)
plt.show()
