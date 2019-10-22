from glob import glob
import gzip

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from rotate import rotate_to_xy_plane
# can also use matplotlib.pyplot.hexbin

import parameters
from parameters import parameters as ps

x = []
y = []
z = []
w = []
for p in ps:

    el = p["tolerance_lower"]
    eu = p["tolerance_upper"]
    dim = p["dimension"]
    name = parameters.outname.format(**p).replace(".", "")
    # points = []
    # weights = []
    line_num = 0
    with gzip.open(name + ".cluster.dat.gz") as f:
        while True:
            line_num += 1
            line = f.readline()[1:].strip().split()
            if not line:
                break
            i = 0
            print(len(line))
            while i+dim <= len(line):
                pt = rotate_to_xy_plane(np.array(list(map(float, line[i:i+dim])))+np.array([-1, 0, 0]))
                x.append(pt[0])
                y.append(pt[1])
                z.append(pt[2])
                i += dim
            line = f.readline().strip().split()
            for weight in line:
                w.append(float(weight))

    # print(x, y, w)
    # print(len(x), len(y), len(w))

plt.hist2d(
    x,
    y,
    weights=w,
    bins=(50, 50),
    range=[[-1.5, 0], [0., 1.3]],
    cmap=plt.cm.viridis_r,
    norm=matplotlib.colors.LogNorm(),
    density=True
)
plt.colorbar()
plt.savefig(name + ".triangle.png")
plt.show()
plt.clf()
