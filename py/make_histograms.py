from glob import glob

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from rotate import rotate_to_xy_plane
# can also use matplotlib.pyplot.hexbin

import parameters
from parameters import parameters as p

x = []
y = []
z = []

for i in range(p["samples"]):
    name = parameters.outname.format(seed=i, **p).replace(".", "")
    a = np.loadtxt(name + ".dat.gz")

    for point in a:
        pt = rotate_to_xy_plane(point+np.array([-1, 0, 0]))
        # assert that the z-component is zero
        assert(pt[-1] < 1e-8)
        # assert that y is positive
        assert(pt[1] > -1e-8)
        x.append(pt[0])
        y.append(pt[1])
        z.append(pt[2])

# plt.hexbin(x, y, gridsize=(15,15))
plt.hist2d(
    x,
    y,
    bins=(50, 50),
    range=[[-1.5, 0], [0., 1]],
    cmap=plt.cm.BuPu,
    norm=matplotlib.colors.LogNorm()
)
plt.show()

# assert(data == np.matmul(r, a))
# print(a)
# print(rotate_to_xy_plane(np.array([-1, 0, 1])))
# print(rotate_to_xy_plane(a[0]+np.array([-1, 0, 0])))
