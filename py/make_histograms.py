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

c_size = []
c_num = []

for i in range(p["samples"]):
    name = parameters.outname.format(seed=i, **p).replace(".", "")
    a = np.loadtxt(name + ".dat.gz")

    xh = []
    yh = []
    zh = []
    for point in a:
        pt = rotate_to_xy_plane(point+np.array([-1, 0, 0]))
        # assert that the z-component is zero
        assert(abs(pt[-1]) < 1e-3)
        # assert that y is positive
        assert(pt[1] > -1e-8)
        xh.append(pt[0])
        yh.append(pt[1])
        zh.append(pt[2])

    x.extend(xh)
    y.extend(yh)
    z.extend(zh)

    b = np.loadtxt(name + ".cluster.dat.gz")
    try:
        len(b)
    except:
        b = [b]
    assert(sum(b) == p["num_agents"])
    c_size.extend(b)
    c_num.append(len(b))

    # plt.hist2d(
    #     xh,
    #     yh,
    #     bins=(50, 50),
    #     range=[[-1.5, 0], [0., 1.5]],
    #     cmap=plt.cm.BuPu,
    #     vmin=0.1,
    #     vmax=p["num_agents"],
    #     norm=matplotlib.colors.LogNorm()
    # )
    # plt.plot([0, -0.7071, -1.41421, 0], [0, 1.2247, 0, 0], "-")
    # print(set("%.4f"%i for i in xh), set("%.4f"%i for i in yh), set("%.4f"%i for i in zh))
    # print(b)
    # plt.show()

# plt.hexbin(x, y, gridsize=(15,15))
plt.hist2d(
    x,
    y,
    bins=(50, 50),
    range=[[-1.5, 0], [0., 1.3]],
    cmap=plt.cm.viridis_r,
    norm=matplotlib.colors.LogNorm(),
    density=True
)
plt.colorbar()
plt.show()

print(c_size)
plt.hist(
    c_size
)
plt.show()

print(c_num)
plt.hist(
    c_num
)
plt.show()

# assert(data == np.matmul(r, a))
# print(a)
# print(rotate_to_xy_plane(np.array([-1, 0, 1])))
# print(rotate_to_xy_plane(a[0]+np.array([-1, 0, 0])))
