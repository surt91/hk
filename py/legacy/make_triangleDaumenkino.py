import sys
from glob import glob

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from rotate import rotate_to_xy_plane
# can also use matplotlib.pyplot.hexbin

a = np.loadtxt(sys.argv[1])
for n, line in enumerate(a):
    ctr = 0
    xs = []
    ys = []
    while ctr < len(line):
        x = line[ctr]
        y = line[ctr+1]
        z = line[ctr+2]
        ctr += 3
        pt = rotate_to_xy_plane(np.array([x, y, z])+np.array([-1, 0, 0]))
        xs.append(pt[0])
        ys.append(pt[1])
        plt.plot(xs, ys, "+")
        circle = plt.Circle((pt[0], pt[1]), 0.23, color='b', fill=False)
        ax = plt.gca()
        ax.add_artist(circle)
    plt.plot([0, -0.7071, -1.41421, 0], [0, 1.2247, 0, 0], "-")
    # plt.show()
    plt.savefig("p{}.png".format(n))
    plt.clf()
