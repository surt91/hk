import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import parameters
from parameters import parameters as p

whole_matrix = np.zeros((p["iterations"], 100))
for i in range(p["samples"]):
    name = parameters.outname.format(seed=i, **p).replace(".", "") + ".density.dat"
    a = np.loadtxt(name)
    whole_matrix += a

plt.imshow(
    whole_matrix.transpose(),
    cmap=plt.cm.viridis_r,
    norm=matplotlib.colors.LogNorm()
)
plt.colorbar()
plt.show()
