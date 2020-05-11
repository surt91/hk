import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import parameters
from parameters import parameters as ps

os.makedirs("dynamics/", exist_ok=True)

cutoff = 40

for p in ps:
        
    el = p["tolerance_lower"]
    eu = p["tolerance_upper"]
    name = parameters.outname.format(**p).replace(".", "")
    basename = "n{}l{:.2f}u{:.2f}s{}eta{}".format(p["num_agents"], el, eu, p["seed"], p["eta"]).replace(".", "")
    a = np.loadtxt(name + ".density.dat") / p["samples"] / p["num_agents"]

    if len(a.shape) != 2:
        continue

    a[a==0] = np.nan

    plt.imshow(
        a[:cutoff,:].transpose(),
        cmap=plt.cm.viridis_r,
        vmin=1e-2,
        aspect="auto",
        origin='lower',
#        vmax=1,
        norm=mpl.colors.LogNorm()

    )

    print(p["num_agents"], p["eta"], el, eu)
    plt.colorbar()
    plt.clim([1e-2, 1])
    plt.xlabel("$t$")
    plt.ylabel("opinion * 100")
    plt.tight_layout()
    plt.savefig("dynamics/heatmap_" + basename)
    plt.clf()
    
