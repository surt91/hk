import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import gzip

import parameters
from parameters import parameters as ps

matplotlib.rc('text', usetex=True)

xs = []
ys = []
zs = []

for p in ps:
    el = p["tolerance_lower"]
    eu = p["tolerance_upper"]
    name = parameters.outname.format(**p).replace(".", "")
    with gzip.open(name + ".cluster.dat.gz") as f:
        for line in f:
            if line.startswith(b"# sweeps"):
                continue
            if line.startswith(b"#"):
                opinions = list(map(float, line.split()))
                continue
            a = list(map(float, line.strip(b"#").strip().split()))

            try:
                len(a)
            except:
                val = a
            else:
                final_opinion = opinions[np.argmax(a)]
                val = max(a)
                # if another cluster is closer than eps_u in opinion, merge its mass
                for s, o in zip(a, opinions):
                    if abs(o - final_opinion) < eu:
                        val += s
            xs.append(el)
            ys.append(eu)
            zs.append(val/p["num_agents"]/p["samples"])

xsort = sorted(set(xs))
xbins = np.linspace(xsort[0]-(xsort[1]-xsort[0])/2, xsort[-1]+(xsort[-1]-xsort[-2])/2, len(set(xs))+1, endpoint=True)
ysort = sorted(set(ys))
ybins = np.linspace(ysort[0]-(ysort[1]-ysort[0])/2, ysort[-1]+(ysort[-1]-ysort[-2])/2, len(set(ys))+1, endpoint=True)

import json
import lzma

data = {
    "x": list(xs),
    "y": list(ys),
    "weight": list(zs),
    "xbins": list(xbins),
    "ybins": list(ybins),
}

with lzma.open("data.json.xz", "w") as f:
    f.write(json.dumps(data).encode('utf-8'))

fig = plt.gcf()
fig.set_size_inches(3+3/8, 2.5)

plt.xlabel("$\\varepsilon_1$")
plt.ylabel("$\\varepsilon_2$")

h, xedges, yedges, _ = plt.hist2d(
    xs,
    ys,
    weights=zs,
    bins=(xbins, ybins),
    cmap=plt.cm.viridis_r,
    vmin=0,
    vmax=1,
#    norm=matplotlib.colors.LogNorm(),
#    density=True
)
plt.colorbar()
plt.tight_layout()

fig.savefig('map.pdf', dpi=300)
fig.savefig("map.png")
#plt.show()
