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
    with gzip.open(name + ".scc.dat.gz") as f:
        for line in f:
            # read two lines, first (beginning with #) is data, second is weight
            l1 = f.readline().decode()
            if "sweep" in l1:
                l1 = f.readline().decode()
            l2 = f.readline().decode()
            if not l1:
                break
            raw_data = list(map(float, l1.strip("#").strip().split()))
            weight = list(map(float, l2.strip().split()))

            data, borders = np.histogram(raw_data, bins=100, range=(0,1), weights=weight, density=False)
            data /= sum(weight)
            centers = (borders[1:] + borders[:-1])/2

            # interpolate the data linearly
            peak_height = np.max(data)
            fn = interp1d(centers, data - peak_height / 3.)

            changed_sign = [centers[0]]
            last_sign = np.sign(data[0] - peak_height / 3.)
            for x, y in zip(centers, data - peak_height / 3.):
                if last_sign != np.sign(y) and np.sign(y) != 0:
                    changed_sign.append(x)
                    last_sign = np.sign(y)

            all_borders = []
            for (a, b) in zip(changed_sign[0:], changed_sign[1:]):
                r = brentq(fn, a, b)
                all_borders.append(r)

            all_borders.sort()
            if len(all_borders) % 2 != 0:
                print(changed_sign)
                print(all_borders)

            assert(len(all_borders) % 2 == 0)
            cluster_sizes = [sum([weight if r1 < pos < r2 else 0 for (pos, weight) in zip(raw_data, weight)]) for (r1, r2) in cluster_borders]
            val = max(cluster_sizes)

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
