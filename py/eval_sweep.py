import os
import gzip
from subprocess import call

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import parameters
from parameters import parameters as ps

os.makedirs("out", exist_ok=True)

def bootstrap(xRaw, f=np.mean, n_resample=100):
    """Bootstrap resampling, returns mean and stderror"""
    if not len(xRaw):
        return float("NaN"), float("NaN")
    bootstrapSample = [f(np.random.choice(xRaw, len(xRaw), replace=True)) for i in range(n_resample)]
    return np.mean(bootstrapSample), np.std(bootstrapSample)
    
def binder(raw):
    return 1 - np.mean([i**4 for i in raw]) / (np.mean([i**2 for i in raw])**2)
    

ns = set()
for p in ps:
    n = p["num_agents"]
    ns.add(n)
    with open("out/d{}.dat".format(n), "w") as f:
        f.write("# eps <S> err\n")
    with open("out/p{}.dat".format(n), "w") as f:
        f.write("# eps P(S) err\n")
    with open("out/s{}.dat".format(n), "w") as f:
        f.write("# eps mean_speed err\n")
    with open("out/v{}.dat".format(n), "w") as f:
        f.write("# eps variance err\n")
    with open("out/b{}.dat".format(n), "w") as f:
        f.write("# eps binder err\n")

m = {}
ma = {}
for p in ps:
    n = p["num_agents"]
    eu = p["tolerance_upper"]

    name = parameters.outname.format(**p).replace(".", "")

    try:
        with gzip.open(name + ".cluster.dat.gz") as f:
            sizes = []
            speeds = []
            for line in f:
                if line.startswith(b"#"):
                    if b"sweep" in line:
                        speeds.append(int(line.split()[-1]))
                    continue
                a = list(map(float, line.split()))

                try:
                    len(a)
                except:
                    val = a
                else:
                    val = max(a)

                sizes.append(val/n)
        with open("out/d{}.dat".format(n), "a") as f:
            f.write("{} {} {}\n".format(eu, np.mean(sizes), np.std(sizes)/len(sizes)**0.5))
        with open("out/p{}.dat".format(n), "a") as f:
            prop = [1 if size == 1 else 0 for size in sizes]
            f.write("{} {} {}\n".format(eu, np.mean(prop), np.std(prop)/len(prop)**0.5))
        with open("out/s{}.dat".format(n), "a") as f:
            f.write("{} {} {}\n".format(eu, np.mean(speeds), np.std(speeds)/len(sizes)**0.5))
        with open("out/b{}.dat".format(n), "a") as f:
            f.write("{} {} {}\n".format(eu, binder(sizes), 0))
        v, verr = bootstrap(sizes, np.var)
        if m.get(n, 0) < v:
            m[n] = v
            ma[n] = eu
        with open("out/v{}.dat".format(n), "a") as f:
            f.write("{} {} {}\n".format(eu, v, verr))
            
    except OSError:
        print("# {} missing".format(name))

with open("plot.gp", "w") as f:        
    f.write("set term pngcairo size 1000,350\n")
    f.write("set output 'plot.png'\n")
    f.write("set multiplot layout 1,2\n")
    f.write("set key bottom\n")
    f.write("set xl 'eps'\n")
    f.write("set yl '<S>'\n")
    f.write("p\\\n")
    for m, n in enumerate(sorted(ns), start=1):
        if os.stat(f'out/d{n}.dat').st_size > 20:
            f.write(f"'out/d{n}.dat' w ye lc {m} t '{n}',\\\n")
            f.write(f"'hk_cmp/d{n}.dat' w l lc {m} lw 2 not 'HK: {n}',\\\n")
            f.write(f"'dw_cmp/d{n}.dat' w l lc {m} lw 2 not 'DW: {n}',\\\n")
    f.write("\n")        
            
    f.write("set yl 'var(S)'\n")    
    f.write("unset key\n")
    f.write("p\\\n")
    for m, n in enumerate(sorted(ns), start=1):
        if os.stat(f'out/d{n}.dat').st_size > 20:
            f.write(f"'out/v{n}.dat' w ye lc {m} t '{n}',\\\n")
            f.write(f"'hk_cmp/v{n}.dat' w l lc {m} lw 2 t 'HK: {n}',\\\n")
            f.write(f"'dw_cmp/v{n}.dat' w l lc {m} lw 2 t 'DW: {n}',\\\n")
            
            
    
#    f.write("set key top left\n")
#    f.write("set yl 'P(S)'\np\\\n")
#    for n in sorted(ns):
#        f.write(f"'out/p{n}.dat' w ye t '{n}',\\\n")
    f.write("\nunset multiplot\n")
    

with open("speed.gp", "w") as f:        
    f.write("set term pngcairo size 1000,350\n")
    f.write("set output 'speed.png'\n")
    f.write("set key bottom\n")
    f.write("set xl 'eps'\n")
    f.write("set yl 'speed'\n")
    f.write("p\\\n")
    for n in sorted(ns):
        f.write(f"'out/s{n}.dat' w ye t '{n}',\\\n")

    
with open("var.gp", "w") as f:        
    f.write("set term pngcairo size 600,350\n")
    f.write("set output 'var.png'\n")
    f.write("set key bottom\n")
    f.write("set xl 'eps'\n")
    f.write("set yl 'chi = n var(S)'\n")
    f.write("p\\\n")
    for n in sorted(ns):
        f.write(f"'out/v{n}.dat' u 1:($2*{n}) w errorlines t '{n}',\\\n")
        
with open("binder.gp", "w") as f:        
    f.write("set term pngcairo size 600,350\n")
    f.write("set output 'binder.png'\n")
    f.write("set key bottom\n")
    f.write("set xl 'eps'\n")
    f.write("set yl 'binder'\n")
    f.write("p\\\n")
    for n in sorted(ns):
        f.write(f"'out/b{n}.dat' u 1:2 w p t '{n}',\\\n")
        
for n, e in ma.items():
    print(n, e)

call(["gnuplot", "plot.gp"])
call(["gnuplot", "speed.gp"])
call(["gnuplot", "var.gp"])
call(["gnuplot", "binder.gp"])
