import os
import sys
from subprocess import call

import parameters
from parameters import parameters as ps


cmds = []
os.makedirs("data", exist_ok=True)

for p in ps:
    with open("jobs.lst", "w") as f:
        for i in range(p["samples"]):
            name = parameters.outname.format(seed=i, **p).replace(".", "")
            cmd = "target/release/hk -n {num_agents} -u {tolerance_upper} -l {tolerance_lower} -m {model} -d {dimension} -i {iterations} -s {seed} -o {filename}".format(seed=i, filename=name, **p)
            indicator = name + ".dat.gz"
            if not os.path.exists(indicator) or "-f" in sys.argv:
                cmds.append(cmd.split())
                f.write("{}\n".format(cmd))

if "run" in sys.argv:
    import multiprocessing
    # we can not be more than 2x parallel, otherwise the computer will be too loud
    with multiprocessing.Pool(2) as p:
        p.map(call, cmds)
