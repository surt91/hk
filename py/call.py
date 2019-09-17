import os

import parameters
from parameters import parameters as p


cmds = []
os.makedirs("data", exist_ok=True)
with open("jobs.lst", "w") as f:
    for i in range(p["num_samples"]):
        name = parameters.outname.format(*p)
        cmd = "target/release/hk -n {num_agents} -u {tolerance_upper} -l {tolerance_lower} -m {model} -d {dimension} -i {iterations} -s {seed} -o {filename}".format(seed=i, filename=name, **p)
        cmds.append(cmd)
        f.write("{}\n")

# import multiprocessing
# with multiprocessing.Pool(1) as p:
#     p.map(os.system, cmds)
