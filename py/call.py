import os
import sys
from subprocess import call

import parameters
from parameters import parameters as ps


cmds = []
os.makedirs("data", exist_ok=True)

with open("jobs.lst", "w") as f:
    ctr = 0
    for p in ps:
        ctr += 1
        name = parameters.outname.format(**p).replace(".", "")
        cmd = "./hk -n {num_agents} -u {tolerance_upper} -l {tolerance_lower} -m {model} -i {iterations} -s {ctr} --eta {eta} --samples {samples} --resource-distribution {resource_distribution} --min-resources {min_resources} --topology {topology} --topology-parameter {topology_parameter} --topology-parameter2 {topology_parameter2} --max-resources {max_resources} -o {filename}".format(filename=name, ctr=ctr, **p)
        indicator = name + ".cluster.dat.gz"
        if not os.path.exists(indicator) or "-f" in sys.argv:
            cmds.append(cmd.split())
            f.write("{}\n".format(cmd))


def c(cmd):
    print(" ".join(cmd))
    call(cmd)


if "run" in sys.argv:
    import multiprocessing
    # we can not be more than 2x parallel, otherwise the computer will be too loud
    with multiprocessing.Pool() as p:
        p.map(c, cmds, 1)
