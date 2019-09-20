import os
import sys
from subprocess import call

import parameters
from parameters import parameters as ps


cmds = []
os.makedirs("data", exist_ok=True)

with open("jobs.lst", "w") as f:
    for p in ps:
        name = parameters.outname.format(**p).replace(".", "")
        cmd = "target/release/hk -n {num_agents} -u {tolerance_upper} -l {tolerance_lower} -m {model} -d {dimension} -i {iterations} -s {seed} --samples {samples} -o {filename}".format(filename=name, **p)
        indicator = name + ".dat.gz"
        if not os.path.exists(indicator) or "-f" in sys.argv:
            cmds.append(cmd.split())
            f.write("{}\n".format(cmd))


def c(cmd):
    print(" ".join(cmd))
    call(cmd)


if "run" in sys.argv:
    import multiprocessing
    # we can not be more than 2x parallel, otherwise the computer will be too loud
    with multiprocessing.Pool(2) as p:
        p.map(c, cmds)
