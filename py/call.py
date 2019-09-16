import os

import multiprocessing

n = 1000

cmds = []
os.makedirs("data", exist_ok=True)
for i in range(1000):
    print(i)
    cmds.append("target/release/hk -n {n} -u 0.3 -l 0.2 -m 2 -d 3 -i 0 -s {seed} -o data/out_n{n}_02-03_s{seed}".format(seed=i, n=n))

with multiprocessing.Pool() as p:
    p.map(os.system, cmds)
