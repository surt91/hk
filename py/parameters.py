outname = "data/out_n{num_agents}_e{tolerance_lower}_s{seed}"


parameters = [
    {
        "model": 9,
        "num_agents": n,
        "resource_distribution": 1,
        "tolerance_lower": eps,
        "tolerance_upper": eps,
        "topology": 12,
        "topology_parameter": 3,
        "topology_parameter2": 3,
        "min_resources": 0.,
        "max_resources": 0.,
        "dimension": 1,
        # number of sweeps, 0 means until equilibration
        "iterations": 0,
        "samples": 1000,
        "seed": 13,
        "eta": 0
    } 
    for n in (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
    for eps in sorted(set([i/500. for i in range(1, int(500*0.6)+1)]))
]

