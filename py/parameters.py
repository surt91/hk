
outname = "data/out_n{num_agents}_d{dimension}_e{tolerance_lower}-{tolerance_upper}_s{seed}"
parameters = {
    # model 1: standard HK,
    #       2: lorenz high dimensional HK
    "model": 2,
    "num_agents": 10000,
    "tolerance_lower": 0.2,
    "tolerance_upper": 0.3,
    "dimension": 3,
    # number of sweeps, 0 means until equilibration
    "iterations": 0,
    "samples": 1000,
}
