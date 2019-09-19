
outname = "data/out_n{num_agents}_d{dimension}_e{tolerance_lower}-{tolerance_upper}_s{seed}"
# parameters = {
#     # model 1: standard HK,
#     #       2: lorenz high dimensional HK
#     "model": 2,
#     "num_agents": 10000,
#     "tolerance_lower": 0.2,
#     "tolerance_upper": 0.3,
#     "dimension": 3,
#     # number of sweeps, 0 means until equilibration
#     "iterations": 0,
#     "samples": 1000,
# }
#
# parameters = {
#     # model 1: standard HK,
#     #       2: lorenz high dimensional HK
#     "model": 2,
#     "num_agents": 1000,
#     "tolerance_lower": 0.1,
#     "tolerance_upper": 0.2,
#     "dimension": 3,
#     # number of sweeps, 0 means until equilibration
#     "iterations": 0,
#     "samples": 1000,
# }

# parameters = {
#     # model 1: standard HK,
#     #       2: lorenz high dimensional HK
#     "model": 2,
#     "num_agents": 1000,
#     "tolerance_lower": 0.2,
#     "tolerance_upper": 0.3,
#     "dimension": 3,
#     # number of sweeps, 0 means until equilibration
#     "iterations": 0,
#     "samples": 100,
# }
#
# parameters = {
#     # model 1: standard HK,
#     #       2: lorenz high dimensional HK
#     "model": 2,
#     "num_agents": 200,
#     "tolerance_lower": 0.23,
#     "tolerance_upper": 0.23,
#     "dimension": 3,
#     # number of sweeps, 0 means until equilibration
#     "iterations": 40,
#     "samples": 4000,
# }
# parameters = {
#     # model 1: standard HK,
#     #       2: lorenz high dimensional HK
#     "model": 2,
#     "num_agents": 200,
#     "tolerance_lower": 0.23,
#     "tolerance_upper": 0.23,
#     "dimension": 3,
#     # number of sweeps, 0 means until equilibration
#     "iterations": 0,
#     "samples": 4000,
# }

# parameters = {
#     # model 1: standard HK,
#     #       2: lorenz high dimensional HK
#     "model": 3,
#     "num_agents": 100,
#     "tolerance_lower": 0.3,
#     "tolerance_upper": 0.3,
#     "dimension": 1,
#     # number of sweeps, 0 means until equilibration
#     "iterations": 40,
#     "samples": 100,
# }

# parameters = {
#     # model 1: standard HK,
#     #       2: lorenz high dimensional HK
#     "model": 3,
#     "num_agents": 1000,
#     "tolerance_lower": 0.3,
#     "tolerance_upper": 0.3,
#     "dimension": 1,
#     # number of sweeps, 0 means until equilibration
#     "iterations": 40,
#     "samples": 100,
# }

# parameters = {
#     # model 1: standard HK,
#     #       2: lorenz high dimensional HK
#     "model": 3,
#     "num_agents": 10000,
#     "tolerance_lower": 0.3,
#     "tolerance_upper": 0.3,
#     "dimension": 1,
#     # number of sweeps, 0 means until equilibration
#     "iterations": 120,
#     "samples": 100,
# }

parameters = [
    {
        # model 1: standard HK,
        #       2: lorenz high dimensional HK
        "model": 2,
        "num_agents": 1000,
        "tolerance_lower": e,
        "tolerance_upper": e,
        "dimension": 3,
        # number of sweeps, 0 means until equilibration
        "iterations": 40,
        "samples": 1000,
    } for e in [0.19, 0.20, 0.21, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3]
] + [
    {
        # model 1: standard HK,
        #       2: lorenz high dimensional HK
        "model": 2,
        "num_agents": 1000,
        "tolerance_lower": 0.23-d,
        "tolerance_upper": 0.23+d,
        "dimension": 3,
        # number of sweeps, 0 means until equilibration
        "iterations": 40,
        "samples": 1000,
    } for d in [0.1, 0.2, 0.3, 0.4]
] + [
    {
        # model 1: standard HK,
        #       2: lorenz high dimensional HK
        "model": 2,
        "num_agents": n,
        "tolerance_lower": 0.23,
        "tolerance_upper": 0.23,
        "dimension": 3,
        # number of sweeps, 0 means until equilibration
        "iterations": 40,
        "samples": 1000,
    } for n in [100, 1000, 10000]
]
