outname = "data/out_n{num_agents}_d{dimension}_e{tolerance_lower}-{tolerance_upper}_s{seed}"

parameters = [
    {
        # model 1: standard HK,
        #       2: lorenz high dimensional HK
        "model": 2,
        "num_agents": 100,
        "tolerance_lower": el,
        "tolerance_upper": eu,
        "dimension": 3,
        # number of sweeps, 0 means until equilibration
        "iterations": 0,
        "samples": 100,
	"seed": 42,
    } for el in [i/20 for i in range(1, 20)] for eu in [i/20 for i in range(1, 20)] if eu >= el
]

parameters = [
    {
        # model 1: standard HK,
        #       2: lorenz high dimensional HK
        "model": 2,
        "num_agents": 300,
        "tolerance_lower": el,
        "tolerance_upper": eu,
        "dimension": 3,
        # number of sweeps, 0 means until equilibration
        "iterations": 0,
        "samples": 100,
	"seed": 42,
    } for el in [i/20 for i in range(1, 8)] for eu in [i/20 for i in range(1, 20)] if eu >= el
]

parameters = [
    {
        # model 1: standard HK,
        #       2: lorenz high dimensional HK
        "model": 2,
        "num_agents": 1024,
        "tolerance_lower": el,
        "tolerance_upper": eu,
        "dimension": 3,
        # number of sweeps, 0 means until equilibration
        "iterations": 0,
        "samples": 100,
	"seed": 42,
    } for el in [i/20 for i in range(1, 8)] for eu in [i/20 for i in range(1, 20)] if eu >= el
]
