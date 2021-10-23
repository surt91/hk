# Bounded Confidence Simulation

This is a rust program to simulate bounded confidence
opinion dynamics models, in particular:

* Hegselmann-Krause on a complete network with the tree-based alogrithm.
* Hegselmann-Krause on networks.
* Hegselmann-Krause with costs.
* Deffuant model on networks.
* Deffuant model generalized to hypergraphs.

Also some unfinished experiments.

It was used in the following publications (all open access):

1. [*When open mindedness hinders consensus*, Hendrik Schawe, Laura Hernández, Scientific Reports **10**, 8273 (2020) ](https://dx.doi.org/10.1038/s41598-020-64691-0)
2. [*Collective effects of the cost of opinion change*, Hendrik Schawe, Laura Hernández, Scientific Reports **10**, 13825 (2020)](https://dx.doi.org/10.1038/s41598-020-70809-1)
3. [*When network bridges foster consensus. Bounded confidence models in networked societies*, Hendrik Schawe, Sylvain Fontaine, Laura Hernández, Physical Review Research **3**, 023208 (2021)](https://dx.doi.org/10.1103/PhysRevResearch.3.023208)

## Setup

Install rust, compile and run it like:

```
# curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo build --release
target/release/hk -h
```

## Usage

```
Simulate a (modified) Hegselmann Krause model

USAGE:
    hk [FLAGS] [OPTIONS] --num-agents <num-agents> [SUBCOMMAND]

FLAGS:
        --betweenness    switch whether to measure and save an approximation of the maximum betweenness centrality of
                         the active graph over the whole simulation
    -h, --help           Prints help information
        --png            switch whether to save an image of the topology in the initial and final state will be
                         `outname` with a .png extention
        --scc            also calculate SCC cluster (needs more memory to hold a graph structure)
        --sync           synchronous update instead of random sequential
    -V, --version        Prints version information

OPTIONS:
        --eta <eta>                                          weight of cost [default: 0.01]
    -i, --iterations <iterations>                            number of sweeps to run the simulation [default: 100]
        --max-resources <max-resources>                      maximal resources for HKCost [default: 1]
    -u, --max-tolerance <max-tolerance>
            maximum tolerance of agents (uniformly distributed) [default: 1.0]

        --min-resources <min-resources>                      minimal resources for HKCost [default: 0]
    -l, --min-tolerance <min-tolerance>
            minimum tolerance of agents (uniformly distributed) [default: 0.0]

    -m, --model <model>
            which model to simulate:
             1 -> Hegselmann Krause
             3 -> HK with active cost
             5 -> HK with passive cost
             9 -> Deffuant Weisbuch
             10 -> Only topology information
             11 -> Hyper-Deffuant with rewiring
             12 -> HK with periodic opinion
             [default: 1]  [possible values: 1, 3, 5, 9, 10, 11, 12]
    -n, --num-agents <num-agents>                            number of interacting agents
    -o, --outname <outname>                                  name of the output data file [default: out]
        --resource-distribution <resource-distribution>
            distribution of the resources c_i:
             1 => uniform between min and max
             2 => pareto with exponent -2.5
             3 => proportional to the tolerances but with same average total resources
             4 => antiproportional to the tolerances but with same average total resources
             5 => half-Gaussian with std of `--max-resources`
             [default: 1]  [possible values: 1, 2, 3, 4, 5]
        --rewiring-modus <rewiring-modus>
            rewiring modus (only for the rewiring Deffuant on hypergraphs):
             1 => join a random edge when frustrated
             2 => join a random edge of the best neighbor
             [default: 1]  [possible values: 1, 2]
        --samples <samples>                                  number of times to repeat the simulation [default: 1]
    -s, --seed <seed>                                        seed to use for the simulation [default: 1]
    -T, --temperature <temperature>                          temperature (only for fixed temperature 8) [default: 1.0]
        --tmp <tmp>                                          directory to store temporary files [default: ./tmp]
        --tolerance-distribution <tolerance-distribution>
            distribution of the tolerances epsilon_i:
             1 => uniform between min and max
             2 => bimodal: half min, half max
             3 => 15% of agents at x(0) = 0.25+-0.05, with confidence eps = 0.075+-0.05
             4 => gaussian: min -> mean, max -> variance
             5 => pareto: min -> lower bound (scale), max -> exponent (= shape+1)
             6 => power law: min -> lower bound, max -> upper bound, exponent: 2.5
             [default: 1]  [possible values: 1, 2, 3, 4, 5, 6]
        --topology <topology>
            topology:
             1 => fully connected
             2 => Erdoes Renyi
             3 => Barabasi Albert
             4 => biased Configuration Model
             5 => correct Configuration Model
             6 => periodic square lattice (num_agents needs to be a perfect square)
             7 => Watts-Strogatz small world network on a ring
             8 => Watts-Strogatz small world network on a square lattice
             9 => BA+Triangles
             10 => Hyper-Erdoes-Renyi
             11 => Hyper-Erdoes-Renyi, Simplical Complex
             12 => Hyper-Barabasi-Albert
             13 => Hyper-Erdoes-Renyi, 2 hypergraph orders
             14 => Hyper-Erdoes-Renyi, Gaussian distributed orders
             15 => Hypergraph with nearest neighbor square lattice structure, c = 12, k = 3
             16 => Hypergraph with third nearest neighbor square lattice structure, c = 15, k = 5
             17 => Watts-Strogatz small world network on a Hypergraph with third nearest neighbor square lattice
            structure, c = 12, k = 3
             [default: 1]  [possible values: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        --topology-parameter <topology-parameter>
            dependent on topology:
             fully connected: unused
             Erdoes Renyi: connectivity
             Barabasi Albert: mean degree
             Configuration Model: exponent (must be negative)
             square lattice: n-th nearest neighbors
             Watts Strogatz: n-th nearest neighbors
             BA+Triangles: m
             HyperBA: m
             Hyper-ER 2: c1
             Hyper-ER Gaussian: c (scale factor)
             Hyper-WS: rewiring probability
             [default: 1]
        --topology-parameter2 <topology-parameter2>
            dependent on topology:
             Configuration Model: minimum degree
             square lattice: unused
             Watts Strogatz: rewiring probability
             BA+Triangles: m_t
             HyperBA: k
             Hyper-ER 2: c2
             Hyper-ER Gaussian: mean mu
             [default: 1]
        --topology-parameter3 <topology-parameter3>          Hyper-ER Gaussian: standard deviation sigma
                                                              [default: 1]

SUBCOMMANDS:
    help           Prints this message or the help of the given subcommand(s)
    metropolis     use biased Metropolis sampling
    wang-landau    use biased Wang Landau sampling
```
