use std::fs::File;
use std::io::prelude::*;
use std::path::PathBuf;

use structopt::StructOpt;

use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use itertools::Itertools;

use hk::{ABM, ABMBuilder, ACC_EPS};
use hk::HegselmannKrause;
use hk::{CostModel, ResourceModel, PopulationModel, TopologyModel, DegreeDist};
use hk::models::graph;

use largedev::{Metropolis, WangLandau};

use git_version::git_version;
const GIT_VERSION: &str = git_version!();

mod io;
use io::Output;

/// Simulate a (modified) Hegselmann Krause model
#[derive(StructOpt)]
#[structopt(version = GIT_VERSION)]
struct Opt {
    #[structopt(short, long)]
    /// number of interacting agents
    num_agents: u32,

    #[structopt(long, default_value = "1", possible_values = &["1", "2", "3", "4", "5", "6"])]
    /// distribution of the tolerances epsilon_i:{n}
    /// 1 => uniform between min and max{n}
    /// 2 => bimodal: half min, half max{n}
    /// 3 => 15% of agents at x(0) = 0.25+-0.05, with confidence eps = 0.075+-0.05{n}
    /// 4 => gaussian: min -> mean, max -> variance{n}
    /// 5 => pareto: min -> lower bound (scale), max -> exponent (= shape+1){n}
    /// 6 => power law: min -> lower bound, max -> upper bound, exponent: 2.5{n}
    tolerance_distribution: u32,

    #[structopt(short = "l", long, default_value = "0.0")]
    /// minimum tolerance of agents (uniformly distributed)
    min_tolerance: f64,

    #[structopt(short = "u", long, default_value = "1.0")]
    /// maximum tolerance of agents (uniformly distributed)
    max_tolerance: f64,

    #[structopt(long, default_value = "1", possible_values = &["1", "2", "3", "4", "5"])]
    /// distribution of the resources c_i:{n}
    /// 1 => uniform between min and max{n}
    /// 2 => pareto with exponent -2.5{n}
    /// 3 => proportional to the tolerances but with same average total resources{n}
    /// 4 => antiproportional to the tolerances but with same average total resources{n}
    /// 5 => half-Gaussian with std of `--max-resources`{n}
    resource_distribution: u32,

    #[structopt(long, default_value = "0")]
    /// minimal resources for HKCost
    min_resources: f64,

    #[structopt(long, default_value = "1")]
    /// maximal resources for HKCost
    max_resources: f64,

    #[structopt(long, default_value = "1", possible_values = &["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17"])]
    /// topology:{n}
    /// 1 => fully connected{n}
    /// 2 => Erdoes Renyi{n}
    /// 3 => Barabasi Albert{n}
    /// 4 => biased Configuration Model{n}
    /// 5 => correct Configuration Model{n}
    /// 6 => periodic square lattice (num_agents needs to be a perfect square){n}
    /// 7 => Watts-Strogatz small world network on a ring{n}
    /// 8 => Watts-Strogatz small world network on a square lattice{n}
    /// 9 => BA+Triangles{n}
    /// 10 => Hyper-Erdoes-Renyi{n}
    /// 11 => Hyper-Erdoes-Renyi, Simplical Complex{n}
    /// 12 => Hyper-Barabasi-Albert{n}
    /// 13 => Hyper-Erdoes-Renyi, 2 hypergraph orders{n}
    /// 14 => Hyper-Erdoes-Renyi, Gaussian distributed orders{n}
    /// 15 => Hypergraph with nearest neighbor square lattice structure, c = 12, k = 3{n}
    /// 16 => Hypergraph with third nearest neighbor square lattice structure, c = 15, k = 5{n}
    /// 17 => Watts-Strogatz small world network on a Hypergraph with third nearest neighbor
    ///       square lattice structure, c = 12, k = 3{n}
    topology: u32,

    #[structopt(long, default_value = "1", allow_hyphen_values = true)]
    /// dependent on topology:{n}
    /// fully connected: unused{n}
    /// Erdoes Renyi: connectivity{n}
    /// Barabasi Albert: mean degree{n}
    /// Configuration Model: exponent (must be negative){n}
    /// square lattice: n-th nearest neighbors{n}
    /// Watts Strogatz: n-th nearest neighbors{n}
    /// BA+Triangles: m{n}
    /// HyperBA: m{n}
    /// Hyper-ER 2: c1{n}
    /// Hyper-ER Gaussian: c (scale factor){n}
    /// Hyper-WS: rewiring probability{n}
    topology_parameter: f32,

    #[structopt(long, default_value = "1")]
    /// dependent on topology:{n}
    /// Configuration Model: minimum degree{n}
    /// square lattice: unused{n}
    /// Watts Strogatz: rewiring probability{n}
    /// BA+Triangles: m_t{n}
    /// HyperBA: k{n}
    /// Hyper-ER 2: c2{n}
    /// Hyper-ER Gaussian: mean mu{n}
    topology_parameter2: f32,

    #[structopt(long, default_value = "1")]
    /// Hyper-ER Gaussian: standard deviation sigma{n}
    topology_parameter3: f32,

    #[structopt(long)]
    /// switch whether to save an image of the topology in the initial and final state
    /// will be `outname` with a .png extention
    png: bool,

    #[structopt(long)]
    /// switch whether to measure and save an approximation of the maximum
    /// betweenness centrality of the active graph over the whole simulation
    betweenness: bool,

    #[structopt(long, default_value = "0.01")]
    /// weight of cost
    eta: f64,

    #[structopt(short = "T", long, default_value = "1.0", allow_hyphen_values = true)]
    /// temperature (only for fixed temperature 8)
    temperature: f64,

    #[structopt(short, long, default_value = "1")]
    /// seed to use for the simulation
    seed: u64,

    #[structopt(short, long, default_value = "100")]
    /// number of sweeps to run the simulation
    iterations: u64,

    #[structopt(long)]
    /// synchronous update instead of random sequential
    sync: bool,

    #[structopt(long)]
    /// also calculate SCC cluster (needs more memory to hold a graph structure)
    scc: bool,

    #[structopt(long, default_value = "1")]
    /// number of times to repeat the simulation
    samples: u32,

    #[structopt(short, long, default_value = "1", possible_values = &["1", "3", "5", "9", "10", "11"])]
    /// which model to simulate:{n}
    /// 1 -> Hegselmann Krause{n}
    /// 3 -> HK with active cost{n}
    /// 5 -> HK with passive cost{n}
    /// 9 -> Deffuant Weisbuch{n}
    /// 10 -> Only topology information{n}
    /// 11 -> Hyper-Deffuant with rewiring{n}
    model: u32,

    #[structopt(long, default_value = "./tmp", parse(from_os_str))]
    /// directory to store temporary files
    tmp: PathBuf,

    #[structopt(short, long, default_value = "out", parse(from_os_str))]
    /// name of the output data file
    outname: PathBuf,

    #[structopt(subcommand)]
    /// test
    cmd: Option<LargeDev>
}

#[derive(StructOpt)]
struct WL {
    low: f64,
    high: f64,
    #[structopt(long, default_value = "100")]
    num_bins: usize,
    #[structopt(long, default_value = "1e-5")]
    lnf: f64,
}

#[derive(StructOpt)]
enum LargeDev {
    /// use biased Metropolis sampling
    Metropolis,
    /// use biased Wang Landau sampling
    WangLandau(WL),
}


pub fn cluster_sizes_from_graph(hk: &HegselmannKrause) -> Vec<usize> {
    let g = graph::from_hk(&hk);
    graph::clustersizes(&g)
}

fn write_cluster_sizes(clusters: &[usize], file: &mut File) -> std::io::Result<()> {
    let s = entropy(&clusters);
    writeln!(file, "# entropy: {}", s)?;

    let string_list = clusters.iter()
        .map(|c| c.to_string())
        .join(" ");
    writeln!(file, "{}", string_list)?;
    Ok(())
}

fn entropy(clustersizes: &[usize]) -> f32 {
    let f = 1. / clustersizes.iter().sum::<usize>() as f32;
    clustersizes.iter().map(|c| {
        let p = *c as f32 * f;
        - p * p.ln()
    }).sum()
}

fn main() -> std::io::Result<()> {
    let args = Opt::from_args();

    // let raw_args: Vec<String> = std::env::args().collect();
    let info_args = format!("# {}", std::env::args().join(" "));
    let info_version = format!("# {}", GIT_VERSION);

    let pop_model = match args.tolerance_distribution {
        1 => PopulationModel::Uniform(args.min_tolerance as f32, args.max_tolerance as f32),
        2 => PopulationModel::Bimodal(args.min_tolerance as f32, args.max_tolerance as f32),
        3 => PopulationModel::Bridgehead(0.25, 0.05, 0.15, 0.075, 0.05, args.min_tolerance as f32, args.max_tolerance as f32),
        4 => PopulationModel::Gaussian(args.min_tolerance as f32, args.max_tolerance as f32),
        5 => PopulationModel::PowerLaw(args.min_tolerance as f32, args.max_tolerance as f32),
        6 => PopulationModel::PowerLawBound(args.min_tolerance as f32, args.max_tolerance as f32, 2.5),
        _ => unreachable!(),
    };

    let cost_model = match args.model {
        1 => CostModel::Free,
        3 => CostModel::Rebounce,
        5 => CostModel::Change(args.eta as f32),
        _ => {
            println!("# Warning: you use a not well tested secondary type!");
            CostModel::Free
        },
    };

    let resource_model = match args.resource_distribution {
        1 => ResourceModel::Uniform(args.min_resources as f32, args.max_resources as f32),
        2 => {
            let k = 1.5; // corresponds to an exponent of -2.5
            let x_min = (args.min_resources + args.max_resources) as f32 / 2. * (k - 1.) / k;
            ResourceModel::Pareto(x_min, k + 1.)
        },
        3 => {
            assert_eq!(args.min_resources, 0.);
            assert_eq!(args.max_resources, 1.);
            let offset = args.min_tolerance;
            let prop = 1. / (args.max_tolerance - args.min_tolerance);
            ResourceModel::Proportional(prop as f32, offset as f32)
        },
        4 => {
            assert_eq!(args.min_resources, 0.);
            assert_eq!(args.max_resources, 1.);
            let offset = args.min_tolerance;
            let prop = 1. / (args.max_tolerance - args.min_tolerance);
            ResourceModel::Antiproportional(prop as f32, offset as f32)
        },
        5 => {
            assert_eq!(args.min_resources, 0.);
            assert_eq!(args.max_resources, 1.);
            ResourceModel::HalfGauss(0.626_657_07)
        },
        _ => unreachable!(),
    };

    let topology_model = match args.topology {
        1 => TopologyModel::FullyConnected,
        2 => TopologyModel::ER(args.topology_parameter),
        3 => TopologyModel::BA(args.topology_parameter as f64, (2.*args.topology_parameter).ceil() as usize),
        4 => {
            assert!(args.topology_parameter < 0.);
            let dd = DegreeDist::PowerLaw(args.num_agents as usize, args.topology_parameter2, -args.topology_parameter);
            TopologyModel::CMBiased(dd)
        },
        5 => {
            assert!(args.topology_parameter < 0.);
            let dd = DegreeDist::PowerLaw(args.num_agents as usize, args.topology_parameter2, -args.topology_parameter);
            TopologyModel::CM(dd)
        },
        6 => TopologyModel::SquareLattice(args.topology_parameter as usize),
        7 => TopologyModel::WS(args.topology_parameter as usize, args.topology_parameter2 as f64),
        8 => TopologyModel::WSlat(args.topology_parameter as usize, args.topology_parameter2 as f64),
        9 => TopologyModel::BAT(args.topology_parameter as usize, args.topology_parameter2 as f64),
        10 => TopologyModel::HyperER(args.topology_parameter as f64, args.topology_parameter2 as usize),
        11 => TopologyModel::HyperERSC(args.topology_parameter as f64, args.topology_parameter2 as usize),
        12 => TopologyModel::HyperBA(args.topology_parameter as usize, args.topology_parameter2 as usize),
        13 => TopologyModel::HyperER2(args.topology_parameter as f64, args.topology_parameter2 as f64, 2, 4),
        // 13 => TopologyModel::HyperER2(args.topology_parameter as f64, args.topology_parameter2 as f64, 3, 5),
        14 => TopologyModel::HyperERGaussian(
            args.topology_parameter as f64,
            args.topology_parameter2 as f64,
            args.topology_parameter3 as f64
        ),
        15 => TopologyModel::HyperLattice_3_12,
        16 => TopologyModel::HyperLattice_5_15,
        17 => TopologyModel::HyperWSlat(args.topology_parameter as f64),
        _ => unreachable!(),
    };

    let mut rng = Pcg64::seed_from_u64(args.seed);

    if let Some(LargeDev::Metropolis) = args.cmd {
        let mut hk = ABMBuilder::new(args.num_agents)
            .cost_model(cost_model)
            .resource_model(resource_model)
            .population_model(pop_model)
            .topology_model(topology_model)
            .hk(&mut rng);
        hk.reset(&mut rng);
        hk.relax(&mut rng);

        let mut out = Output::new(&args.outname, "mcmc.dat", &args.tmp)?;

        let mut mc_rng = Pcg64::seed_from_u64(rng.gen());

        let (tries, rejects) = Metropolis::new(hk)
                .temperature(args.temperature)
                .sweep(args.num_agents as usize)
                .iterations(args.iterations as usize)
                .run(&mut mc_rng, out.file())?;

        println!("rejection: {}%", rejects as f32 / tries as f32 * 100.);

        out.finalize()?;

        return Ok(());
    }

    if let Some(LargeDev::WangLandau(wl)) = args.cmd {
        let mut hk = ABMBuilder::new(args.num_agents)
            .cost_model(cost_model)
            .resource_model(resource_model)
            .population_model(pop_model)
            .topology_model(topology_model)
            .hk(&mut rng);
        hk.reset(&mut rng);
        hk.relax(&mut rng);

        let mut out = Output::new(&args.outname, "mcmc.dat", &args.tmp)?;

        let mut mc_rng = Pcg64::seed_from_u64(rng.gen());

        let (tries, rejects) = WangLandau::new(hk, wl.low, wl.high)
                .bins(wl.num_bins)
                .lnf_final(wl.lnf)
                .sweep(args.num_agents as usize)
                .run(&mut mc_rng, out.file())?;

        println!("rejection: {}%", rejects as f32 / tries as f32 * 100.);

        out.finalize()?;

        return Ok(());
    }

    match args.model {
        1 | 3 | 5 => {
            let mut hk = ABMBuilder::new(args.num_agents)
                .cost_model(cost_model)
                .resource_model(resource_model)
                .population_model(pop_model)
                .topology_model(topology_model)
                .hk(&mut rng);

            let mut out_cluster = Output::new(&args.outname, "cluster.dat", &args.tmp)?;
            let mut out_nopoor = Output::new(&args.outname, "nopoor.dat", &args.tmp)?;
            let mut out_scc = Output::new(&args.outname, "scc.dat", &args.tmp)?;
            let mut out_density = Output::new(&args.outname, "density.dat", &args.tmp)?;
            let mut out_entropy = Output::new(&args.outname, "entropy.dat", &args.tmp)?;
            let mut out_topology = Output::new(&args.outname, "topology.dat", &args.tmp)?;
            let mut out_info = Output::new(&args.outname, "info.dat", &args.tmp)?;

            // if we only do one sample, we also save a detailed evolution
            let mut out_detailed = Output::new(&args.outname, "detailed.dat", &args.tmp)?;

            write!(out_info.file(), "{}\n{}\n", info_version, info_args)?;

            for _ in 0..args.samples {
                hk.reset(&mut rng);

                if args.png {
                    hk.write_graph_png(&args.outname.with_extension("init.png"), true)?;
                }

                // if we only do one sample, we also save a detailed evoluti
                if args.samples == 1 {
                    hk.write_state(out_detailed.file())?;
                }

                let mut ctr = 0;
                let mut thr = 1.;
                loop {
                    // draw before the sweep, to get the initial condition
                    // if we only do one sample, we also save a detailed evolution
                    if args.samples == 1 {
                        hk.write_state(out_detailed.file())?;
                    }

                    // test if we are converged
                    ctr += 1;

                    if args.sync {
                        hk.sweep_synchronous();
                    } else {
                        hk.sweep(&mut rng);
                    }

                    if args.betweenness && thr <= ctr as f64 {
                        hk.update_max_betweenness();
                        thr *= 1.5;
                    }

                    if hk.get_acc_change() < ACC_EPS || (args.iterations > 0 && ctr > args.iterations) {
                        writeln!(out_cluster.file(), "# sweeps: {}", ctr)?;
                        hk.fill_density();
                        break;
                    }
                    hk.acc_change_reset();
                }
                hk.write_cluster_sizes(&mut out_cluster.file())?;
                hk.write_cluster_sizes_nopoor(&mut out_nopoor.file())?;
                hk.write_topology_info(&mut out_topology.file())?;

                if args.scc {
                    let clusters = cluster_sizes_from_graph(&hk);
                    write_cluster_sizes(&clusters, &mut out_scc.file())?;
                }
                if args.png {
                    hk.write_graph_png(&args.outname.with_extension("final.png"), true)?;
                    hk.write_graph_png(&args.outname.with_extension("all.png"), false)?;
                }
            }

            hk.write_density(&mut out_density.file())?;
            hk.write_entropy(&mut out_entropy.file())?;

            out_cluster.finalize()?;
            out_nopoor.finalize()?;
            out_scc.finalize()?;
            out_density.finalize()?;
            out_entropy.finalize()?;
            out_topology.finalize()?;
            out_info.finalize()?;
            out_detailed.finalize()?;

            Ok(())
        },
        9 => {
            let mut dw = ABMBuilder::new(args.num_agents)
                .population_model(pop_model)
                .topology_model(topology_model)
                .dw(&mut rng);

            let mut out_cluster = Output::new(&args.outname, "cluster.dat", &args.tmp)?;
            let mut out_density = Output::new(&args.outname, "density.dat", &args.tmp)?;
            let mut out_entropy = Output::new(&args.outname, "entropy.dat", &args.tmp)?;
            let mut out_topology = Output::new(&args.outname, "topology.dat", &args.tmp)?;
            let mut out_info = Output::new(&args.outname, "info.dat", &args.tmp)?;

            // if we only do one sample, we also save a detailed evolution
            let mut out_detailed = Output::new(&args.outname, "detailed.dat", &args.tmp)?;

            write!(out_info.file(), "{}\n{}\n", info_version, info_args)?;

            for _ in 0..args.samples {
                dw.reset(&mut rng);

                if args.png {
                    dw.write_graph_png(&args.outname.with_extension("init.png"), true)?;
                }

                // if we only do one sample, we also save a detailed evoluti
                if args.samples == 1 {
                    dw.write_state(out_detailed.file())?;
                }

                let mut ctr = 0;
                loop {
                    // draw before the sweep, to get the initial condition
                    // if we only do one sample, we also save a detailed evolution
                    if args.samples == 1 {
                        dw.write_state(out_detailed.file())?;
                        if ctr % 1 == 0 {
                            dw.write_state_png(&args.outname.with_extension(format!("{:04}.png", ctr)))?;
                        }
                    }

                    // test if we are converged
                    ctr += 1;

                    if args.sync {
                        unimplemented!();
                    } else {
                        dw.sweep(&mut rng);
                    }

                    if dw.get_acc_change() < ACC_EPS || (args.iterations > 0 && ctr > args.iterations) {
                        writeln!(out_cluster.file(), "# sweeps: {}", ctr)?;
                        dw.fill_density();
                        break;
                    }
                    dw.acc_change_reset();
                }
                dw.write_cluster_sizes(&mut out_cluster.file())?;
                dw.write_topology_info(&mut out_topology.file())?;

                if args.png {
                    dw.write_graph_png(&args.outname.with_extension("final.png"), true)?;
                    dw.write_graph_png(&args.outname.with_extension("all.png"), false)?;
                }
            }

            dw.write_density(&mut out_density.file())?;
            dw.write_entropy(&mut out_entropy.file())?;

            out_cluster.finalize()?;
            out_density.finalize()?;
            out_entropy.finalize()?;
            out_topology.finalize()?;
            out_info.finalize()?;
            out_detailed.finalize()?;

            Ok(())
        },
        11 => {
            let mut rew = ABMBuilder::new(args.num_agents)
                .population_model(pop_model)
                .topology_model(topology_model)
                .rewiring(&mut rng);

            let mut out_cluster = Output::new(&args.outname, "cluster.dat", &args.tmp)?;
            let mut out_density = Output::new(&args.outname, "density.dat", &args.tmp)?;
            let mut out_entropy = Output::new(&args.outname, "entropy.dat", &args.tmp)?;
            let mut out_topology = Output::new(&args.outname, "topology.dat", &args.tmp)?;
            let mut out_info = Output::new(&args.outname, "info.dat", &args.tmp)?;

            // if we only do one sample, we also save a detailed evolution
            let mut out_detailed = Output::new(&args.outname, "detailed.dat", &args.tmp)?;

            write!(out_info.file(), "{}\n{}\n", info_version, info_args)?;

            for _ in 0..args.samples {
                rew.reset(&mut rng);

                if args.png {
                    rew.write_graph_png(&args.outname.with_extension("init.png"), true)?;
                }

                // if we only do one sample, we also save a detailed evoluti
                if args.samples == 1 {
                    rew.write_state(out_detailed.file())?;
                }

                let mut ctr = 0;
                loop {
                    // draw before the sweep, to get the initial condition
                    // if we only do one sample, we also save a detailed evolution
                    if args.samples == 1 {
                        rew.write_state(out_detailed.file())?;
                    }

                    // test if we are converged
                    ctr += 1;

                    if args.sync {
                        unimplemented!();
                    } else {
                        rew.sweep(&mut rng);
                    }

                    if rew.get_acc_change() < ACC_EPS || (args.iterations > 0 && ctr > args.iterations) {
                        writeln!(out_cluster.file(), "# sweeps: {}", ctr)?;
                        rew.fill_density();
                        break;
                    }
                    rew.acc_change_reset();
                }
                rew.write_cluster_sizes(&mut out_cluster.file())?;
                rew.write_topology_info(&mut out_topology.file())?;

                if args.png {
                    rew.write_graph_png(&args.outname.with_extension("final.png"), true)?;
                    rew.write_graph_png(&args.outname.with_extension("all.png"), false)?;
                }
            }

            rew.write_density(&mut out_density.file())?;
            rew.write_entropy(&mut out_entropy.file())?;

            out_cluster.finalize()?;
            out_density.finalize()?;
            out_entropy.finalize()?;
            out_topology.finalize()?;
            out_info.finalize()?;
            out_detailed.finalize()?;

            Ok(())
        },
        10 => {
            use counter::Counter;
            use hk::models::hypergraph::build_hyper_uniform_ba;
            if args.topology != 10 {
                println!("only implemented for HyperBA yet");
                unimplemented!()
            }

            let c = (0..args.iterations).flat_map(|_| {
                    let h = build_hyper_uniform_ba(
                        args.num_agents as usize,
                        args.topology_parameter as usize,
                        args.topology_parameter2 as usize,
                        &mut rng
                    );
                    h.degrees()
                })
                .collect::<Counter<_>>();

            for (i, j) in c.most_common() {
                println!("{} {}", i, j)
            }


            Ok(())
        }
        _ => unreachable!()
    }
}
