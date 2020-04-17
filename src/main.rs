use std::fs::File;
use std::io::prelude::*;
use std::path::PathBuf;

use std::process::Command;

use structopt::StructOpt;

use itertools::Itertools;

use hk::{HegselmannKrauseBuilder,HegselmannKrause};
use hk::HegselmannKrauseLorenz;
use hk::HegselmannKrauseLorenzSingle;
use hk::{anneal, anneal_sweep, local_anneal, Exponential, Constant, CostModel, ResourceModel, PopulationModel};
use hk::models::graph;

use git_version::git_version;
const GIT_VERSION: &str = git_version!();

const ACC_EPS: f32 = 1e-3;

/// Simulate a (modified) Hegselmann Krause model
#[derive(StructOpt, Debug)]
#[structopt(version = GIT_VERSION)]
struct Opt {
    #[structopt(short, long)]
    /// number of interacting agents
    num_agents: u32,

    #[structopt(short, long, default_value = "2")]
    /// number of dimensions (only for Lorenz modification)
    dimension: u32,

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

    #[structopt(long, default_value = "1", possible_values = &["1", "2", "3", "4"])]
    /// distribution of the resources c_i:{n}
    /// 1 => uniform between min and max{n}
    /// 2 => pareto with exponent -2.5{n}
    /// 3 => proportional to the tolerances but with same average total resources{n}
    /// 4 => antiproportional to the tolerances but with same average total resources{n}
    resource_distribution: u32,

    #[structopt(long, default_value = "0")]
    /// minimal resources for HKCost
    min_resources: f64,

    #[structopt(long, default_value = "1")]
    /// maximal resources for HKCost
    max_resources: f64,

    #[structopt(long, default_value = "0.01")]
    /// weight of cost
    eta: f64,

    #[structopt(short = "r", long, default_value = "5")]
    /// start resources for HKAC
    start_resources: f64,

    #[structopt(short = "T", long, default_value = "1.")]
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

    #[structopt(short, long, default_value = "1", possible_values = &["1", "2", "3", "4", "5", "6", "7", "8"])]
    /// which model to simulate:{n}
    /// 1 -> Hegselmann Krause,{n}
    /// 2 -> multidimensional Hegselmann Krause (Lorenz){n}
    /// 3 -> HK with active cost{n}
    /// 4 -> multidimensional Hegselmann Krause (Lorenz) but only updating one dimension{n}
    /// 5 -> HK with passive cost{n}
    /// 6 -> HK annealing with cost and resources{n}
    /// 7 -> HK annealing with local energy{n}
    /// 8 -> HK annealing with constant temperature{n}
    model: u32,

    #[structopt(long, default_value = ".", parse(from_os_str))]
    /// directory to store temporary files
    tmp: PathBuf,

    #[structopt(short, long, default_value = "out", parse(from_os_str))]
    /// name of the output data file
    outname: PathBuf,
}

// TODO: I should introduce the trait `model` and make everything below more generic
// a model should implement sweep, write_state and write_gp

fn vis_hk_as_graph(hk: &HegselmannKrause, dotname: &std::path::PathBuf) -> Result<(), std::io::Error>{
    // let dotname = filename.with_extension(format!("{}.dot", ctr));
    println!("{:?}", dotname);
    let mut dotfile = File::create(&dotname)?;
    let g = graph::from_hk(&hk);
    // let g = graph::condense(&g);
    let dot = graph::dot(&g);
    writeln!(dotfile, "{}", dot)?;
    drop(dotfile);
    let dotimage = Command::new("fdp")
        .arg(dotname.to_str().unwrap())
        .arg("-Tpng")
        .output()
        .expect("failed to create dot image");
    let mut dotimagefile = File::create(&dotname.with_extension("png"))?;
    dotimagefile.write_all(&dotimage.stdout)?;

    Ok(())
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

fn write_entropy(clusters: &[usize], file: &mut File) -> std::io::Result<()> {
    let s = entropy(&clusters);
    writeln!(file, "{}", s)?;
    Ok(())
}

fn entropy (clustersizes: &[usize]) -> f32 {
    let f = 1. / clustersizes.iter().sum::<usize>() as f32;
    clustersizes.iter().map(|c| {
        let p = *c as f32 * f;
        - p * p.ln()
    }).sum()
}

struct Output {
    tmp_file: File,
    tmp_path: PathBuf,
    final_path: PathBuf,
}

impl Output {
    fn new(outname: &PathBuf, extension: &str, tmp_path: &PathBuf) -> std::io::Result<Output> {
        let final_path = outname.with_extension(extension);
        let tmp_path = tmp_path.join(
                outname.file_name().expect("no filename specified")
            ).with_extension(extension);

        if let Some(dirs) = tmp_path.parent() {
            std::fs::create_dir_all(&dirs)?;
        }
        let tmp_file = File::create(&tmp_path)?;

        Ok(Output {
            tmp_file,
            tmp_path,
            final_path,
        })
    }

    fn file(&mut self) -> &mut File {
        &mut self.tmp_file
    }

    fn zip(name: &std::path::PathBuf) {
        Command::new("gzip")
            .arg(name.to_str().unwrap())
            .output()
            .expect("failed to zip output file");
    }

    fn finalize(self) -> std::io::Result<()> {
        let tmp_path = self.tmp_path;
        let final_path = self.final_path;
        // flush and close temporary file
        self.tmp_file.sync_all()?;
        drop(self.tmp_file);

        // zip temporary file
        Output::zip(&tmp_path);
        let mut gz_ext = tmp_path.extension().unwrap().to_os_string();
        gz_ext.push(".gz");
        let tmp_path = tmp_path.with_extension(&gz_ext);
        let final_path = final_path.with_extension(&gz_ext);

        // move finished file to final location, (if they differ)
        if let Some(dirs) = final_path.parent() {
            std::fs::create_dir_all(&dirs)?;
        }
        if tmp_path != final_path {
            std::fs::rename(&tmp_path, &final_path).or_else(|_| {
                std::fs::copy(&tmp_path, &final_path).expect("could not move or copy the file");
                std::fs::remove_file(&tmp_path)
            })?;
        }

        Ok(())
    }
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
            println!("Warning: you use a not well tested secondary type!");
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
            let prop = (args.min_resources + args.max_resources) / (args.min_tolerance + args.max_tolerance);
            ResourceModel::Proportional(prop as f32)
        },
        4 => {
            let prop = (args.min_resources + args.max_resources) * (args.min_tolerance + args.max_tolerance) / 4.;
            ResourceModel::Antiproportional(prop as f32)
        },
        _ => unreachable!(),
    };

    match args.model {
        1 | 3 | 5 => {
            let mut hk = HegselmannKrauseBuilder::new(
                args.num_agents,
            ).seed(args.seed)
            .cost_model(cost_model)
            .resource_model(resource_model)
            .population_model(pop_model)
            .build();

            let mut out_cluster = Output::new(&args.outname, "cluster.dat", &args.tmp)?;
            let mut out_nopoor = Output::new(&args.outname, "nopoor.dat", &args.tmp)?;
            let mut out_scc = Output::new(&args.outname, "scc.dat", &args.tmp)?;
            let mut out_density = Output::new(&args.outname, "density.dat", &args.tmp)?;
            let mut out_entropy = Output::new(&args.outname, "entropy.dat", &args.tmp)?;
            let mut out_info = Output::new(&args.outname, "info.dat", &args.tmp)?;

            write!(out_info.file(), "{}\n{}\n", info_version, info_args)?;

            for _ in 0..args.samples {
                hk.reset();

                let mut ctr = 0;
                loop {
                    // test if we are converged
                    ctr += 1;

                    if args.sync {
                        hk.sweep_synchronous();
                    } else {
                        hk.sweep();
                    }

                    if hk.acc_change < ACC_EPS || (args.iterations > 0 && ctr > args.iterations) {
                        writeln!(out_cluster.file(), "# sweeps: {}", ctr)?;
                        hk.fill_density();
                        break;
                    }
                    hk.acc_change = 0.;
                }
                hk.write_cluster_sizes(&mut out_cluster.file())?;
                hk.write_cluster_sizes_nopoor(&mut out_nopoor.file())?;

                if args.scc {
                    let clusters = cluster_sizes_from_graph(&hk);
                    write_cluster_sizes(&clusters, &mut out_scc.file())?;
                }

                // vis_hk_as_graph(&hk, &args.outname.with_extension(format!("{}.dot", n)))?;
            }

            hk.write_density(&mut out_density.file())?;
            hk.write_entropy(&mut out_entropy.file())?;

            out_cluster.finalize()?;
            out_nopoor.finalize()?;
            out_scc.finalize()?;
            out_density.finalize()?;
            out_entropy.finalize()?;
            out_info.finalize()?;

            Ok(())
        },
        2 => {
            let mut hk = HegselmannKrauseLorenz::new(args.num_agents, args.min_tolerance as f32, args.max_tolerance as f32, args.dimension, args.seed);

            let mut out_data = Output::new(&args.outname, "dat", &args.tmp)?;
            let mut out_cluster = Output::new(&args.outname, "cluster.dat", &args.tmp)?;
            let mut out_density = Output::new(&args.outname, "density.dat", &args.tmp)?;

            // simulate until converged
            if args.iterations == 0 {
                for _ in 0..args.samples {
                    hk.reset();
                    let mut ctr = 0;
                    loop {
                        // test if we are converged
                        ctr += 1;
                        hk.sweep();
                        // hk.sweep_synchronous();
                        if hk.acc_change < ACC_EPS {
                            writeln!(out_data.file(), "# sweeps: {}", ctr)?;
                            // hk.write_equilibrium(&mut output)?;
                            hk.write_cluster_sizes(&mut out_cluster.file())?;
                            break;
                        }
                        hk.acc_change = 0.;
                    }
                }

            }
            // } else {
            //     unimplemented!();
            //     let mut gp = File::create(args.outname.with_extension("gp"))?;
            //     hk.write_gp(&mut gp, dataname.to_str().unwrap())?;
            //     for _ in 0..args.iterations {
            //         hk.sweep();
            //
            //         hk.acc_change = 0.;
            //         hk.write_state(&mut output)?;
            //     }
            //     hk.write_cluster_sizes(&mut output_cluster)?;
            // }
            hk.write_density(&mut out_density.file())?;

            out_data.finalize()?;
            out_cluster.finalize()?;
            out_density.finalize()?;

            Ok(())
        },
        4 => {
            let mut hk = HegselmannKrauseLorenzSingle::new(args.num_agents, args.min_tolerance as f32, args.max_tolerance as f32, args.dimension, args.seed);

            let mut out_cluster = Output::new(&args.outname, "cluster.dat", &args.tmp)?;
            let mut out_density = Output::new(&args.outname, "density.dat", &args.tmp)?;

            for _ in 0..args.samples {
                hk.reset();
                // let mut ctr = 0;
                for _ in 0..args.iterations {
                    // test if we are converged
                    // ctr += 1;
                    hk.sweep();
                    // hk.sweep_synchronous();
                    // if hk.acc_change < 1e-7 {
                    //     write!(output, "# sweeps: {}\n", ctr)?;
                    //     // hk.write_equilibrium(&mut output)?;
                    //     hk.write_cluster_sizes(&mut output_cluster)?;
                    //     break;
                    // }
                    hk.acc_change = 0.;
                }
                hk.write_cluster_sizes(&mut out_cluster.file())?;
            }
            hk.write_density(&mut out_density.file())?;

            out_cluster.finalize()?;
            out_density.finalize()?;

            Ok(())
        },
        6 => {
            use rand::SeedableRng;
            use rand_pcg::Pcg64;

            let mut hk = HegselmannKrauseBuilder::new(
                args.num_agents,
            ).seed(args.seed)
            .cost_model(CostModel::Annealing(args.eta as f32))
            .resource_model(resource_model)
            .population_model(pop_model)
            .build();

            let mut rng = Pcg64::seed_from_u64(args.seed);

            let mut out_cluster = Output::new(&args.outname, "cluster.dat", &args.tmp)?;
            let mut out_energy = Output::new(&args.outname, "energy.dat", &args.tmp)?;
            let mut out_density = Output::new(&args.outname, "density.dat", &args.tmp)?;
            let mut out_entropy = Output::new(&args.outname, "entropy.dat", &args.tmp)?;

            for _n in 0..args.samples {
                let schedule = Exponential::new(args.iterations as usize, 3., 0.98);
                // let schedule = Linear::new(args.iterations as usize, 0.1);
                // let schedule = Linear::new(args.iterations as usize, 0.);
                hk.reset();
                let e = anneal(&mut hk, schedule, &mut rng);
                writeln!(out_energy.file(), "{}", e)?;

                if args.scc {
                    let clusters = cluster_sizes_from_graph(&hk);
                    write_cluster_sizes(&clusters, &mut out_cluster.file())?;
                    write_entropy(&clusters, &mut out_entropy.file())?;
                } else {
                    hk.write_cluster_sizes(&mut out_cluster.file())?;
                }

                // vis_hk_as_graph(&hk, &args.outname.with_extension(format!("{}.dot", n)))?;
                // println!("{}", hk.agents.iter().filter(|x| x.resources > 0.).count())
            }

            hk.write_density(&mut out_density.file())?;

            out_cluster.finalize()?;
            out_energy.finalize()?;
            out_density.finalize()?;
            out_entropy.finalize()?;

            Ok(())
        },
        7 => {
            use rand::SeedableRng;
            use rand_pcg::Pcg64;

            let mut hk = HegselmannKrauseBuilder::new(
                args.num_agents,
            ).seed(args.seed)
            .cost_model(CostModel::Annealing(args.eta as f32))
            .resource_model(resource_model)
            .population_model(pop_model)
            .build();

            let mut rng = Pcg64::seed_from_u64(args.seed);

            let mut out_cluster = Output::new(&args.outname, "cluster.dat", &args.tmp)?;
            let mut out_energy = Output::new(&args.outname, "energy.dat", &args.tmp)?;
            let mut out_density = Output::new(&args.outname, "density.dat", &args.tmp)?;

            for _ in 0..args.samples {
                let schedule = Exponential::new(args.iterations as usize, 3., 0.98);
                // let schedule = Linear::new(args.iterations as usize, 0.1);
                // let schedule = Linear::new(args.iterations as usize, 0.);
                hk.reset();
                let e = local_anneal(&mut hk, schedule, &mut rng);
                writeln!(out_energy.file(), "{}", e)?;
                hk.write_cluster_sizes(&mut out_cluster.file())?;
            }

            hk.write_density(&mut out_density.file())?;

            out_cluster.finalize()?;
            out_energy.finalize()?;
            out_density.finalize()?;

            Ok(())
        },
        8 => {
            use rand::SeedableRng;
            use rand_pcg::Pcg64;

            let mut hk = HegselmannKrauseBuilder::new(
                args.num_agents,
            ).seed(args.seed)
            .cost_model(CostModel::Annealing(args.eta as f32))
            .resource_model(resource_model)
            .population_model(pop_model)
            .build();

            let mut rng = Pcg64::seed_from_u64(args.seed);

            let mut out_cluster = Output::new(&args.outname, "cluster.dat", &args.tmp)?;
            let mut out_energy = Output::new(&args.outname, "energy.dat", &args.tmp)?;
            let mut out_density = Output::new(&args.outname, "density.dat", &args.tmp)?;
            let mut out_entropy = Output::new(&args.outname, "entropy.dat", &args.tmp)?;
            let mut out_changes = Output::new(&args.outname, "changes.dat", &args.tmp)?;

            for _n in 0..args.samples {
                let schedule = Constant::new(args.temperature as f32, args.iterations as usize);
                hk.reset();
                for t in schedule {
                    hk.acc_change = 0.;
                    let e = anneal_sweep(&mut hk, &mut rng, t);
                    writeln!(out_energy.file(), "{}", e)?;
                    writeln!(out_changes.file(), "{}", hk.acc_change)?;
                    hk.add_state_to_density();
                    hk.time += 1;
                }

                // hk.write_cluster_sizes(&mut output)?;
                let clusters = cluster_sizes_from_graph(&hk);
                write_cluster_sizes(&clusters, &mut out_cluster.file())?;
                write_entropy(&clusters, &mut out_entropy.file())?;
            }

            hk.write_density(&mut out_density.file())?;

            out_cluster.finalize()?;
            out_energy.finalize()?;
            out_density.finalize()?;
            out_entropy.finalize()?;
            out_changes.finalize()?;

            Ok(())
        },
        _ => unreachable!()
    }
}
