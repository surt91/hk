use std::fs::File;
use std::io::prelude::*;

use std::process::Command;

use structopt::StructOpt;

use hk::{HegselmannKrauseBuilder,HegselmannKrause};
use hk::HegselmannKrauseLorenz;
use hk::HegselmannKrauseLorenzSingle;
use hk::{anneal, local_anneal, Exponential, CostModel, PopulationModel};
use hk::models::graph;

/// Simulate a (modified) Hegselmann Krause model
#[derive(StructOpt, Debug)]
struct Opt {
    #[structopt(short, long)]
    /// number of interacting agents
    num_agents: u32,

    #[structopt(short, long, default_value = "2")]
    /// number of dimensions (only for Lorenz modification)
    dimension: u32,

    #[structopt(long, default_value = "1", possible_values = &["1", "2", "3", "4", "5"])]
    /// distribution of the tolerances epsilon_i
    /// 1 => uniform between min and max
    /// 2 => bimodal: half min, half max
    /// 3 => 15% of agents at x(0) = 0.25+-0.05, with confidence eps = 0.075+-0.05
    /// 4 => gaussian: min -> mean, max -> variance
    /// 5 => power law: min -> lower bound (scale), max -> exponent (= shape+1)
    tolerance_distribution: u32,

    #[structopt(short = "l", long, default_value = "0.0")]
    /// minimum tolerance of agents (uniformly distributed)
    min_tolerance: f64,

    #[structopt(short = "u", long, default_value = "1.0")]
    /// maximum tolerance of agents (uniformly distributed)
    max_tolerance: f64,

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

    #[structopt(short, long, default_value = "1")]
    /// seed to use for the simulation
    seed: u64,

    #[structopt(short, long, default_value = "100")]
    /// number of sweeps to run the simulation
    iterations: u64,

    #[structopt(long)]
    /// synchronous update instead of random sequential
    sync: bool,

    #[structopt(long, default_value = "1")]
    /// number of times to repeat the simulation
    samples: u32,

    #[structopt(short, long, default_value = "1", possible_values = &["1", "2", "3", "4", "5", "6", "7"])]
    /// which model to simulate:
    /// 1 -> Hegselmann Krause,
    /// 2 -> multidimensional Hegselmann Krause (Lorenz)
    /// 3 -> HK with active cost
    /// 4 -> multidimensional Hegselmann Krause (Lorenz) but only updating one dimension
    /// 5 -> HK with passive cost
    /// 6 -> HK annealing with cost and resources
    /// 7 -> HK annealing with local energy
    model: u32,

    #[structopt(short, long, default_value = "out", parse(from_os_str))]
    /// name of the output data file
    outname: std::path::PathBuf,
}

// TODO: I should introduce the trait `model` and make everything below more generic
// a model should implement sweep, write_state and write_gp

fn vis_hk_as_graph(hk: &HegselmannKrause, dotname: &std::path::PathBuf) -> Result<(), std::io::Error>{
    // let dotname = filename.with_extension(format!("{}.dot", ctr));
    println!("{:?}", dotname);
    let mut dotfile = File::create(&dotname)?;
    let g = graph::from_hk(&hk);
    let g2 = graph::condense(&g);
    let dot = graph::dot(&g2);
    write!(dotfile, "{}\n", dot)?;
    drop(dotfile);
    let dotimage = Command::new("fdp")
        .arg(format!("{}", dotname.to_str().unwrap()))
        .arg("-Tpng")
        .output()
        .expect("failed to create dot image");
    let mut dotimagefile = File::create(&dotname.with_extension("png"))?;
    dotimagefile.write(&dotimage.stdout)?;

    Ok(())
}

fn main() -> std::io::Result<()> {
    let args = Opt::from_args();
    let pop_model = match args.tolerance_distribution {
        1 => PopulationModel::Uniform,
        2 => PopulationModel::Bimodal,
        3 => PopulationModel::Bridgehead(0.25, 0.05, 0.15, 0.75, 0.05),
        4 => PopulationModel::Gaussian,
        5 => PopulationModel::PowerLaw,
        _ => unreachable!(),
    };

    match args.model {
        1 => {
            let mut hk = HegselmannKrauseBuilder::new(
                args.num_agents,
                args.min_tolerance as f32,
                args.max_tolerance as f32,
            ).seed(args.seed)
            .population_model(pop_model)
            .build();

            // let outname = args.outname.with_extension("dat");
            let clustername = args.outname.with_extension("cluster.dat");
            let densityname = args.outname.with_extension("density.dat");
            let mut density = File::create(&densityname)?;
            let mut output = File::create(&clustername)?;

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

                    if hk.acc_change < 1e-4 || (args.iterations > 0 && ctr > args.iterations) {
                        write!(output, "# sweeps: {}\n", ctr)?;
                        hk.fill_density();
                        break;
                    }
                    hk.acc_change = 0.;
                }
                hk.write_cluster_sizes(&mut output)?;
            }

            hk.write_density(&mut density)?;

            drop(output);
            Command::new("gzip")
                .arg(format!("{}", clustername.to_str().unwrap()))
                .output()
                .expect("failed to zip output file: cluster");

            drop(density);
            Command::new("gzip")
                .arg(format!("{}", densityname.to_str().unwrap()))
                .output()
                .expect("failed to zip output file: density");

            Ok(())
        },
        2 => {
            let mut hk = HegselmannKrauseLorenz::new(args.num_agents, args.min_tolerance as f32, args.max_tolerance as f32, args.dimension, args.seed);

            let dataname = args.outname.with_extension("dat");
            let clustername = args.outname.with_extension("cluster.dat");

            let mut output = File::create(&dataname)?;
            let mut output_cluster = File::create(&clustername)?;
            let mut density = File::create(args.outname.with_extension("density.dat"))?;

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
                        if hk.acc_change < 1e-7 {
                            write!(output, "# sweeps: {}\n", ctr)?;
                            // hk.write_equilibrium(&mut output)?;
                            hk.write_cluster_sizes(&mut output_cluster)?;
                            break;
                        }
                        hk.acc_change = 0.;
                    }
                }
                drop(output_cluster);
                Command::new("gzip")
                    // .arg(format!("{}", dataname.to_str().unwrap()))
                    .arg(format!("{}", clustername.to_str().unwrap()))
                    .output()
                    .expect("failed to zip output file");
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
            hk.write_density(&mut density)?;
            Ok(())
        },
        3 => {
            let mut hk = HegselmannKrauseBuilder::new(
                args.num_agents,
                args.min_tolerance as f32,
                args.max_tolerance as f32,
            ).seed(args.seed)
            .eta(args.eta as f32)
            .cost_model(CostModel::Rebounce)
            .population_model(pop_model)
            .resources(args.min_resources as f32, args.max_resources as f32)
            .build();

            // let outname = args.outname.with_extension("dat");
            // let mut gp = File::create(args.outname.with_extension("gp"))?;
            // hk.write_gp(&mut gp, outname.to_str().unwrap())?;
            let mut density = File::create(args.outname.with_extension("density.dat"))?;

            // let mut output = File::create(outname)?;
            for _ in 0..args.samples {
                for _ in 0..args.iterations {
                    hk.sweep();
                    // hk.write_state(&mut output)?;
                }
            }
            hk.write_density(&mut density)?;
            Ok(())
        },
        4 => {
            let mut hk = HegselmannKrauseLorenzSingle::new(args.num_agents, args.min_tolerance as f32, args.max_tolerance as f32, args.dimension, args.seed);

            // let dataname = args.outname.with_extension("dat");
            let clustername = args.outname.with_extension("cluster.dat");

            // let mut output = File::create(&dataname)?;
            let mut output_cluster = File::create(&clustername)?;
            let mut density = File::create(args.outname.with_extension("density.dat"))?;

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
                hk.write_cluster_sizes(&mut output_cluster)?;
            }
            drop(output_cluster);
            Command::new("gzip")
                // .arg(format!("{}", dataname.to_str().unwrap()))
                .arg(format!("{}", clustername.to_str().unwrap()))
                .output()
                .expect("failed to zip output file");
            hk.write_density(&mut density)?;
            Ok(())
        },
        5 => {
            let mut hk = HegselmannKrauseBuilder::new(
                args.num_agents,
                args.min_tolerance as f32,
                args.max_tolerance as f32,
            ).seed(args.seed)
            .eta(args.eta as f32)
            .cost_model(CostModel::Change)
            .population_model(pop_model)
            .resources(args.min_resources as f32, args.max_resources as f32)
            .build();

            // let outname = args.outname.with_extension("dat");
            let clustername = args.outname.with_extension("cluster.dat");
            let mut density = File::create(args.outname.with_extension("density.dat"))?;
            let mut output = File::create(&clustername)?;

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

                    if hk.acc_change < 1e-4 || (args.iterations > 0 && ctr > args.iterations) {
                        write!(output, "# sweeps: {}\n", ctr)?;
                        hk.fill_density();
                        break;
                    }
                    hk.acc_change = 0.;
                }
                hk.write_cluster_sizes(&mut output)?;
            }

            hk.write_density(&mut density)?;

            drop(output);
            Command::new("gzip")
                .arg(format!("{}", clustername.to_str().unwrap()))
                .output()
                .expect("failed to zip output file");

            Ok(())
        },
        6 => {
            use rand::SeedableRng;
            use rand_pcg::Pcg64;

            let mut hk = HegselmannKrauseBuilder::new(
                args.num_agents,
                args.min_tolerance as f32,
                args.max_tolerance as f32,
            ).seed(args.seed)
            .eta(args.eta as f32)
            .cost_model(CostModel::Annealing)
            .population_model(pop_model)
            .resources(args.min_resources as f32, args.max_resources as f32)
            .build();

            let mut rng = Pcg64::seed_from_u64(args.seed);

            let clustername = args.outname.with_extension("cluster.dat");
            let mut density = File::create(args.outname.with_extension("density.dat"))?;
            let mut energy = File::create(args.outname.with_extension("energy.dat"))?;
            let mut output = File::create(&clustername)?;

            for n in 0..args.samples {
                let schedule = Exponential::new(args.iterations as usize, 3., 0.98);
                // let schedule = Linear::new(args.iterations as usize, 0.1);
                // let schedule = Linear::new(args.iterations as usize, 0.);
                hk.reset();
                let e = anneal(&mut hk, schedule, &mut rng);
                write!(energy, "{}\n", e)?;
                hk.write_cluster_sizes(&mut output)?;
            }

            hk.write_density(&mut density)?;

            drop(output);
            Command::new("gzip")
                .arg(format!("{}", clustername.to_str().unwrap()))
                .output()
                .expect("failed to zip output file");

            Ok(())
        },
        7 => {
            use rand::SeedableRng;
            use rand_pcg::Pcg64;

            let mut hk = HegselmannKrauseBuilder::new(
                args.num_agents,
                args.min_tolerance as f32,
                args.max_tolerance as f32,
            ).seed(args.seed)
            .eta(args.eta as f32)
            .cost_model(CostModel::Annealing)
            .population_model(pop_model)
            .resources(args.min_resources as f32, args.max_resources as f32)
            .build();

            let mut rng = Pcg64::seed_from_u64(args.seed);

            let clustername = args.outname.with_extension("cluster.dat");
            let mut energy = File::create(args.outname.with_extension("energy.dat"))?;
            let mut density = File::create(args.outname.with_extension("density.dat"))?;
            let mut output = File::create(&clustername)?;

            for _ in 0..args.samples {
                let schedule = Exponential::new(args.iterations as usize, 3., 0.98);
                // let schedule = Linear::new(args.iterations as usize, 0.1);
                // let schedule = Linear::new(args.iterations as usize, 0.);
                hk.reset();
                let e = local_anneal(&mut hk, schedule, &mut rng);
                write!(energy, "{}\n", e)?;
                hk.write_cluster_sizes(&mut output)?;
            }

            hk.write_density(&mut density)?;

            drop(output);
            Command::new("gzip")
                .arg(format!("{}", clustername.to_str().unwrap()))
                .output()
                .expect("failed to zip output file");

            Ok(())
        },
        _ => unreachable!()
    }
}
