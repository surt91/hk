use std::fs::File;
use std::io::prelude::*;

use std::process::Command;

use structopt::StructOpt;

use hk::HegselmannKrause;
use hk::HegselmannKrauseLorenz;

/// Simulate a (modified) Hegselmann Krause model
#[derive(StructOpt, Debug)]
struct Opt {
    #[structopt(short, long)]
    /// number of interacting agents
    num_agents: u32,
    #[structopt(short, long, default_value = "2")]
    /// number of dimensions (only for Lorenz modification)
    dimension: u32,
    #[structopt(short = "l", long, default_value = "0.0")]
    /// minimum tolerance of agents (uniformly distributed)
    min_tolerance: f64,
    #[structopt(short = "u", long, default_value = "1.0")]
    /// maximum tolerance of agents (uniformly distributed)
    max_tolerance: f64,
    #[structopt(short, long, default_value = "1")]
    /// seed to use for the simulation
    seed: u64,
    #[structopt(short, long, default_value = "100")]
    /// number of sweeps to run the simulation
    iterations: u64,
    #[structopt(short, long, default_value = "1", possible_values = &["1", "2"])]
    /// which model to simulate:
    /// 1 -> Hegselmann Krause,
    /// 2 -> multidimensional Hegselmann Krause (Lorenz)
    model: u32,
    #[structopt(short, long, default_value = "out", parse(from_os_str))]
    /// name of the output data file
    outname: std::path::PathBuf,
}

// TODO: I should introduce the trait `model` and make everything below more generic
// a model should implement sweep, write_state and write_gp

fn main() -> std::io::Result<()> {
    let args = Opt::from_args();

    match args.model {
        1 => {
            let mut hk = HegselmannKrause::new(args.num_agents, args.min_tolerance as f32, args.max_tolerance as f32, args.seed);

            let outname = args.outname.with_extension("dat");
            let mut gp = File::create(args.outname.with_extension("gp"))?;
            hk.write_gp(&mut gp, outname.to_str().unwrap())?;

            let mut output = File::create(outname)?;
            for _ in 0..args.iterations {
                hk.sweep();
                hk.write_state(&mut output)?;
            }
            Ok(())
        },
        2 => {
            let mut hk = HegselmannKrauseLorenz::new(args.num_agents, args.min_tolerance as f32, args.max_tolerance as f32, args.dimension, args.seed);

            let dataname = args.outname.with_extension("dat");
            let clustername = args.outname.with_extension("cluster.dat");

            let mut output = File::create(&dataname)?;
            let mut output_cluster = File::create(&clustername)?;

            // simulate until converged
            if args.iterations == 0 {
                let mut ctr = 0;
                loop {
                    // test if we are converged
                    ctr += 1;
                    hk.sweep();
                    if hk.acc_change < 1e-7 {
                        write!(output, "# sweeps: {}\n", ctr)?;
                        hk.write_equilibrium(&mut output)?;
                        hk.write_cluster_sizes(&mut output_cluster)?;
                        Command::new("gzip")
                            .arg(format!("{}", dataname.to_str().unwrap()))
                            .arg(format!("{}", clustername.to_str().unwrap()))
                            .output()
                            .expect("failed to zip output file");
                        return Ok(())
                    }
                    hk.acc_change = 0.;
                }
            } else {
                let mut gp = File::create(args.outname.with_extension("gp"))?;
                hk.write_gp(&mut gp, dataname.to_str().unwrap())?;
                for _ in 0..args.iterations {
                    hk.sweep();
                    println!("{}", hk.acc_change);

                    hk.acc_change = 0.;
                    hk.write_state(&mut output)?;
                }
            }
            Ok(())
        },
        _ => unreachable!()
    }
}
