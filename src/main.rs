use std::fs::File;

use structopt::StructOpt;

use hk::HegselmannKrause;
use hk::HegselmannKrauseLorenz;

/// Search for a pattern in a file and display the lines that contain it.
#[derive(StructOpt, Debug)]
struct Opt {
    #[structopt(short, long)]
    /// number of interacting agents
    num_agents: u32,
    #[structopt(short, long, default_value = "2")]
    /// number of dimensions (only for Lorenz modification)
    dimension: u32,
    #[structopt(short, long, default_value = "1")]
    /// seed to use for the simulation
    seed: u64,
    #[structopt(short, long, default_value = "100")]
    /// number of sweeps to run the simulation
    iterations: u64,
    #[structopt(short, long, default_value = "1", possible_values = &["1", "2"])]
    /// which model to simulate
    /// 1 -> Hegselmann Krause
    /// 2 -> multidimensional Hegselmann Krause (Lorenz)
    model: u32,
    #[structopt(short, long, default_value = "out", parse(from_os_str))]
    /// name of the output data file
    outname: std::path::PathBuf,
}

fn main() -> std::io::Result<()> {
    let args = Opt::from_args();

    match args.model {
        1 => {
            let mut hk = HegselmannKrause::new(args.num_agents, args.seed);

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
            let mut hk = HegselmannKrauseLorenz::new(args.num_agents, args.dimension, args.seed);

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
        _ => unreachable!()
    }
}
