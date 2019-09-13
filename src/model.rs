use std::fs::File;
use std::io::prelude::*;

use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use itertools::Itertools;

struct HKAgent {
    opinion: f32,
    tolerance: f32,
}

impl HKAgent {
    fn new(opinion: f32, tolerance: f32) -> HKAgent {
        HKAgent {
            opinion,
            tolerance,
        }
    }
}

pub struct HegselmannKrause {
    num_agents: u32,
    agents: Vec<HKAgent>,
    // we need many, good (but not crypto) random numbers
    // we will use here the pcg generator
    rng: Pcg64,
}

impl HegselmannKrause {
    pub fn new(n: u32, seed: u64) -> HegselmannKrause {
        let mut rng = Pcg64::seed_from_u64(seed);
        let agents = (0..n).map(|_| HKAgent::new(rng.gen(), rng.gen())).collect();

        HegselmannKrause {
            num_agents: n,
            agents,
            rng
        }
    }

    fn step_naive(&mut self) {
        // get a random agent
        let idx = self.rng.gen_range(0, self.num_agents) as usize;
        let i = &self.agents[idx];

        let (sum, count) = self.agents.iter()
            .map(|j| j.opinion)
            .filter(|j| (i.opinion - j).abs() < i.tolerance)
            .fold((0f32, 0u32), |(sum, count), i| (sum + i, count + 1));

        self.agents[idx].opinion = sum/count as f32;
    }

    pub fn sweep(&mut self) {
        for _ in 0..self.num_agents {
            self.step_naive();
        }
    }

    pub fn write_state(&self, file: &mut File) -> std::io::Result<()> {
        let string_list = self.agents.iter()
            .map(|j| j.opinion.to_string())
            .join(" ");
        write!(file, "{}\n", string_list)
    }

    pub fn write_gp(&self, file: &mut File, outfilename: &str) -> std::io::Result<()> {
        write!(file, "set xl 't'\n")?;
        write!(file, "set yl 'x_i'\n")?;
        write!(file, "p '{}' u 0:1 w l not, ", outfilename)?;

        let string_list = (2..self.num_agents)
            .map(|j| format!("'' u 0:{} w l not,", j))
            .join(" ");
        write!(file, "{}", string_list)
    }
}
