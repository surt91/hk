use std::mem;
use std::fmt;
use std::fs::File;
use std::io::prelude::*;

// use rand::{Rng, SeedableRng};
use rand::prelude::*;
use rand_pcg::Pcg64;
use rand_distr::Dirichlet;
use itertools::Itertools;

const EPS: f64 = 1e-6;

#[derive(Clone, Debug)]
struct HKLorenzAgent {
    opinion: Vec<f64>,
    tolerance: f64,
}

impl HKLorenzAgent {
    fn new(opinion: Vec<f64>, tolerance: f64) -> HKLorenzAgent {
        HKLorenzAgent {
            opinion,
            tolerance,
        }
    }

    fn dist(&self, other: &HKLorenzAgent) -> f64 {
        assert!(self.opinion.len() == other.opinion.len());
        self.opinion.iter()
            .zip(&other.opinion)
            .map(|(a, b)| (a-b)*(a-b))
            .sum::<f64>()
            .sqrt()
    }
}

impl PartialEq for HKLorenzAgent {
    fn eq(&self, other: &HKLorenzAgent) -> bool {
        // (self.opinion - other.opinion).abs() < EPS
            // && (self.tolerance - other.tolerance).abs() < EPS
        (self.tolerance - other.tolerance).abs() < EPS
    }
}

pub struct HegselmannKrauseLorenz {
    num_agents: u32,
    dimension: u32,
    agents: Vec<HKLorenzAgent>,

    tmp: Vec<f64>,

    // we need many, good (but not crypto) random numbers
    // we will use here the pcg generator
    rng: Pcg64,
}

impl PartialEq for HegselmannKrauseLorenz {
    fn eq(&self, other: &HegselmannKrauseLorenz) -> bool {
        self.agents == other.agents
    }
}

impl fmt::Debug for HegselmannKrauseLorenz {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HKL {{ N: {}, agents: {:?} }}", self.num_agents, self.agents)
    }
}

impl HegselmannKrauseLorenz {
    pub fn new(n: u32, dim: u32, seed: u64) -> HegselmannKrauseLorenz {
        let mut rng = Pcg64::seed_from_u64(seed);
        let dirichlet = Dirichlet::new_with_size(1.0, dim as usize).unwrap();
        let agents: Vec<HKLorenzAgent> = (0..n).map(|_| HKLorenzAgent::new(dirichlet.sample(&mut rng), rng.gen())).collect();

        HegselmannKrauseLorenz {
            num_agents: n,
            dimension: dim,
            agents,
            tmp: vec![0.; dim as usize],
            rng,
        }
    }

    pub fn step_naive(&mut self) {
        // get a random agent
        let idx = self.rng.gen_range(0, self.num_agents) as usize;
        let i = &self.agents[idx];

        // reset our allocated temporary vector
        for j in 0..self.dimension as usize {
            self.tmp[j] = 0.0;
        }
        let mut count = 0;

        for j in self.agents.iter()
                .filter(|j| i.dist(j) < i.tolerance) {
            for k in 0..self.dimension as usize {
                self.tmp[k] += j.opinion[k];
            }
            count += 1;
        }

        for j in 0..self.dimension as usize {
            self.tmp[j] /= count as f64;
        }

        mem::swap(&mut self.agents[idx].opinion, &mut self.tmp);
    }

    pub fn sweep(&mut self) {
        for _ in 0..self.num_agents {
            self.step_naive();
        }
    }

    pub fn write_state(&self, file: &mut File) -> std::io::Result<()> {
        let string_list = self.agents.iter()
            .map(|j| j.opinion.iter().map(|x| x.to_string()).join(" "))
            .join(" ");
        write!(file, "{}\n", string_list)
    }

    pub fn write_gp(&self, file: &mut File, outfilename: &str) -> std::io::Result<()> {
        write!(file, "set terminal pngcairo\n")?;

        for i in 0..self.dimension {
            write!(file, "set output '{}_d{}.png'\n", outfilename, i)?;
            write!(file, "set xl 't'\n")?;
            write!(file, "set yl 'x_i'\n")?;
            write!(file, "p '{}' u 0:{} w l not, ", outfilename, i+1)?;

            let string_list = (2..self.num_agents)
                .map(|j| format!("'' u 0:{} w l not,", j*self.dimension+i))
                .join(" ");
            write!(file, "{}\n", string_list)?;
        }

        Ok(())
    }
}
