use std::mem;
use std::fmt;
use std::fs::File;
use std::io::prelude::*;

// use rand::{Rng, SeedableRng};
use rand::prelude::*;
use rand_pcg::Pcg64;
use rand_distr::Dirichlet;
use itertools::Itertools;

use super::hk_lorenz::HKLorenzAgent;

const EPS: f32 = 1e-6;


pub struct HegselmannKrauseLorenzSingle {
    num_agents: u32,
    dimension: u32,
    agents: Vec<HKLorenzAgent>,
    min_tolerance: f32,
    max_tolerance: f32,

    pub acc_change: f32,
    dirichlet: Dirichlet<f32>,

    dynamic_density: Vec<Vec<Vec<u64>>>,

    // we need many, good (but not crypto) random numbers
    // we will use here the pcg generator
    rng: Pcg64,
}

impl PartialEq for HegselmannKrauseLorenzSingle {
    fn eq(&self, other: &HegselmannKrauseLorenzSingle) -> bool {
        self.agents == other.agents
    }
}

impl fmt::Debug for HegselmannKrauseLorenzSingle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HKL {{ N: {}, agents: {:?} }}", self.num_agents, self.agents)
    }
}

impl HegselmannKrauseLorenzSingle {
    pub fn new(n: u32, min_tolerance: f32, max_tolerance: f32, dim: u32, seed: u64) -> HegselmannKrauseLorenzSingle {
        let rng = Pcg64::seed_from_u64(seed);
        let dirichlet = Dirichlet::new_with_size(1.0, dim as usize).unwrap();
        let agents: Vec<HKLorenzAgent> = Vec::new();

        let dynamic_density =  Vec::new();

        let mut hk = HegselmannKrauseLorenzSingle {
            num_agents: n,
            dimension: dim,
            agents,
            min_tolerance,
            max_tolerance,
            acc_change: 0.,
            dirichlet,
            dynamic_density,
            rng,
        };
        hk.reset();
        hk
    }

    fn stretch(x: f32, low: f32, high: f32) -> f32 {
        x*(high-low)+low
    }

    pub fn reset(&mut self) {
        self.agents = (0..self.num_agents).map(|_| HKLorenzAgent::new(
            self.dirichlet.sample(&mut self.rng),
            HegselmannKrauseLorenzSingle::stretch(self.rng.gen(), self.min_tolerance, self.max_tolerance)
        )).collect();

        self.dynamic_density = vec![Vec::new(); self.dimension as usize];
    }

    // TOOD: maybe we can get some speedup using an r-tree or similar
    pub fn step_naive(&mut self) {
        // get a random agent
        let idx = self.rng.gen_range(0, self.num_agents) as usize;
        let i = &self.agents[idx];

        // reset our allocated temporary vector
        let mut tmp = 0.0;
        let mut count = 0;

        // get a random dimension and assign the mean of the neighbors
        // opinion in this aspect
        // then modifiy the other opinions to preserve the bounds on the opinion values
        let chosen_dim = (self.rng.gen::<f32>() * self.dimension as f32) as usize;
        for j in self.agents.iter()
                .filter(|j| i.dist(j) < i.tolerance) {
            tmp += j.opinion[chosen_dim];
            count += 1;
        }

        tmp /= count as f32;
        self.acc_change += (tmp - i.opinion[chosen_dim]).abs();

        let tmp2 = i.opinion.clone();
        self.agents[idx].opinion[chosen_dim] = tmp;
        let diff = 1. - self.agents[idx].opinion.iter().sum::<f32>();
        let sum = self.agents[idx].opinion.iter()
            .enumerate()
            .filter(|(n, _)| n != &chosen_dim)
            .map(|(_, x)| x)
            .sum::<f32>();

        for j in 0..self.dimension as usize {
            if j == chosen_dim {
                continue;
            }
            self.agents[idx].opinion[j] += diff * tmp2[j]/sum;
            assert!(self.agents[idx].opinion[j] <= 1. && self.agents[idx].opinion[j] >= 0.);
        }

        assert!((1. - self.agents[idx].opinion.iter().sum::<f32>()).abs() < 1e-6);
    }

    pub fn sweep(&mut self) {
        for _ in 0..self.num_agents {
            self.step_naive();
        }
        self.add_state_to_density()
    }

    fn sync_new_opinions(&self) -> (Vec<Vec<f32>>, f32) {
        let mut acc_change = 0.;
        let op = self.agents.iter().map(|i| {
            let mut tmp = vec![0.; self.dimension as usize];
            let mut count = 0;
            for j in self.agents.iter()
                    .filter(|j| i.dist(j) < i.tolerance) {
                for k in 0..self.dimension as usize {
                    tmp[k] += j.opinion[k];
                }
                count += 1;
            }
            for k in 0..self.dimension as usize {
                tmp[k] /= count as f32;
                acc_change += (tmp[k] - i.opinion[k]).abs();
            }
            tmp
        }).collect();
        (op, acc_change)
    }

    pub fn sweep_synchronous(&mut self) {
        let (mut new_opinions, acc_change) = self.sync_new_opinions();
        self.acc_change += acc_change;
        for i in 0..self.num_agents as usize {
            mem::swap(&mut self.agents[i].opinion, &mut new_opinions[i]);
        }
        self.add_state_to_density()
    }

    /// A cluster are agents whose distance is less than EPS
    fn list_clusters(&self) -> Vec<Vec<HKLorenzAgent>> {
        let mut clusters: Vec<Vec<HKLorenzAgent>> = Vec::new();
        'agent: for i in &self.agents {
            for c in &mut clusters {
                if i.dist(&c[0]) < EPS {
                    c.push(i.clone());
                    continue 'agent;
                }
            }
            clusters.push(vec![i.clone(); 1])
        }
        clusters
    }

    pub fn cluster_sizes(&self) -> Vec<u32> {
        let clusters = self.list_clusters();
        clusters.iter()
            .map(|c| c.len() as u32)
            .collect()
    }

    fn add_state_to_density(&mut self) {
        for d in 0..self.dimension as usize {
            let mut slice = vec![0; 100];
            for i in &self.agents {
                // subtract a little bit to avoid problems if opinion is 1
                slice[(i.opinion[d]*100.-1e-4) as usize] += 1;
            }
            self.dynamic_density[d].push(slice);
        }
    }

    pub fn write_density(&self, file: &mut File) -> std::io::Result<()> {
        for d in 0..self.dimension as usize {
            let string_list = self.dynamic_density[d].iter()
                .map(|x| x.iter().join(" "))
                .join("\n");
            write!(file, "{}\n", string_list)?;
        }
        Ok(())
    }

    pub fn write_state(&self, file: &mut File) -> std::io::Result<()> {
        let string_list = self.agents.iter()
            .map(|j| j.opinion.iter().map(|x| x.to_string()).join(" "))
            .join(" ");
        write!(file, "{}\n", string_list)
    }

    pub fn write_equilibrium(&self, file: &mut File) -> std::io::Result<()> {
        let string_list = self.agents.iter()
            .map(|j| j.opinion.iter().map(|x| x.to_string()).join(" "))
            .join("\n");
        write!(file, "{}\n", string_list)
    }

    pub fn write_cluster_sizes(&self, file: &mut File) -> std::io::Result<()> {
        let clusters = self.list_clusters();

        let string_list = clusters.iter()
            .map(|c| c[0].opinion.iter().map(|x| x.to_string()).join(" "))
            .join(" ");
        write!(file, "# {}\n", string_list)?;

        let string_list = clusters.iter()
            .map(|c| c.len().to_string())
            .join(" ");
        write!(file, "{}\n", string_list)?;
        Ok(())
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
