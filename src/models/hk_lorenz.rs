use std::mem;
use std::fmt;
use std::fs::File;
use std::io::prelude::*;

// use rand::{Rng, SeedableRng};
use rand::prelude::*;
use rand_pcg::Pcg64;
use rand_distr::Dirichlet;
use itertools::Itertools;

const EPS: f32 = 1e-6;

#[derive(Clone, Debug)]
struct HKLorenzAgent {
    opinion: Vec<f32>,
    tolerance: f32,
}

impl HKLorenzAgent {
    fn new(opinion: Vec<f32>, tolerance: f32) -> HKLorenzAgent {
        HKLorenzAgent {
            opinion,
            tolerance,
        }
    }

    fn dist(&self, other: &HKLorenzAgent) -> f32 {
        assert!(self.opinion.len() == other.opinion.len());
        self.opinion.iter()
            .zip(&other.opinion)
            .map(|(a, b)| (a-b)*(a-b))
            .sum::<f32>()
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

    tmp: Vec<f32>,
    pub acc_change: f32,

    dynamic_density: Vec<Vec<Vec<u64>>>,

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
    pub fn new(n: u32, min_tolerance: f32, max_tolerance: f32, dim: u32, seed: u64) -> HegselmannKrauseLorenz {
        let mut rng = Pcg64::seed_from_u64(seed);
        let dirichlet = Dirichlet::new_with_size(1.0, dim as usize).unwrap();
        let stretch = |x: f32| x*(max_tolerance-min_tolerance)+min_tolerance;
        let agents: Vec<HKLorenzAgent> = (0..n).map(|_| HKLorenzAgent::new(
            dirichlet.sample(&mut rng),
            stretch(rng.gen())
        )).collect();

        let dynamic_density = vec![Vec::new(); dim as usize];

        HegselmannKrauseLorenz {
            num_agents: n,
            dimension: dim,
            agents,
            tmp: vec![0.; dim as usize],
            acc_change: 0.,
            dynamic_density,
            rng,
        }
    }

    // TOOD: maybe we can get some speedup using an r-tree or similar
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
            self.tmp[j] /= count as f32;
            self.acc_change += (self.tmp[j] - self.agents[idx].opinion[j]).abs();
        }

        mem::swap(&mut self.agents[idx].opinion, &mut self.tmp);
    }

    pub fn sweep(&mut self) {
        for _ in 0..self.num_agents {
            self.step_naive();
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
                slice[(i.opinion[d]*100.) as usize] += 1;
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
