use std::collections::BTreeMap;
use std::ops::Bound::Included;
use std::fmt;
use std::fs::File;
use std::io::prelude::*;

use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use itertools::Itertools;

use ordered_float::OrderedFloat;

const EPS: f32 = 1e-6;

#[derive(Clone, Debug)]
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

impl PartialEq for HKAgent {
    fn eq(&self, other: &HKAgent) -> bool {
        (self.opinion - other.opinion).abs() < EPS
            && (self.tolerance - other.tolerance).abs() < EPS
    }
}

pub struct HegselmannKrause {
    num_agents: u32,
    agents: Vec<HKAgent>,
    min_tolerance: f32,
    max_tolerance: f32,

    opinion_set: BTreeMap<OrderedFloat<f32>, u32>,
    pub acc_change: f32,
    // we need many, good (but not crypto) random numbers
    // we will use here the pcg generator
    rng: Pcg64,
}

impl PartialEq for HegselmannKrause {
    fn eq(&self, other: &HegselmannKrause) -> bool {
        self.agents == other.agents
    }
}

impl fmt::Debug for HegselmannKrause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HK {{ N: {}, agents: {:?} }}", self.num_agents, self.agents)
    }
}

impl HegselmannKrause {
    pub fn new(n: u32, min_tolerance: f32, max_tolerance: f32, seed: u64) -> HegselmannKrause {
        let rng = Pcg64::seed_from_u64(seed);
        let agents: Vec<HKAgent> = Vec::new();

        // datastructure for `step_bisect`
        let opinion_set = BTreeMap::new();

        let mut hk = HegselmannKrause {
            num_agents: n,
            agents,
            min_tolerance,
            max_tolerance,
            opinion_set,
            acc_change: 0.,
            rng,
        };

        hk.reset();
        hk
    }

    fn stretch(x: f32, low: f32, high: f32) -> f32 {
        x*(high-low)+low
    }

    pub fn reset(&mut self) {
        self.agents = (0..self.num_agents).map(|_| HKAgent::new(
            self.rng.gen(),
            HegselmannKrause::stretch(self.rng.gen(), self.min_tolerance, self.max_tolerance)
        )).collect();

        self.opinion_set.clear();
        for i in self.agents.iter() {
            *self.opinion_set.entry(OrderedFloat(i.opinion)).or_insert(0) += 1;
        }
        assert!(self.opinion_set.iter().map(|(_, v)| v).sum::<u32>() == self.num_agents);
    }

    pub fn step_naive(&mut self) {
        // get a random agent
        let idx = self.rng.gen_range(0, self.num_agents) as usize;
        let i = &self.agents[idx];

        let (sum, count) = self.agents.iter()
            .map(|j| j.opinion)
            .filter(|j| (i.opinion - j).abs() < i.tolerance)
            .fold((0., 0), |(sum, count), i| (sum + i, count + 1));

        self.agents[idx].opinion = sum / count as f32;
    }

    pub fn step_bisect(&mut self) {
        // get a random agent
        let idx = self.rng.gen_range(0, self.num_agents) as usize;
        let i = &self.agents[idx];

        let (sum, count) = self.opinion_set
            .range((Included(&OrderedFloat(i.opinion-i.tolerance)), Included(&OrderedFloat(i.opinion+i.tolerance))))
            .map(|(j, ctr)| (j.into_inner(), ctr))
            .fold((0., 0), |(sum, count), (j, ctr)| (sum + *ctr as f32 * j, count + ctr));

        let new_opinion = sum / count as f32;

        // often, nothing changes -> optimize for this converged case
        if i.opinion == new_opinion {
            return
        }

        *self.opinion_set.entry(OrderedFloat(i.opinion)).or_insert_with(|| panic!("todo")) -= 1;
        if self.opinion_set[&OrderedFloat(i.opinion)] == 0 {
            self.opinion_set.remove(&OrderedFloat(i.opinion));
        }
        *self.opinion_set.entry(OrderedFloat(new_opinion)).or_insert(0) += 1;

        self.acc_change += (i.opinion - new_opinion).abs();
        self.agents[idx].opinion = new_opinion;
    }

    pub fn sweep(&mut self) {
        for _ in 0..self.num_agents {
            // self.step_naive();
            self.step_bisect();
        }
    }

    /// A cluster are agents whose distance is less than EPS
    fn list_clusters(&self) -> Vec<Vec<HKAgent>> {
        let mut clusters: Vec<Vec<HKAgent>> = Vec::new();
        'agent: for i in &self.agents {
            for c in &mut clusters {
                if (i.opinion - &c[0].opinion).abs() < EPS {
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

    pub fn write_cluster_sizes(&self, file: &mut File) -> std::io::Result<()> {
        let clusters = self.list_clusters();

        let string_list = clusters.iter()
            .map(|c| c[0].opinion)
            .join(" ");
        write!(file, "# {}\n", string_list)?;

        let string_list = clusters.iter()
            .map(|c| c.len().to_string())
            .join(" ");
        write!(file, "{}\n", string_list)?;
        Ok(())
    }

    pub fn write_state(&self, file: &mut File) -> std::io::Result<()> {
        let string_list = self.agents.iter()
            .map(|j| j.opinion.to_string())
            .join(" ");
        write!(file, "{}\n", string_list)
    }

    pub fn write_gp(&self, file: &mut File, outfilename: &str) -> std::io::Result<()> {
        write!(file, "set terminal pngcairo\n")?;
        write!(file, "set output '{}.png'\n", outfilename)?;
        write!(file, "set xl 't'\n")?;
        write!(file, "set yl 'x_i'\n")?;
        write!(file, "p '{}' u 0:1 w l not, ", outfilename)?;

        let string_list = (2..self.num_agents)
            .map(|j| format!("'' u 0:{} w l not,", j))
            .join(" ");
        write!(file, "{}", string_list)
    }
}
