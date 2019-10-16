use std::collections::BTreeMap;
use std::ops::Bound::Included;
use std::fmt;
use std::fs::File;
use std::io::prelude::*;

use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use itertools::Itertools;

use ordered_float::OrderedFloat;

/// maximal time to save density information for
const THRESHOLD: usize = 4000;
const EPS: f32 = 1e-5;
const DENSITYBINS: usize = 100;

#[derive(Clone, Debug)]
pub struct HKAgent {
    pub opinion: f32,
    pub tolerance: f32,
    pub initial_opinion: f32,
}

impl HKAgent {
    fn new(opinion: f32, tolerance: f32) -> HKAgent {
        HKAgent {
            opinion,
            tolerance,
            initial_opinion: opinion
        }
    }
}

impl PartialEq for HKAgent {
    fn eq(&self, other: &HKAgent) -> bool {
        (self.opinion - other.opinion).abs() < EPS
            && (self.tolerance - other.tolerance).abs() < EPS
    }
}
//
// trait Agent {
//
// }
//
// trait UpdateSync {
//
// }
//
// trait UpdateSeq {
//
// }
//
// trait Model: UpdateSync + UpdateSeq {
//     fn sweep_seq(&mut self);
//     fn sweep_sync(&mut self);
//
//     fn new() -> Self;
//     fn reset(&mut self);
//
//     fn agents() -> impl Iterator<Item = HKAgent>;
//
//     fn list_clusters(&self);
//     fn cluster_sizes(&self);
//     fn density(&self);
// }

pub struct HegselmannKrause {
    pub num_agents: u32,
    pub agents: Vec<HKAgent>,
    pub time: usize,
    min_tolerance: f32,
    max_tolerance: f32,

    pub eta: f32,

    pub opinion_set: BTreeMap<OrderedFloat<f32>, u32>,
    pub acc_change: f32,
    dynamic_density: Vec<Vec<u64>>,

    pub ji: Vec<f32>,
    pub jin: Vec<i32>,

    density_slice: Vec<u64>,
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
    pub fn new(n: u32, min_tolerance: f32, max_tolerance: f32, eta: f32, seed: u64) -> HegselmannKrause {
        let rng = Pcg64::seed_from_u64(seed);
        let agents: Vec<HKAgent> = Vec::new();

        // datastructure for `step_bisect`
        let opinion_set = BTreeMap::new();

        let dynamic_density = Vec::new();

        let mut hk = HegselmannKrause {
            num_agents: n,
            agents,
            time: 0,
            min_tolerance,
            max_tolerance,
            eta,
            opinion_set,
            acc_change: 0.,
            dynamic_density,
            ji: Vec::new(),
            jin: Vec::new(),
            density_slice: vec![0; DENSITYBINS],
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

        self.time = 0;
    }

    pub fn update_entry(&mut self, old: f32, new: f32) {
        // often, nothing changes -> optimize for this converged case
        if old == new {
            return
        }

        *self.opinion_set.entry(OrderedFloat(old)).or_insert_with(|| panic!("todo")) -= 1;
        if self.opinion_set[&OrderedFloat(old)] == 0 {
            self.opinion_set.remove(&OrderedFloat(old));
        }
        *self.opinion_set.entry(OrderedFloat(new)).or_insert(0) += 1;
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


        self.acc_change += (i.opinion - new_opinion).abs();
        self.update_entry(i.opinion, new_opinion);
        self.agents[idx].opinion = new_opinion;
    }

    pub fn sweep(&mut self) {
        for _ in 0..self.num_agents {
            // self.step_naive();
            self.step_bisect();
        }
        self.add_state_to_density();
        self.time += 1;
    }

    fn sync_new_opinions_naive(&self) -> (Vec<f32>, f32) {
        let mut acc_change = 0.;
        let op = self.agents.iter().map(|i| {
            let mut tmp = 0.;
            let mut count = 0;
            for j in self.agents.iter()
                    .filter(|j| (i.opinion - j.opinion).abs() < i.tolerance) {
                tmp += j.opinion;
                count += 1;
            }

            tmp /= count as f32;
            acc_change += (tmp - i.opinion).abs();
            tmp
        }).collect();
        (op, acc_change)
    }

    pub fn sweep_synchronous_naive(&mut self) {
        let (new_opinions, acc_change) = self.sync_new_opinions_naive();
        self.acc_change += acc_change;
        for i in 0..self.num_agents as usize {
            self.agents[i].opinion = new_opinions[i];
        }
        self.add_state_to_density()
    }

    fn sync_new_opinions_bisect(&self) -> (Vec<f32>, f32) {
        let mut acc_change = 0.;
        let op = self.agents.clone().iter().map(|i| {
            let (sum, count) = self.opinion_set
                .range((Included(&OrderedFloat(i.opinion-i.tolerance)), Included(&OrderedFloat(i.opinion+i.tolerance))))
                .map(|(j, ctr)| (j.into_inner(), ctr))
                .fold((0., 0), |(sum, count), (j, ctr)| (sum + *ctr as f32 * j, count + ctr));

            let new_opinion = sum / count as f32;
            acc_change += (new_opinion - i.opinion).abs();
            new_opinion
        }).collect();

        (op, acc_change)
    }

    pub fn sweep_synchronous_bisect(&mut self) {
        let (new_opinions, acc_change) = self.sync_new_opinions_bisect();
        self.acc_change += acc_change;

        for i in 0..self.num_agents as usize {
            // often, nothing changes -> optimize for this converged case
            let old = self.agents[i].opinion;
            self.update_entry(old, new_opinions[i]);

            self.agents[i].opinion = new_opinions[i];
        }
        self.add_state_to_density()
    }

    pub fn sweep_synchronous(&mut self) {
        self.sweep_synchronous_bisect();
        self.time += 1;
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

    pub fn add_state_to_density(&mut self) {
        if self.time > THRESHOLD {
            return
        }

        for i in 0..DENSITYBINS {
            self.density_slice[i] = 0;
        }

        for i in &self.agents {
            self.density_slice[(i.opinion*DENSITYBINS as f32) as usize] += 1;
        }
        if self.dynamic_density.len() <= self.time {
            self.dynamic_density.push(self.density_slice.clone());
        } else {
            for i in 0..DENSITYBINS {
                self.dynamic_density[self.time][i] += self.density_slice[i];
            }
        }
    }

    pub fn fill_density(&mut self) {
        let mut j = self.time;
        while j < THRESHOLD {
            if self.dynamic_density.len() <= j {
                self.dynamic_density.push(self.density_slice.clone());
            } else {
                for i in 0..DENSITYBINS {
                    self.dynamic_density[j][i] += self.density_slice[i];
                }
            }
            j += 1;
        }
    }

    pub fn write_density(&self, file: &mut File) -> std::io::Result<()> {
        let string_list = self.dynamic_density.iter()
            .map(|x| x.iter().join(" "))
            .join("\n");
        write!(file, "{}\n", string_list)
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
