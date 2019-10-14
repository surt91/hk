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
struct HKAgentCost {
    opinion: f32,
    tolerance: f32,
    resources: f32,
    initial_opinion: f32,
}

impl HKAgentCost {
    fn new(opinion: f32, tolerance: f32, resources: f32) -> HKAgentCost {
        HKAgentCost {
            opinion,
            tolerance,
            resources,
            initial_opinion: opinion,
        }
    }
}

impl PartialEq for HKAgentCost {
    fn eq(&self, other: &HKAgentCost) -> bool {
        (self.opinion - other.opinion).abs() < EPS
            && (self.tolerance - other.tolerance).abs() < EPS
    }
}

pub struct HegselmannKrauseCost {
    num_agents: u32,
    agents: Vec<HKAgentCost>,
    time: usize,
    min_tolerance: f32,
    max_tolerance: f32,
    eta: f32,
    min_resources: f32,
    max_resources: f32,

    opinion_set: BTreeMap<OrderedFloat<f32>, u32>,
    pub acc_change: f32,
    dynamic_density: Vec<Vec<u64>>,

    density_slice: Vec<u64>,
    // we need many, good (but not crypto) random numbers
    // we will use here the pcg generator
    rng: Pcg64,
}

impl PartialEq for HegselmannKrauseCost {
    fn eq(&self, other: &HegselmannKrauseCost) -> bool {
        self.agents == other.agents
    }
}

impl fmt::Debug for HegselmannKrauseCost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HK {{ N: {}, agents: {:?} }}", self.num_agents, self.agents)
    }
}

impl HegselmannKrauseCost {
    pub fn new(
            n: u32,
            min_tolerance: f32,
            max_tolerance: f32,
            eta: f32,
            min_resources: f32,
            max_resources: f32,
            seed: u64
    ) -> HegselmannKrauseCost {
        let rng = Pcg64::seed_from_u64(seed);
        let agents: Vec<HKAgentCost> = Vec::new();

        // datastructure for `step_bisect`
        let opinion_set = BTreeMap::new();

        let dynamic_density = Vec::new();

        let mut hk = HegselmannKrauseCost {
            num_agents: n,
            agents,
            time: 0,
            min_tolerance,
            max_tolerance,
            eta,
            min_resources,
            max_resources,
            opinion_set,
            acc_change: 0.,
            dynamic_density,
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
        self.agents = (0..self.num_agents).map(|_| HKAgentCost::new(
            self.rng.gen(),
            HegselmannKrauseCost::stretch(self.rng.gen(), self.min_tolerance, self.max_tolerance),
            HegselmannKrauseCost::stretch(self.rng.gen(), self.min_resources, self.max_resources)
        )).collect();

        self.opinion_set.clear();
        for i in self.agents.iter() {
            *self.opinion_set.entry(OrderedFloat(i.opinion)).or_insert(0) += 1;
        }
        assert!(self.opinion_set.iter().map(|(_, v)| v).sum::<u32>() == self.num_agents);

        self.time = 0;
    }

    pub fn step_bisect(&mut self) {
        // get a random agent
        let idx = self.rng.gen_range(0, self.num_agents) as usize;
        let i = &self.agents[idx];

        let (sum, count) = self.opinion_set
            .range((Included(&OrderedFloat(i.opinion-i.tolerance)), Included(&OrderedFloat(i.opinion+i.tolerance))))
            .map(|(j, ctr)| (j.into_inner(), ctr))
            .fold((0., 0), |(sum, count), (j, ctr)| (sum + *ctr as f32 * j, count + ctr));

        let mut new_opinion = (1.-self.eta) * sum / count as f32 + self.eta*i.opinion;

        if idx == 3 {
            println!("{:?} -> {:?}", i.opinion, new_opinion);
        }

        // pay a cost
        let mut new_resources = i.resources;
        new_resources -= self.eta * (i.opinion - new_opinion).abs();
        if new_resources < 0. {
            if i.opinion > new_opinion {
                new_opinion -= new_resources / self.eta;
            } else {
                new_opinion += new_resources / self.eta;
            }
            new_resources = 0.;
        }

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
        self.agents[idx].resources = new_resources;
    }

    pub fn sweep(&mut self) {
        for _ in 0..self.num_agents {
            // self.step_naive();
            self.step_bisect();
        }
        self.add_state_to_density();
        self.time += 1;
    }

    fn sync_new_opinions_bisect(&self) -> (Vec<(f32, f32)>, f32) {
        let mut acc_change = 0.;
        let op = self.agents.clone().iter().map(|i| {
            let (sum, count) = self.opinion_set
                .range((Included(&OrderedFloat(i.opinion-i.tolerance)), Included(&OrderedFloat(i.opinion+i.tolerance))))
                .map(|(j, ctr)| (j.into_inner(), ctr))
                .fold((0., 0), |(sum, count), (j, ctr)| (sum + *ctr as f32 * j, count + ctr));

            let mut new_opinion = (1.-self.eta) * sum / count as f32 + self.eta*i.opinion;

            // pay a cost
            let mut new_resources = i.resources;
            new_resources -= self.eta * (i.opinion - new_opinion).abs();
            if new_resources < 0. {
                if i.opinion > new_opinion {
                    new_opinion -= new_resources / self.eta;
                } else {
                    new_opinion += new_resources / self.eta;
                }
                new_resources = 0.;
            }

            acc_change += (new_opinion - i.opinion).abs();
            (new_opinion, new_resources)
        }).collect();

        (op, acc_change)
    }

    pub fn sweep_synchronous_bisect(&mut self) {
        let (new_opinions_and_resources, acc_change) = self.sync_new_opinions_bisect();
        self.acc_change += acc_change;
        let new_opinions: Vec<f32> = new_opinions_and_resources.iter().map(|(x, _)| *x).collect();
        let new_resources: Vec<f32> = new_opinions_and_resources.iter().map(|(_, x)| *x).collect();

        for i in 0..self.num_agents as usize {
            // often, nothing changes -> optimize for this converged case
            let old = self.agents[i].opinion;
            if self.agents[i].opinion != new_opinions[i] {
                *self.opinion_set.entry(OrderedFloat(self.agents[i].opinion)).or_insert_with(|| panic!(format!("The old opinion is not in the tree: This should never happen! {}", old))) -= 1;
                if self.opinion_set[&OrderedFloat(self.agents[i].opinion)] == 0 {
                    self.opinion_set.remove(&OrderedFloat(self.agents[i].opinion));
                }
                *self.opinion_set.entry(OrderedFloat(new_opinions[i])).or_insert(0) += 1;

                // if i == 3 {
                //     println!("r {:?} -> {:?}", self.agents[i].resources, new_resources[i]);
                //     println!("o {:?} -> {:?}", self.agents[i].opinion, new_opinions[i]);
                // }

                self.agents[i].opinion = new_opinions[i];
                self.agents[i].resources = new_resources[i];
            }
        }
        self.add_state_to_density()
    }

    pub fn sweep_synchronous(&mut self) {
        self.sweep_synchronous_bisect();
        self.time += 1;
    }

    /// A cluster are agents whose distance is less than EPS
    fn list_clusters(&self) -> Vec<Vec<HKAgentCost>> {
        let mut clusters: Vec<Vec<HKAgentCost>> = Vec::new();
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

    fn add_state_to_density(&mut self) {
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
}
