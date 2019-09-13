use std::collections::BTreeMap;
use std::ops::Bound::Included;
use std::fmt;
use std::fs::File;
use std::io::prelude::*;

use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use itertools::Itertools;

use ordered_float::OrderedFloat;

const EPS: f32 = 1e-4;

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
    opinion_set: BTreeMap<OrderedFloat<f32>, u32>,
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
    pub fn new(n: u32, seed: u64) -> HegselmannKrause {
        let mut rng = Pcg64::seed_from_u64(seed);
        let agents: Vec<HKAgent> = (0..n).map(|_| HKAgent::new(rng.gen(), rng.gen())).collect();

        let mut opinion_set = BTreeMap::new();
        for i in agents.iter() {
            opinion_set.insert(OrderedFloat(i.opinion), 1);
        }
        assert!(opinion_set.len() == n as usize);

        HegselmannKrause {
            num_agents: n,
            agents,
            opinion_set,
            rng
        }
    }

    pub fn step_naive(&mut self) {
        // get a random agent
        let idx = self.rng.gen_range(0, self.num_agents) as usize;
        let i = &self.agents[idx];

        let (sum, count) = self.agents.iter()
            .map(|j| j.opinion)
            .filter(|j| (i.opinion - j).abs() < i.tolerance)
            .fold((0f32, 0u32), |(sum, count), i| (sum + i, count + 1));

        self.agents[idx].opinion = sum / count as f32;
    }

    // this is very slow due to copying
    pub fn step_bisect(&mut self) {
        // get a random agent
        let idx = self.rng.gen_range(0, self.num_agents) as usize;
        let i = &self.agents[idx];

        let (sum, count) = self.opinion_set
            .range((Included(&OrderedFloat(i.opinion-i.tolerance)), Included(&OrderedFloat(i.opinion+i.tolerance))))
            .map(|(j, ctr)| (j.into_inner(), ctr))
            .fold((0f32, 0u32), |(sum, count), (j, ctr)| (sum + *ctr as f32 * j, count + ctr));

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

        self.agents[idx].opinion = new_opinion;
    }

    pub fn sweep(&mut self) {
        for _ in 0..self.num_agents {
            self.step_bisect();
            // self.step_naive();
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
