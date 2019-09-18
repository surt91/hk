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
struct HKAgentAC {
    opinion: f32,
    tolerance: f32,
    resources: f32,
    initial_opinion: f32,
}

impl HKAgentAC {
    fn new(opinion: f32, tolerance: f32, resources: f32) -> HKAgentAC {
        HKAgentAC {
            opinion,
            tolerance,
            resources,
            initial_opinion: opinion,
        }
    }
}

impl PartialEq for HKAgentAC {
    fn eq(&self, other: &HKAgentAC) -> bool {
        (self.opinion - other.opinion).abs() < EPS
            && (self.tolerance - other.tolerance).abs() < EPS
            && (self.resources - other.resources).abs() < EPS
            && (self.initial_opinion - other.initial_opinion).abs() < EPS
    }
}

pub struct HegselmannKrauseAC {
    num_agents: u32,
    agents: Vec<HKAgentAC>,

    opinion_set: BTreeMap<OrderedFloat<f32>, u32>,

    dynamic_density: Vec<Vec<u64>>,

    // we need many, good (but not crypto) random numbers
    // we will use here the pcg generator
    rng: Pcg64,
}

impl PartialEq for HegselmannKrauseAC {
    fn eq(&self, other: &HegselmannKrauseAC) -> bool {
        self.agents == other.agents
    }
}

impl fmt::Debug for HegselmannKrauseAC {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HK {{ N: {}, agents: {:?} }}", self.num_agents, self.agents)
    }
}

impl HegselmannKrauseAC {
    pub fn new(n: u32, min_tolerance: f32, max_tolerance: f32, start_resources: f32, seed: u64) -> HegselmannKrauseAC {
        let mut rng = Pcg64::seed_from_u64(seed);
        let stretch = |x: f32| x*(max_tolerance-min_tolerance)+min_tolerance;
        let agents: Vec<HKAgentAC> = (0..n).map(|_| HKAgentAC::new(
            rng.gen(),
            stretch(rng.gen()),
            start_resources,
        )).collect();

        // datastructure for `step_bisect`
        let mut opinion_set = BTreeMap::new();
        for i in agents.iter() {
            *opinion_set.entry(OrderedFloat(i.opinion)).or_insert(0) += 1;
        }
        assert!(opinion_set.iter().map(|(_, v)| v).sum::<u32>() == n);

        let dynamic_density = Vec::new();

        HegselmannKrauseAC {
            num_agents: n,
            agents,
            opinion_set,
            dynamic_density,
            rng,
        }
    }

    pub fn step_bisect(&mut self) {
        // get a random agent
        let idx = self.rng.gen_range(0, self.num_agents) as usize;
        let i = &self.agents[idx];

        let (sum, count) = self.opinion_set
            .range((Included(&OrderedFloat(i.opinion-i.tolerance)), Included(&OrderedFloat(i.opinion+i.tolerance))))
            .map(|(j, ctr)| (j.into_inner(), ctr))
            .fold((0., 0), |(sum, count), (j, ctr)| (sum + *ctr as f32 * j, count + ctr));

        let mut new_opinion = sum / count as f32;

        // pay a cost
        let mut new_resources = i.resources;
        new_resources -= (i.initial_opinion - new_opinion).abs();
        if new_resources < 0. {
            if i.initial_opinion > new_opinion {
                new_opinion -= new_resources;
            } else {
                new_opinion += new_resources;
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

        self.agents[idx].opinion = new_opinion;
        self.agents[idx].resources = new_resources;
    }

    pub fn sweep(&mut self) {
        for _ in 0..self.num_agents {
            self.step_bisect();
        }
        self.add_state_to_density();
    }

    fn add_state_to_density(&mut self) {
        let mut slice = vec![0; 100];
        for i in &self.agents {
            slice[(i.opinion*100.) as usize] += 1;
        }
        self.dynamic_density.push(slice);
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
