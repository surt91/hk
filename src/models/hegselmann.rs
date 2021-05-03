// we use float comparision to test if an entry did change during an iteration for performance
// false positives do not lead to wrong results
#![allow(clippy::float_cmp)]

use std::collections::BTreeMap;
use std::ops::Bound::Included;
use std::fmt;

use rand::Rng;

use ordered_float::OrderedFloat;

use petgraph::graph::NodeIndex;

use super::{CostModel, ResourceModel, PopulationModel, TopologyModel, TopologyRealization, Agent};
use super::ABM;
use super::ABMBuilder;
use super::abm::ABMinternals;

use largedev::{MarkovChain, Model};


impl ABMBuilder {
    pub fn hk(&mut self, mut rng: &mut impl Rng) -> HegselmannKrause {
        let agents: Vec<Agent> = Vec::new();

        // datastructure for `step_bisect`
        let opinion_set = BTreeMap::new();

        let mut hk = HegselmannKrause {
            num_agents: self.num_agents,
            agents: agents.clone(),
            time: 0,
            topology: TopologyRealization::None,
            cost_model: self.cost_model.clone(),
            resource_model: self.resource_model.clone(),
            population_model: self.population_model.clone(),
            topology_model: self.topology_model.clone(),
            opinion_set,
            undo_idx: 0,
            undo_val: 0.,
            agents_initial: agents,
            abm_internals: ABMinternals::new(),
        };

        hk.reset(&mut rng);
        hk
    }
}

#[derive(Clone)]
pub struct HegselmannKrause {
    pub num_agents: u32,
    pub agents: Vec<Agent>,
    pub time: usize,

    /// topology of the possible interaction between agents
    /// None means fully connected
    topology: TopologyRealization,

    pub cost_model: CostModel,
    resource_model: ResourceModel,
    population_model: PopulationModel,
    topology_model: TopologyModel,

    pub opinion_set: BTreeMap<OrderedFloat<f32>, u32>,

    // for Markov chains
    undo_idx: usize,
    undo_val: f32,
    pub agents_initial: Vec<Agent>,

    abm_internals: ABMinternals,
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

impl ABM for HegselmannKrause {
    fn sweep(&mut self, mut _rng: &mut impl Rng) {
        self.sweep_synchronous()
    }

    fn reset(&mut self, mut rng: &mut impl Rng) {
        self.agents = (0..self.num_agents).map(|_| {
            let xi = self.gen_init_opinion(&mut rng);
            let ei = self.gen_init_tolerance(&mut rng);
            let ci = self.gen_init_resources(ei, &mut rng);
            Agent::new(
                xi,
                ei,
                ci,
            )
        }).collect();

        self.agents_initial = self.agents.clone();

        self.topology = self.gen_init_topology(&mut rng);

        self.prepare_opinion_set();
        self.time = 0;
    }

    fn get_mut_abm_internals(&mut self) -> &mut ABMinternals {
        &mut self.abm_internals
    }

    fn get_abm_internals(&self) -> &ABMinternals {
        &self.abm_internals
    }

    fn get_population_model(&self) -> PopulationModel {
        self.population_model.clone()
    }

    fn get_topology_model(&self) -> TopologyModel {
        self.topology_model.clone()
    }

    fn get_topology(&self) -> &TopologyRealization {
        &self.topology
    }

    fn get_resource_model(&self) -> ResourceModel {
        self.resource_model.clone()
    }

    fn get_agents(&self) -> &Vec<Agent> {
        &self.agents
    }

    fn get_time(&self) -> usize {
        self.time
    }
}

impl HegselmannKrause {
    pub fn prepare_opinion_set(&mut self) {
        self.opinion_set.clear();
        for i in self.agents.iter() {
            *self.opinion_set.entry(OrderedFloat(i.opinion)).or_insert(0) += 1;
        }
            assert!(self.opinion_set.iter().map(|(_, v)| v).sum::<u32>() == self.num_agents);
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

    pub fn pay(&mut self, idx: usize, mut new_opinion: f32) -> (f32, f32)  {
        let i = &mut self.agents[idx];
        let mut new_resources = i.resources;
        match self.cost_model {
            CostModel::Free => {},
            CostModel::Rebounce => {
                new_resources -= (i.initial_opinion - new_opinion).abs();
                if new_resources < 0. {
                    if i.initial_opinion > new_opinion {
                        new_opinion -= new_resources;
                    } else {
                        new_opinion += new_resources;
                    }
                    new_resources = 0.;
                }
            }
            CostModel::Change(eta) => {
                new_resources -= eta * (i.opinion - new_opinion).abs();
                if new_resources < 0. {
                    if i.opinion > new_opinion {
                        new_opinion -= new_resources / eta;
                    } else {
                        new_opinion += new_resources / eta;
                    }
                    new_resources = 0.;
                }
            }
            CostModel::Annealing(_eta) => {
                panic!("CostModel::Annealing may not be used with the deterministic dynamics!")
            }
        }
        (new_opinion, new_resources)
    }

    pub fn step_naive(&mut self, rng: &mut impl Rng) {
        // get a random agent
        let idx = rng.gen_range(0, self.num_agents) as usize;
        let i = &self.agents[idx];

        let (sum, count) = match &self.topology {
            TopologyRealization::None =>
                self.agents.iter()
                    .map(|j| j.opinion)
                    .filter(|j| (i.opinion - j).abs() < i.tolerance)
                    .fold((0., 0), |(sum, count), i| (sum + i, count + 1)),
            TopologyRealization::Graph(g) => {
                let nodes: Vec<NodeIndex<u32>> = g.node_indices().collect();
                g.neighbors(nodes[idx])
                    .chain(std::iter::once(nodes[idx]))
                    .map(|j| self.agents[g[j]].opinion)
                    .filter(|j| (i.opinion - j).abs() < i.tolerance)
                    .fold((0., 0), |(sum, count), i| (sum + i, count + 1))
            }
            TopologyRealization::Hypergraph(_) => unimplemented!(),
        };

        let new_opinion = sum / count as f32;
        let old = i.opinion;
        let (new_opinion, new_resources) = self.pay(idx, new_opinion);

        self.acc_change((old - new_opinion).abs());

        self.agents[idx].opinion = new_opinion;
        self.agents[idx].resources = new_resources;

    }

    pub fn step_bisect(&mut self, rng: &mut impl Rng) {
        // get a random agent
        let idx = rng.gen_range(0, self.num_agents) as usize;
        let i = &self.agents[idx];

        if self.topology_model != TopologyModel::FullyConnected {
            panic!("The tree update does only work for fully connected topologies (though it could be extended)");
        }

        let (sum, count) = self.opinion_set
            .range((Included(&OrderedFloat(i.opinion-i.tolerance)), Included(&OrderedFloat(i.opinion+i.tolerance))))
            .map(|(j, ctr)| (j.into_inner(), ctr))
            .fold((0., 0), |(sum, count), (j, ctr)| (sum + *ctr as f32 * j, count + ctr));

        let new_opinion = sum / count as f32;

        let old = i.opinion;
        let (new_opinion, new_resources) = self.pay(idx, new_opinion);

        self.acc_change((old - new_opinion).abs());
        self.update_entry(old, new_opinion);
        self.agents[idx].opinion = new_opinion;
        self.agents[idx].resources = new_resources
    }

    fn sync_new_opinions_naive(&self) -> Vec<f32> {
        self.agents.iter().enumerate().map(|(idx, i)| {
            let mut tmp = 0.;
            let mut count = 0;

            match &self.topology {
                TopologyRealization::None => {
                    for j in self.agents.iter()
                        .filter(|j| (i.opinion - j.opinion).abs() < i.tolerance) {
                            tmp += j.opinion;
                            count += 1;
                        }
                },
                TopologyRealization::Graph(g) => {
                    // a node index is actually exactly the same as our index, since we do not change
                    // the graph after construction.
                    let node = NodeIndex::new(idx);
                    for j in g.neighbors(node).chain(std::iter::once(node))
                        .filter(|j| (i.opinion - self.agents[g[*j]].opinion).abs() < i.tolerance) {
                            tmp += self.agents[g[j]].opinion;
                            count += 1;
                        }
                }
                TopologyRealization::Hypergraph(h) => {
                    // maybe calculate contribution of every edge behforhand and get it from a
                    // vec here
                    let g = &h.factor_graph;
                    for e in g.neighbors(h.node_nodes[idx]) {
                        let it = g.neighbors(e).map(|n| OrderedFloat(self.agents[*g.node_weight(n).unwrap()].opinion));
                        let min = it.clone().min().unwrap().into_inner();
                        let max = it.clone().max().unwrap().into_inner();
                        let mintol = g.neighbors(e).map(|n| OrderedFloat(self.agents[*g.node_weight(n).unwrap()].tolerance)).min().unwrap().into_inner();
                        let sum: f32 = g.neighbors(e).map(|n| self.agents[*g.node_weight(n).unwrap()].opinion).sum();
                        let len = g.neighbors(e).count();

                        // if all nodes of the hyperedge are pairwise compatible
                        // `i` takes the average opinion of this hyperedge into consideration
                        if max - min < mintol {
                            tmp += sum / len as f32;
                            count += 1;
                        }
                    }
                    // if there are no compatible edges, change nothing (and avoid dividing by zero)
                    if count == 0 {
                        count = 1;
                        tmp = i.opinion;
                    }
                },
            };

            tmp /= count as f32;
            tmp
        }).collect()
    }

    pub fn sweep_synchronous_naive(&mut self) {
        let new_opinions = self.sync_new_opinions_naive();

        for i in 0..self.num_agents as usize {
            let (new_opinion, new_resources) = self.pay(i, new_opinions[i]);

            self.acc_change((self.agents[i].opinion - new_opinion).abs());

            self.agents[i].opinion = new_opinion;
            self.agents[i].resources = new_resources
        }
        self.add_state_to_density()
    }

    fn sync_new_opinions_bisect(&self) -> Vec<f32> {
        if self.topology_model != TopologyModel::FullyConnected {
            panic!("The tree update does only work for fully connected topologies (though it could be extended)")
        }

        self.agents.clone().iter().map(|i| {
            let (sum, count) = self.opinion_set
                .range((Included(&OrderedFloat(i.opinion-i.tolerance)), Included(&OrderedFloat(i.opinion+i.tolerance))))
                .map(|(j, ctr)| (j.into_inner(), ctr))
                .fold((0., 0), |(sum, count), (j, ctr)| (sum + *ctr as f32 * j, count + ctr));

            let new_opinion = sum / count as f32;
            new_opinion
        }).collect()
    }

    pub fn sweep_synchronous_bisect(&mut self) {
        let new_opinions = self.sync_new_opinions_bisect();

        for i in 0..self.num_agents as usize {
            // often, nothing changes -> optimize for this converged case
            let old = self.agents[i].opinion;
            let (new_opinion, new_resources) = self.pay(i, new_opinions[i]);
            self.update_entry(old, new_opinion);

            self.acc_change((self.agents[i].opinion - new_opinion).abs());

            self.agents[i].opinion = new_opinion;
            self.agents[i].resources = new_resources
        }
        self.add_state_to_density()
    }

    pub fn sweep_synchronous(&mut self) {
        match self.topology_model {
            // For topologies with few connections, use `step_naive`, otherwise the `step_bisect`
            TopologyModel::FullyConnected => self.sweep_synchronous_bisect(),
            _ => self.sweep_synchronous_naive(),
        }
        self.time += 1;
    }

    pub fn sweep_async(&mut self, mut rng: &mut impl Rng) {
        for _ in 0..self.num_agents {
            match self.topology_model {
                // For topologies with few connections, use `step_naive`, otherwise the `step_bisect`
                TopologyModel::FullyConnected => self.step_bisect(&mut rng),
                _ => self.step_naive(&mut rng),
            }
        }
        self.add_state_to_density();
        self.time += 1;
    }
}

impl Model for HegselmannKrause {
    fn value(&self) -> f64 {
        self.cluster_max() as f64 / self.num_agents as f64
    }
}

impl MarkovChain for HegselmannKrause {
    fn change(&mut self, mut rng: &mut impl Rng) {
        self.undo_idx = rng.gen_range(0, self.agents.len());
        let val: f32 = rng.gen();
        self.undo_val = self.agents_initial[self.undo_idx].initial_opinion;

        self.agents_initial[self.undo_idx].opinion = val;
        self.agents_initial[self.undo_idx].initial_opinion = val;

        self.agents = self.agents_initial.clone();
        self.prepare_opinion_set();

        self.relax(&mut rng);
    }

    fn undo(&mut self) {
        self.agents_initial[self.undo_idx].initial_opinion = self.undo_val;
        self.agents_initial[self.undo_idx].opinion = self.undo_val;
    }
}