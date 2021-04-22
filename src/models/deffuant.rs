// we use float comparision to test if an entry did change during an iteration for performance
// false positives do not lead to wrong results
#![allow(clippy::float_cmp)]

use std::fmt;

use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use rand::seq::IteratorRandom;
use ordered_float::OrderedFloat;

use super::{PopulationModel, TopologyModel, TopologyRealization, ResourceModel, Agent};
use super::{ABM, ABMBuilder};
use super::abm::ABMinternals;

use petgraph::graph::NodeIndex;

use largedev::{MarkovChain, Model};

impl ABMBuilder {
    pub fn dw(&self) -> Deffuant {
        let rng = Pcg64::seed_from_u64(self.seed);
        let agents: Vec<Agent> = Vec::new();

        let mut dw = Deffuant {
            num_agents: self.num_agents,
            agents: agents.clone(),
            time: 0,
            mu: 1.,
            topology: TopologyRealization::None,
            population_model: self.population_model.clone(),
            topology_model: self.topology_model.clone(),
            abm_internals: ABMinternals::new(),
            rng,
            undo_idx: 0,
            undo_val: 0.,
            rn: Vec::new(),
            ctr: 0,
            agents_initial: agents,
        };

        dw.reset();
        dw
    }
}

#[derive(Clone)]
pub struct Deffuant {
    pub num_agents: u32,
    pub agents: Vec<Agent>,
    pub time: usize,

    // weight of the agent itself
    mu: f64,

    /// topology of the possible interaction between agents
    /// None means fully connected
    topology: TopologyRealization,

    population_model: PopulationModel,
    topology_model: TopologyModel,

    abm_internals: ABMinternals,

    // we need many, good (but not crypto) random numbers
    // we will use here the pcg generator
    rng: Pcg64,

    // for Markov chains
    undo_idx: usize,
    undo_val: f64,
    rn: Vec<f64>,
    ctr: usize,
    pub agents_initial: Vec<Agent>,
}

impl PartialEq for Deffuant {
    fn eq(&self, other: &Deffuant) -> bool {
        self.agents == other.agents
    }
}

impl fmt::Debug for Deffuant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DW {{ N: {}, agents: {:?} }}", self.num_agents, self.agents)
    }
}

impl ABM for Deffuant {
    fn sweep(&mut self) {
        for _ in 0..self.num_agents {
            self.step_naive()
        }
        self.add_state_to_density();
        self.time += 1;
    }

    fn reset(&mut self) {
        self.agents = (0..self.num_agents).map(|_| {
            let xi = self.gen_init_opinion();
            let ei = self.gen_init_tolerance();
            Agent::new(
                xi,
                ei,
                0.,
            )
        }).collect();

        self.agents_initial = self.agents.clone();

        self.topology = self.gen_init_topology();

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
        ResourceModel::None
    }

    fn get_agents(&self) -> &Vec<Agent> {
        &self.agents
    }

    fn get_time(&self) -> usize {
        self.time
    }

    fn get_rng(&mut self) -> &mut Pcg64 {
        &mut self.rng
    }
}

impl Deffuant {
    pub fn step_naive(&mut self) {
        let old_opinion;
        let new_opinion = match &self.topology {
            TopologyRealization::None => {
                if self.ctr+2 >= self.rn.len() {
                    for _ in 0..std::cmp::max(self.rn.len(), 100) {
                        self.rn.push(self.rng.gen())
                    }
                }

                // get a random agent
                let idx = (self.rn[self.ctr] * self.num_agents as f64) as usize;
                self.ctr += 1;
                let i = &self.agents[idx];
                old_opinion = i.opinion;

                let mut tmp = (self.rn[self.ctr] * (self.num_agents-1) as f64) as usize;
                self.ctr += 1;
                if tmp == idx {
                    tmp = self.num_agents as usize - 1 ;
                }
                let idx2 = tmp;

                if (i.opinion - self.agents[idx2].opinion).abs() < i.tolerance {
                    let new_opinion = (i.opinion + self.agents[idx2].opinion) / 2.;
                    // change the opinion of both endpoints
                    self.agents[idx].opinion = new_opinion;
                    self.agents[idx2].opinion = new_opinion;
                    new_opinion
                } else {
                    old_opinion
                }
            }
            TopologyRealization::Graph(g) => {
                // get a random agent
                let idx = self.rng.gen_range(0, self.num_agents) as usize;
                let i = &self.agents[idx];
                old_opinion = i.opinion;

                let nodes: Vec<NodeIndex<u32>> = g.node_indices().collect();
                let j = g.neighbors(nodes[idx])
                    .choose(&mut self.rng);
                if let Some(idx2) = j {
                    if (i.opinion - self.agents[g[idx2]].opinion).abs() < i.tolerance {
                        let new_opinion = (i.opinion + self.agents[g[idx2]].opinion) / 2.;
                        // change the opinion of both endpoints
                        self.agents[idx].opinion = new_opinion;
                        self.agents[g[idx2]].opinion = new_opinion;
                        new_opinion
                    } else {
                        old_opinion
                    }
                } else {
                    old_opinion
                }
            }
            TopologyRealization::Hypergraph(h) => {
                // get a random hyperdege
                let eidx = self.rng.gen_range(0, h.edge_nodes.len()) as usize;
                let e = h.edge_nodes[eidx];

                let g = &h.factor_graph;

                let it = g.neighbors(e).map(|n| OrderedFloat(self.agents[*g.node_weight(n).unwrap()].opinion));
                let min = it.clone().min().unwrap().into_inner();
                let max = it.clone().max().unwrap().into_inner();
                let mintol = g.neighbors(e).map(|n| OrderedFloat(self.agents[*g.node_weight(n).unwrap()].tolerance)).min().unwrap().into_inner();
                let sum: f32 = g.neighbors(e).map(|n| self.agents[*g.node_weight(n).unwrap()].opinion).sum();
                let len = g.neighbors(e).count();

                old_opinion = min;

                // if all nodes of the hyperedge are pairwise compatible
                // all members of this hyperedge assume its average opinion
                if max - min < mintol {
                    let new_opinion = sum / len as f32;
                    for n in g.neighbors(e) {
                        self.agents[*g.node_weight(n).unwrap()].opinion = new_opinion
                    }
                    new_opinion
                } else {
                    old_opinion
                }
            },
        };

        self.acc_change((old_opinion - new_opinion).abs());
    }
}


impl Model for Deffuant {
    fn value(&self) -> f64 {
        self.cluster_max() as f64 / self.num_agents as f64
    }
}

impl MarkovChain for Deffuant {
    fn change(&mut self, rng: &mut impl Rng) {
        self.undo_idx = rng.gen_range(0, self.ctr);
        let val: f64 = rng.gen();
        let n = self.agents.len();
        self.undo_val = if self.undo_idx < n {
            let tmp = self.agents_initial[self.undo_idx].initial_opinion as f64;
            self.agents_initial[self.undo_idx].opinion = val as f32;
            self.agents_initial[self.undo_idx].initial_opinion = val as f32;
            tmp
        } else {
            let tmp = self.rn[self.undo_idx - n];
            self.rn[self.undo_idx - n] = val;
            tmp
        };

        self.agents = self.agents_initial.clone();
        self.ctr = 0;
        self.relax();
    }

    fn undo(&mut self) {
        let n = self.agents.len();
        if self.undo_idx < n {
            self.agents_initial[self.undo_idx].initial_opinion = self.undo_val as f32;
            self.agents_initial[self.undo_idx].opinion = self.undo_val as f32;
        } else {
            self.rn[self.undo_idx - n] = self.undo_val;
        }
    }
}