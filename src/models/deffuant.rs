// we use float comparision to test if an entry did change during an iteration for performance
// false positives do not lead to wrong results
#![allow(clippy::float_cmp)]

use std::fmt;

use rand::Rng;
use rand::seq::IteratorRandom;
use ordered_float::OrderedFloat;

use super::{PopulationModel, TopologyModel, TopologyRealization, ResourceModel, Agent};
use super::{ABM, ABMBuilder};
use super::abm::ABMinternals;

use petgraph::graph::NodeIndex;

impl ABMBuilder {
    pub fn dw(&mut self) -> Deffuant {
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
            agents_initial: agents,
        };

        dw.reset(&mut self.rng);
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

    // for Markov chains
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
    fn sweep(&mut self, mut rng: &mut impl Rng) {
        for _ in 0..self.num_agents {
            self.step_naive(&mut rng)
        }
        self.add_state_to_density();
        self.time += 1;
    }

    fn reset(&mut self, mut rng: &mut impl Rng) {
        self.agents = (0..self.num_agents).map(|_| {
            let xi = self.gen_init_opinion(&mut rng);
            let ei = self.gen_init_tolerance(&mut rng);
            Agent::new(
                xi,
                ei,
                0.,
            )
        }).collect();

        self.agents_initial = self.agents.clone();

        self.topology = self.gen_init_topology(&mut rng);

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
}

impl Deffuant {
    pub fn step_naive(&mut self, mut rng: &mut impl Rng) {
        let old_opinion;
        let new_opinion = match &self.topology {
            TopologyRealization::None => {
                // get a random agent
                let idx = rng.gen_range(0, self.num_agents) as usize;
                let i = &self.agents[idx];
                old_opinion = i.opinion;

                let idx2 = loop{
                    let tmp = rng.gen_range(0, self.num_agents) as usize;
                    if tmp != idx {
                        break tmp
                    }
                };
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
                let idx = rng.gen_range(0, self.num_agents) as usize;
                let i = &self.agents[idx];
                old_opinion = i.opinion;

                let nodes: Vec<NodeIndex<u32>> = g.node_indices().collect();
                let j = g.neighbors(nodes[idx])
                    .choose(&mut rng);
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
                let eidx = rng.gen_range(0, h.edge_nodes.len()) as usize;
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
