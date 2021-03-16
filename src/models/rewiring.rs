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

impl ABMBuilder {
    pub fn rewiring(&self) -> RewDeffuant {
        let rng = Pcg64::seed_from_u64(self.seed);
        let agents: Vec<Agent> = Vec::new();

        let mut dw = RewDeffuant {
            num_agents: self.num_agents,
            agents: agents.clone(),
            time: 0,
            mu: 1.,
            topology: TopologyRealization::None,
            population_model: self.population_model.clone(),
            topology_model: self.topology_model.clone(),
            abm_internals: ABMinternals::new(),
            rng,
            agents_initial: agents,
        };

        dw.reset();
        dw
    }
}

#[derive(Clone)]
pub struct RewDeffuant {
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
    pub agents_initial: Vec<Agent>,
}

impl PartialEq for RewDeffuant {
    fn eq(&self, other: &RewDeffuant) -> bool {
        self.agents == other.agents
    }
}

impl fmt::Debug for RewDeffuant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DW {{ N: {}, agents: {:?} }}", self.num_agents, self.agents)
    }
}

impl ABM for RewDeffuant {
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

impl RewDeffuant {
    pub fn step_naive(&mut self) {
        let old_opinion;
        let (new_opinion, changed_edge) = match &self.topology {
            TopologyRealization::Hypergraph(h) => {
                // get a random, non-empty hyperdege
                let e = loop {
                    let e = h.edge_nodes.iter().choose(&mut self.rng).unwrap();
                    if h.factor_graph.neighbors(*e).count() > 0 {
                        break *e
                    };
                };

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
                    (Some(new_opinion), e)
                } else {
                    (None, e)
                }
            },
            _ => panic!("only implemented for hypergraphs")
        };

        if let Some(new_o) = new_opinion {
            self.acc_change((old_opinion - new_o).abs());
        } else {
            if let TopologyRealization::Hypergraph(h) = &mut self.topology {
                let e = changed_edge;
                // since they could not reach an agreement, a random agent leaves this hyperedge
                // frustrated and joins a random different hyperedge
                let leaving = h.factor_graph.neighbors(e).choose(&mut self.rng).unwrap();
                let new_edge = loop {
                    let tmp = h.edge_nodes.iter().choose(&mut self.rng).unwrap();
                    if h.factor_graph.find_edge(*tmp, leaving).is_none() {
                        break tmp
                    }
                };
                h.factor_graph.add_edge(leaving, *new_edge, 1);
                let to_remove = h.factor_graph.find_edge(leaving, e).unwrap();
                h.factor_graph.remove_edge(to_remove);
            }
        }
    }
}
