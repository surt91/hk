// we use float comparision to test if an entry did change during an iteration for performance
// false positives do not lead to wrong results
#![allow(clippy::float_cmp)]

use std::fmt;
use std::path::Path;
use std::fs::File;
use std::io::BufWriter;
use std::io::prelude::*;

use rand::{Rng, SeedableRng};
use rand_distr::{Normal, Pareto, Distribution};
use rand_pcg::Pcg64;
use itertools::Itertools;
use rand::seq::IteratorRandom;
use ordered_float::OrderedFloat;

#[cfg(feature = "graphtool")]
use inline_python::{python,Context};

use super::{PopulationModel, TopologyModel, TopologyRealization, ResourceModel, Agent, EPS};
use super::ABM;

use petgraph::graph::{Graph, NodeIndex};
use petgraph::Undirected;
use petgraph::visit::EdgeRef;
use petgraph::algo::connected_components;
use super::graph::{
    size_largest_connected_component,
    build_er,
    build_ba,
    build_cm,
    build_cm_biased,
    build_lattice,
    build_ws,
    build_ws_lattice,
    build_ba_with_clustering,
};
use super::hypergraph::{
    Hypergraph,
    build_hyper_uniform_er,
    convert_to_simplical_complex,
    build_hyper_uniform_ba,
};

/// maximal time to save density information for
const THRESHOLD: usize = 400;
const ACC_EPS: f32 = 1e-3;
const DENSITYBINS: usize = 100;


pub struct DeffuantBuilder {
    num_agents: u32,

    population_model: PopulationModel,
    topology_model: TopologyModel,

    seed: u64,
}

impl DeffuantBuilder {
    pub fn new(num_agents: u32) -> DeffuantBuilder {
        DeffuantBuilder {
            num_agents,

            population_model: PopulationModel::Uniform(0., 1.),
            topology_model: TopologyModel::FullyConnected,

            seed: 42,
        }
    }

    pub fn population_model(&mut self, population_model: PopulationModel) -> &mut DeffuantBuilder {
        self.population_model = population_model;
        self
    }

    pub fn topology_model(&mut self, topology_model: TopologyModel) -> &mut DeffuantBuilder {
        self.topology_model = topology_model;
        self
    }

    pub fn seed(&mut self, seed: u64) -> &mut DeffuantBuilder {
        self.seed = seed;
        self
    }

    pub fn build(&self) -> Deffuant {
        let rng = Pcg64::seed_from_u64(self.seed);
        let agents: Vec<Agent> = Vec::new();

        let dynamic_density = Vec::new();

        let mut hk = Deffuant {
            num_agents: self.num_agents,
            agents: agents.clone(),
            time: 0,
            mu: 1.,
            topology: TopologyRealization::None,
            population_model: self.population_model.clone(),
            topology_model: self.topology_model.clone(),
            acc_change: 0.,
            dynamic_density,
            density_slice: vec![0; DENSITYBINS+1],
            entropies_acc: Vec::new(),
            rng,
            agents_initial: agents,
        };

        hk.reset();
        hk
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

    pub acc_change: f32,
    dynamic_density: Vec<Vec<u64>>,
    entropies_acc: Vec<f32>,

    density_slice: Vec<u64>,
    // we need many, good (but not crypto) random numbers
    // we will use here the pcg generator
    rng: Pcg64,

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
    fn get_population_model(&self) -> PopulationModel {
        self.population_model.clone()
    }

    fn get_topology_model(&self) -> TopologyModel {
        self.topology_model.clone()
    }

    fn get_resource_model(&self) -> ResourceModel {
        ResourceModel::None
    }

    fn get_agents(&self) -> &Vec<Agent> {
        &self.agents
    }

    fn get_rng(&mut self) -> &mut Pcg64 {
        &mut self.rng
    }
}

impl Deffuant {

    // TODO: separate common functions with HK into a AgentBased Trait

    pub fn reset(&mut self) {
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

        // println!("min:  {}", self.agents.iter().map(|x| OrderedFloat(x.resources)).min().unwrap());
        // println!("max:  {}", self.agents.iter().map(|x| OrderedFloat(x.resources)).max().unwrap());
        // println!("mean: {}", self.agents.iter().map(|x| x.resources).sum::<f32>() / self.agents.len() as f32);

        self.time = 0;
    }

    pub fn step_naive(&mut self) {
        let old_opinion;
        let new_opinion = match &self.topology {
            TopologyRealization::None => {
                // get a random agent
                let idx = self.rng.gen_range(0, self.num_agents) as usize;
                let i = &self.agents[idx];
                old_opinion = i.opinion;

                let idx2 = loop{
                    let tmp = self.rng.gen_range(0, self.num_agents) as usize;
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

        self.acc_change += (old_opinion - new_opinion).abs();
    }

    pub fn sweep(&mut self) {
        for _ in 0..self.num_agents {
            self.step_naive()
        }
        self.add_state_to_density();
        self.time += 1;
    }

    /// A cluster are agents whose distance is less than EPS
    fn list_clusters(&self) -> Vec<Vec<Agent>> {
        let mut clusters: Vec<Vec<Agent>> = Vec::new();
        'agent: for i in &self.agents {
            for c in &mut clusters {
                if (i.opinion - c[0].opinion).abs() < EPS {
                    c.push(i.clone());
                    continue 'agent;
                }
            }
            clusters.push(vec![i.clone(); 1])
        }
        clusters
    }

    pub fn cluster_sizes(&self) -> Vec<usize> {
        let clusters = self.list_clusters();
        clusters.iter()
            .map(|c| c.len() as usize)
            .collect()
    }

    pub fn cluster_max(&self) -> usize {
        let clusters = self.list_clusters();
        clusters.iter()
            .map(|c| c.len() as usize)
            .max()
            .unwrap()
    }

    pub fn write_cluster_sizes(&self, file: &mut File) -> std::io::Result<()> {
        let clusters = self.list_clusters();

        let string_list = clusters.iter()
            .map(|c| c[0].opinion)
            .join(" ");
        writeln!(file, "# {}", string_list)?;

        let string_list = clusters.iter()
            .map(|c| c.len().to_string())
            .join(" ");
        writeln!(file, "{}", string_list)?;
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

        let entropy = self.density_slice.iter().map(|x| {
            let p = *x as f32 / self.num_agents as f32;
            if x > &0 {-p * p.ln()} else {0.}
        }).sum();

        if self.entropies_acc.len() <= self.time {
            self.entropies_acc.push(entropy)
        } else {
            self.entropies_acc[self.time] += entropy;
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

            let entropy = self.density_slice.iter().map(|x| {
                let p = *x as f32 / self.num_agents as f32;
                if x > &0 {-p * p.ln()} else {0.}
            }).sum();
            if self.entropies_acc.len() <= j {
                self.entropies_acc.push(entropy);
            } else {
                self.entropies_acc[j] += entropy;
            }

            j += 1;
        }
    }

    pub fn write_density(&self, file: &mut File) -> std::io::Result<()> {
            let string_list = self.dynamic_density.iter()
            .map(|x| x.iter().join(" "))
            .join("\n");
        writeln!(file, "{}", string_list)
    }

    pub fn write_entropy(&self, file: &mut File) -> std::io::Result<()> {
        let string_list = self.entropies_acc.iter()
            .map(|x| x.to_string())
            .join("\n");
        writeln!(file, "{}", string_list)
    }

    pub fn write_state(&self, file: &mut File) -> std::io::Result<()> {
        let string_list = self.agents.iter()
            .map(|j| j.opinion.to_string())
            .join(" ");
        writeln!(file, "{}", string_list)
    }

    pub fn write_gp(&self, file: &mut File, outfilename: &str) -> std::io::Result<()> {
        writeln!(file, "set terminal pngcairo")?;
        writeln!(file, "set output '{}.png'", outfilename)?;
        writeln!(file, "set xl 't'")?;
        writeln!(file, "set yl 'x_i'")?;
        write!(file, "p '{}' u 0:1 w l not, ", outfilename)?;

        let string_list = (2..self.num_agents)
            .map(|j| format!("'' u 0:{} w l not,", j))
            .join(" ");
        write!(file, "{}", string_list)
    }

    pub fn write_topology_info(&self, file: &mut File) -> std::io::Result<()> {
        let (num_components, lcc_num, lcc, mean_degree) = match &self.topology {
            TopologyRealization::None => (1, 1, self.num_agents as usize, self.num_agents as f64 - 1.),
            TopologyRealization::Graph(g) => {
                let (num, size) = size_largest_connected_component(&g);

                let d = 2. * g.edge_count() as f64 / g.node_count() as f64;

                (connected_components(&g), num, size, d)
            },
            TopologyRealization::Hypergraph(g) => {
                (0, 0, 0, g.mean_deg())
            }
            ,
        };

        // TODO: save more information: size of the largest component, ...
        writeln!(file, "{} {} {} {}", num_components, lcc_num, lcc, mean_degree)
        // println!("n {}, c {}, p {}, m {}, num components: {:?}", n, c, p, m, components);
    }

    pub fn write_state_png(&self, path: &Path) -> std::io::Result<()> {
        let file = File::create(path).unwrap();

        let ref mut w = BufWriter::new(file);
        let gradient = colorous::VIRIDIS;

        let n = self.num_agents;
        let m = (n as f64).sqrt() as u32;
        assert!(m*m == n);

        let mut encoder = png::Encoder::new(w, m, m); // Width is 2 pixels and height is 1.
        encoder.set_color(png::ColorType::RGB);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();

        let data: Vec<Vec<u8>> = self.agents.iter().map(|i| {
            let gr = gradient.eval_continuous(i.opinion as f64);
            vec![gr.r, gr.g, gr.b]
        }).collect();

        let data: Vec<u8> = data.into_iter().flatten().collect();

        writer.write_image_data(&data).unwrap();

        Ok(())
    }

    pub fn relax(&mut self) {
        self.acc_change = ACC_EPS;

        // println!("{:?}", self.agents);
        while self.acc_change >= ACC_EPS {
            self.acc_change = 0.;
            self.sweep();
        }
    }
}
