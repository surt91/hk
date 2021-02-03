// we use float comparision to test if an entry did change during an iteration for performance
// false positives do not lead to wrong results
#![allow(clippy::float_cmp)]

use std::collections::BTreeMap;
use std::ops::Bound::Included;
use std::fmt;
use std::path::Path;
use std::fs::File;
use std::io::BufWriter;
use std::io::prelude::*;

use rand::{Rng, SeedableRng};
use rand_distr::{Normal, Pareto, Distribution};
use rand_pcg::Pcg64;
use itertools::Itertools;

#[cfg(feature = "graphtool")]
use inline_python::{python,Context};

use ordered_float::OrderedFloat;

use petgraph::graph::NodeIndex;
use petgraph::algo::connected_components;
use super::graph::{
    size_largest_connected_component,
};

use super::{EPS, CostModel, ResourceModel, PopulationModel, TopologyModel, TopologyRealization, Agent};
use super::ABM;
use super::ABMBuilder;

use largedev::{MarkovChain, Model};

/// maximal time to save density information for
const THRESHOLD: usize = 400;
const ACC_EPS: f32 = 1e-3;
const DENSITYBINS: usize = 100;

impl ABMBuilder {
    pub fn hk(&self) -> HegselmannKrause {
        let rng = Pcg64::seed_from_u64(self.seed);
        let agents: Vec<Agent> = Vec::new();

        // datastructure for `step_bisect`
        let opinion_set = BTreeMap::new();

        let dynamic_density = Vec::new();

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
            acc_change: 0.,
            dynamic_density,
            ji: Vec::new(),
            jin: Vec::new(),
            density_slice: vec![0; DENSITYBINS+1],
            entropies_acc: Vec::new(),
            rng,
            undo_idx: 0,
            undo_val: 0.,
            agents_initial: agents,
        };

        hk.reset();
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
    pub acc_change: f32,
    dynamic_density: Vec<Vec<u64>>,
    entropies_acc: Vec<f32>,

    pub ji: Vec<f32>,
    pub jin: Vec<i32>,

    density_slice: Vec<u64>,
    // we need many, good (but not crypto) random numbers
    // we will use here the pcg generator
    rng: Pcg64,

    // for Markov chains
    undo_idx: usize,
    undo_val: f32,
    pub agents_initial: Vec<Agent>,
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
    fn sweep(&mut self) {
        for _ in 0..self.num_agents {
            match self.topology_model {
                // For topologies with few connections, use `step_naive`, otherwise the `step_bisect`
                TopologyModel::FullyConnected => self.step_bisect(),
                _ => self.step_naive(),
            }
        }
        self.add_state_to_density();
        self.time += 1;
    }

    fn reset(&mut self) {
        self.agents = (0..self.num_agents).map(|_| {
            let xi = self.gen_init_opinion();
            let ei = self.gen_init_tolerance();
            let ci = self.gen_init_resources(ei);
            Agent::new(
                xi,
                ei,
                ci,
            )
        }).collect();

        self.agents_initial = self.agents.clone();

        self.topology = self.gen_init_topology();

        self.prepare_opinion_set();
        self.time = 0;
    }

    fn get_population_model(&self) -> PopulationModel {
        self.population_model.clone()
    }

    fn get_topology_model(&self) -> TopologyModel {
        self.topology_model.clone()
    }

    fn get_resource_model(&self) -> ResourceModel {
        self.resource_model.clone()
    }

    fn get_agents(&self) -> &Vec<Agent> {
        &self.agents
    }

    fn get_rng(&mut self) -> &mut Pcg64 {
        &mut self.rng
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

    pub fn step_naive(&mut self) {
        // get a random agent
        let idx = self.rng.gen_range(0, self.num_agents) as usize;
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
            TopologyRealization::Hypergraph(g) => unimplemented!(),
        };

        let new_opinion = sum / count as f32;
        let old = i.opinion;
        let (new_opinion, new_resources) = self.pay(idx, new_opinion);

        self.acc_change += (old - new_opinion).abs();

        self.agents[idx].opinion = new_opinion;
        self.agents[idx].resources = new_resources;

    }

    pub fn step_bisect(&mut self) {
        // get a random agent
        let idx = self.rng.gen_range(0, self.num_agents) as usize;
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

        self.acc_change += (old - new_opinion).abs();
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
                    let nodes: Vec<NodeIndex<u32>> = g.node_indices().collect();
                    for j in g.neighbors(nodes[idx]).chain(std::iter::once(nodes[idx]))
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
        self.acc_change = 0.;

        for i in 0..self.num_agents as usize {
            let (new_opinion, new_resources) = self.pay(i, new_opinions[i]);

            self.acc_change += (self.agents[i].opinion - new_opinion).abs();

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
        self.acc_change = 0.;

        for i in 0..self.num_agents as usize {
            // often, nothing changes -> optimize for this converged case
            let old = self.agents[i].opinion;
            let (new_opinion, new_resources) = self.pay(i, new_opinions[i]);
            self.update_entry(old, new_opinion);

            self.acc_change += (self.agents[i].opinion - new_opinion).abs();

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

    /// A cluster are agents whose distance is less than EPS
    fn list_clusters_nopoor(&self) -> Vec<Vec<Agent>> {
        let mut clusters: Vec<Vec<Agent>> = Vec::new();
        'agent: for i in &self.agents {
            for c in &mut clusters {
                if (i.opinion - c[0].opinion).abs() < EPS && i.resources > 1e-4 {
                    c.push(i.clone());
                    continue 'agent;
                }
            }
            if i.resources > 1e-4 {
                clusters.push(vec![i.clone(); 1])
            }
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

    pub fn write_cluster_sizes_nopoor(&self, file: &mut File) -> std::io::Result<()> {
        let clusters = self.list_clusters_nopoor();

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

    pub fn write_gp_with_resources(&self, file: &mut File, outfilename: &str) -> std::io::Result<()> {
        let t_max = 400;

        writeln!(file, "set terminal pngcairo")?;
        writeln!(file, "set output '{}.png'", outfilename)?;
        writeln!(file, "set xl 't'")?;
        writeln!(file, "set yl 'x_i'")?;
        writeln!(file, "set xr [0:{}]", t_max)?;
        writeln!(file, "set yr [0:1]")?;

        // let l = (self.num_agents as f64).sqrt() as usize;
        // fn n2color(num: usize, l: usize) -> u8 {
        //     (num as f64 / l as f64 * 255.) as u8
        // }

        // let string_list = self.agents.iter().take(100).enumerate()
        //     //.map(|(n, a)| format!("  '<zcat {}' u 0:{} w l lc {} not, \\\n", outfilename, n+1, if a.resources < 1e-4 {3} else {1}))
        //     .map(|(n, _a)| format!("  '<zcat {}' u 0:{} w l lc rgb \"#{:02x}{:02x}{:02x}\" not, \\\n", outfilename, n+1, n2color(n%l, l), n2color(n/l, l), 255-n2color(n%l, l)-n2color(n/l, l)))
        //     .join(" ");

        let string_list = format!("p for [i=1:100] '<zcat {} | head -n {}' u 0:i w l lc 8 not", outfilename, t_max);

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
            self.sweep_synchronous();
        }
    }
}

impl Model for HegselmannKrause {
    fn value(&self) -> f64 {
        self.cluster_max() as f64 / self.num_agents as f64
    }
}

impl MarkovChain for HegselmannKrause {
    fn change(&mut self, rng: &mut impl Rng) {
        self.undo_idx = rng.gen_range(0, self.agents.len());
        let val: f32 = rng.gen();
        self.undo_val = self.agents_initial[self.undo_idx].initial_opinion;

        self.agents_initial[self.undo_idx].opinion = val;
        self.agents_initial[self.undo_idx].initial_opinion = val;

        self.agents = self.agents_initial.clone();
        self.prepare_opinion_set();

        self.relax();
    }

    fn undo(&mut self) {
        self.agents_initial[self.undo_idx].initial_opinion = self.undo_val;
        self.agents_initial[self.undo_idx].opinion = self.undo_val;
    }
}