// we use float comparision to test if an entry did change during an iteration for performance
// false positives do not lead to wrong results
#![allow(clippy::float_cmp)]

use std::collections::BTreeMap;
use std::ops::Bound::Included;
use std::fmt;
use std::fs::File;
use std::io::prelude::*;

use rand::{Rng, SeedableRng};
use rand_distr::{Normal, Pareto, Distribution};
use rand_pcg::Pcg64;
use itertools::Itertools;

use ordered_float::OrderedFloat;

use petgraph::graph::{Graph, NodeIndex};
use petgraph::Undirected;
use petgraph::algo::connected_components;
use super::graph::{size_largest_connected_component, build_er, build_ba, build_cm};

use largedev::{MarkovChain, Model};

/// maximal time to save density information for
const THRESHOLD: usize = 4000;
const EPS: f32 = 2e-3;
const ACC_EPS: f32 = 1e-3;
const DENSITYBINS: usize = 100;

#[derive(PartialEq, Clone)]
pub enum CostModel {
    Rebounce,
    Change(f32),
    Free,
    Annealing(f32),
}

#[derive(PartialEq, Clone)]
pub enum ResourceModel {
    Uniform(f32, f32),
    Pareto(f32, f32),
    Proportional(f32, f32),
    Antiproportional(f32, f32),
    HalfGauss(f32),
}

#[derive(PartialEq, Clone)]
pub enum PopulationModel {
    /// uniform opinions, uniform tolerances
    Uniform(f32, f32),
    /// uniform opinions, bimodal tolerances
    Bimodal(f32, f32),
    /// A fraction of the agents with a different tolerace and a concentrated initial opinion
    /// initial opinion, opinion spread, fraction of agents, epsilon, epsilonspread
    Bridgehead(f32, f32, f32, f32, f32, f32, f32),
    /// uniform opinions, Gaussian tolerances
    Gaussian(f32, f32),
    /// uniform opinions, power law tolerances
    PowerLaw(f32, f32),
    /// uniform opinions, power law with upper bound tolerances
    PowerLawBound(f32, f32, f32),
}

#[derive(PartialEq, Clone)]
pub enum DegreeDist {
    PowerLaw(usize, f32, f32)
}

impl DegreeDist {
    fn gen(self, mut rng: &mut impl Rng) -> Vec<usize> {
        match self{
            DegreeDist::PowerLaw(n, min, exp) => {
                let pareto = Pareto::new(min, exp - 1.).unwrap();
                (0..n).map(|_| pareto.sample(&mut rng).floor() as usize).collect()
            }
        }

    }
}

#[derive(PartialEq, Clone)]
pub enum TopologyModel {
    /// every agent can interact with any other agent
    FullyConnected,
    /// Erdos-Renyi
    ER(f32),
    /// Barabasi-Albert
    BA(f64, usize),
    /// Configuration Model
    CM(DegreeDist),
}

#[derive(Clone, Debug)]
pub struct HKAgent {
    pub opinion: f32,
    pub tolerance: f32,
    pub initial_opinion: f32,
    pub resources: f32,
}

impl HKAgent {
    fn new(opinion: f32, tolerance: f32, resources: f32) -> HKAgent {
        HKAgent {
            opinion,
            tolerance,
            initial_opinion: opinion,
            resources,
        }
    }
}

impl PartialEq for HKAgent {
    fn eq(&self, other: &HKAgent) -> bool {
        (self.opinion - other.opinion).abs() < EPS
            && (self.tolerance - other.tolerance).abs() < EPS
    }
}

pub struct HegselmannKrauseBuilder {
    num_agents: u32,

    cost_model: CostModel,
    resource_model: ResourceModel,
    population_model: PopulationModel,
    topology_model: TopologyModel,

    seed: u64,
}

impl HegselmannKrauseBuilder {
    pub fn new(num_agents: u32) -> HegselmannKrauseBuilder {
        HegselmannKrauseBuilder {
            num_agents,

            cost_model: CostModel::Free,
            resource_model: ResourceModel::Uniform(0., 1.),
            population_model: PopulationModel::Uniform(0., 1.),
            topology_model: TopologyModel::FullyConnected,

            seed: 42,
        }
    }

    pub fn cost_model(&mut self, cost_model: CostModel) -> &mut HegselmannKrauseBuilder {
        self.cost_model = cost_model;
        self
    }

    pub fn resource_model(&mut self, resource_model: ResourceModel) -> &mut HegselmannKrauseBuilder {
        self.resource_model = resource_model;
        self
    }

    pub fn population_model(&mut self, population_model: PopulationModel) -> &mut HegselmannKrauseBuilder {
        self.population_model = population_model;
        self
    }

    pub fn topology_model(&mut self, topology_model: TopologyModel) -> &mut HegselmannKrauseBuilder {
        self.topology_model = topology_model;
        self
    }

    pub fn seed(&mut self, seed: u64) -> &mut HegselmannKrauseBuilder {
        self.seed = seed;
        self
    }

    pub fn build(&self) -> HegselmannKrause {
        let rng = Pcg64::seed_from_u64(self.seed);
        let agents: Vec<HKAgent> = Vec::new();

        // datastructure for `step_bisect`
        let opinion_set = BTreeMap::new();

        let dynamic_density = Vec::new();

        let mut hk = HegselmannKrause {
            num_agents: self.num_agents,
            agents: agents.clone(),
            time: 0,
            topology: None,
            topology_idx: None,
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
    pub agents: Vec<HKAgent>,
    pub time: usize,

    /// topology of the possible interaction between agents
    /// None means fully connected
    topology: Option<Graph<usize, u32, Undirected>>,
    topology_idx: Option<Vec<NodeIndex>>,

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
    pub agents_initial: Vec<HKAgent>,
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

    fn stretch(x: f32, low: f32, high: f32) -> f32 {
        x*(high-low)+low
    }

    fn gen_init_opinion(&mut self) -> f32 {
        match self.population_model {
            PopulationModel::Bridgehead(x_init, x_spread, frac, _eps_init, _eps_spread, _eps_min, _eps_max) => {
                if self.rng.gen::<f32>() > frac {
                    self.rng.gen()
                } else {
                    HegselmannKrause::stretch(self.rng.gen(), x_init-x_spread, x_init+x_spread)
                }
            },
            _ => self.rng.gen(),
        }
    }

    fn gen_init_tolerance(&mut self) -> f32 {
        match self.population_model {
            PopulationModel::Uniform(min, max) => HegselmannKrause::stretch(self.rng.gen(), min, max),
            PopulationModel::Bimodal(first, second) => if self.rng.gen::<f32>() < 0.5 {first} else {second},
            PopulationModel::Bridgehead(_x_init, _x_spread, frac, eps_init, eps_spread, eps_min, eps_max) => {
                if self.rng.gen::<f32>() > frac {
                    HegselmannKrause::stretch(self.rng.gen(), eps_min, eps_max)
                } else {
                    HegselmannKrause::stretch(self.rng.gen(), eps_init-eps_spread, eps_init+eps_spread)
                }
            },
            PopulationModel::Gaussian(mean, sdev) => {
                let gauss = Normal::new(mean, sdev).unwrap();
                // draw gaussian RN until you get one in range
                loop {
                    let x = gauss.sample(&mut self.rng);
                    if x <= 1. && x >= 0. {
                        break x
                    }
                }
            },
            PopulationModel::PowerLaw(min, exponent) => {
                let pareto = Pareto::new(min, exponent - 1.).unwrap();
                pareto.sample(&mut self.rng)
            }
            PopulationModel::PowerLawBound(min, max, exponent) => {
                // http://mathworld.wolfram.com/RandomNumber.html
                fn powerlaw(y: f32, low: f32, high: f32, alpha: f32) -> f32 {
                    ((high.powf(alpha+1.) - low.powf(alpha+1.))*y + low.powf(alpha+1.)).powf(1./(alpha+1.))
                }
                powerlaw(self.rng.gen(), min, max, exponent)
            }
        }
    }

    fn gen_init_resources(&mut self, confidence: f32) -> f32 {
        match self.resource_model {
            ResourceModel::Uniform(l, u) => HegselmannKrause::stretch(self.rng.gen(), l, u),
            ResourceModel::Pareto(x0, a) => {
                let pareto = Pareto::new(x0, a - 1.).unwrap();
                pareto.sample(&mut self.rng)
            },
            ResourceModel::Proportional(a, offset) => (confidence - offset) * a,
            ResourceModel::Antiproportional(a, offset) => 1. - (confidence - offset) * a,
            ResourceModel::HalfGauss(sigma) => {
                let gauss = Normal::new(0., sigma).unwrap();
                gauss.sample(&mut self.rng).abs()
            },
        }
    }

    fn gen_init_topology(&mut self) -> Option<Graph<usize, u32, Undirected>> {
        match &self.topology_model {
            TopologyModel::FullyConnected => None,
            TopologyModel::ER(c) => {
                let mut g;
                while {
                    let n = self.agents.len();
                    g = build_er(n, *c as f64, &mut self.rng);
                    size_largest_connected_component(&g).0 != 1
                } {}

                self.topology_idx = Some(g.node_indices().collect());

                Some(g)
            },
            TopologyModel::BA(degree, m0) => {
                let n = self.agents.len();
                let g = build_ba(n, *degree, *m0, &mut self.rng);

                self.topology_idx = Some(g.node_indices().collect());

                Some(g)
            }
            TopologyModel::CM(degree_dist) => {
                let g = build_cm(move |r| degree_dist.clone().gen(r), &mut self.rng);

                self.topology_idx = Some(g.node_indices().collect());

                Some(g)
            }
        }
    }

    pub fn prepare_opinion_set(&mut self) {
        self.opinion_set.clear();
        for i in self.agents.iter() {
            *self.opinion_set.entry(OrderedFloat(i.opinion)).or_insert(0) += 1;
        }
            assert!(self.opinion_set.iter().map(|(_, v)| v).sum::<u32>() == self.num_agents);
    }

    pub fn reset(&mut self) {
        self.agents = (0..self.num_agents).map(|_| {
            let xi = self.gen_init_opinion();
            let ei = self.gen_init_tolerance();
            let ci = self.gen_init_resources(ei);
            HKAgent::new(
                xi,
                ei,
                ci,
            )
        }).collect();

        self.agents_initial = self.agents.clone();

        self.topology = self.gen_init_topology();

        // println!("min:  {}", self.agents.iter().map(|x| OrderedFloat(x.resources)).min().unwrap());
        // println!("max:  {}", self.agents.iter().map(|x| OrderedFloat(x.resources)).max().unwrap());
        // println!("mean: {}", self.agents.iter().map(|x| x.resources).sum::<f32>() / self.agents.len() as f32);

        self.prepare_opinion_set();
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

        let (sum, count) = if let (Some(g), Some(nodes)) = (self.topology.as_ref(), self.topology_idx.as_ref()) {
            // iterate over all neighbors and yourself to find all interaction partners
            g.neighbors(nodes[idx])
                .chain(std::iter::once(nodes[idx]))
                .map(|j| self.agents[g[j]].opinion)
                .filter(|j| (i.opinion - j).abs() < i.tolerance)
                .fold((0., 0), |(sum, count), i| (sum + i, count + 1))
        } else {
            self.agents.iter()
                .map(|j| j.opinion)
                .filter(|j| (i.opinion - j).abs() < i.tolerance)
                .fold((0., 0), |(sum, count), i| (sum + i, count + 1))
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

    pub fn sweep(&mut self) {
        for _ in 0..self.num_agents {
            match self.topology_model {
                // For topologies with few connections, use `step_naive`, otherwise the `step_bisect`
                TopologyModel::ER(_) | TopologyModel::BA(_, _) | TopologyModel::CM(_)
                    => self.step_naive(),
                TopologyModel::FullyConnected => self.step_bisect(),
            }
        }
        self.add_state_to_density();
        self.time += 1;
    }

    fn sync_new_opinions_naive(&self) -> Vec<f32> {
        self.agents.iter().enumerate().map(|(idx, i)| {
            let mut tmp = 0.;
            let mut count = 0;

            if let (Some(g), Some(nodes)) = (self.topology.as_ref(), self.topology_idx.as_ref()) {
                // iterate over all neighbors and yourself (without yourself there could be infinite loops of two flipping agents)
                for j in g.neighbors(nodes[idx]).chain(std::iter::once(nodes[idx]))
                       .filter(|j| (i.opinion - self.agents[g[*j]].opinion).abs() < i.tolerance) {
                    tmp += self.agents[g[j]].opinion;
                    count += 1;
                }
            } else {
                for j in self.agents.iter()
                        .filter(|j| (i.opinion - j.opinion).abs() < i.tolerance) {
                    tmp += j.opinion;
                    count += 1;
                }
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
            TopologyModel::ER(_) | TopologyModel::BA(_, _) | TopologyModel::CM(_)
                => self.sweep_synchronous_naive(),
            TopologyModel::FullyConnected => self.sweep_synchronous_bisect(),
        }
        self.time += 1;
    }

    /// A cluster are agents whose distance is less than EPS
    fn list_clusters(&self) -> Vec<Vec<HKAgent>> {
        let mut clusters: Vec<Vec<HKAgent>> = Vec::new();
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
    fn list_clusters_nopoor(&self) -> Vec<Vec<HKAgent>> {
        let mut clusters: Vec<Vec<HKAgent>> = Vec::new();
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
        writeln!(file, "set terminal pngcairo")?;
        writeln!(file, "set output '{}.png'", outfilename)?;
        writeln!(file, "set xl 't'")?;
        writeln!(file, "set yl 'x_i'")?;
        writeln!(file, "set xr [0:40]")?;
        writeln!(file, "set yr [0:1]")?;
        writeln!(file, "p \\")?;

        let string_list = self.agents.iter().take(100).enumerate()
            .map(|(n, a)| format!("  '<zcat {}' u 0:{} w l lc {} not, \\\n", outfilename, n+1, if a.resources < 1e-4 {3} else {1}))
            .join(" ");

        write!(file, "{}", string_list)
    }

    pub fn write_topology_info(&self, file: &mut File) -> std::io::Result<()> {
        let (num_components, lcc_num, lcc) =
            if let Some(g) = &self.topology {
                let (num, size) = size_largest_connected_component(&g);

                (connected_components(&g), num, size)
            } else {
                // fully connected
                (1, 1, self.num_agents as usize)
            };

        // TODO: save more information: size of the largest component, ...
        writeln!(file, "{} {} {}", num_components, lcc_num, lcc)
        // println!("n {}, c {}, p {}, m {}, num components: {:?}", n, c, p, m, components);
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