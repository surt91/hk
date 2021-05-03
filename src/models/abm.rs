
use std::path::Path;

use rand::{Rng, SeedableRng};
use rand_distr::{Normal, Pareto, Distribution};
use rand_pcg::Pcg64;

#[cfg(feature = "graphtool")]
use ordered_float::OrderedFloat;

#[cfg(feature = "graphtool")]
use inline_python::{python,Context};

use petgraph::graph::Graph;
use petgraph::Undirected;
use super::graph::{
    size_largest_connected_component,
    max_betweenness_approx,
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
    build_hyper_gaussian_er,
    build_hyper_uniform_lattice_3_12,
    build_hyper_uniform_lattice_5_15,
};

use std::fs::File;
use std::io::BufWriter;
use std::io::prelude::*;
use itertools::Itertools;
use petgraph::algo::connected_components;

#[cfg(feature = "graphtool")]
use petgraph::visit::EdgeRef;

pub const EPS: f32 = 2e-3;
pub const ACC_EPS: f32 = 1e-3;
/// maximal time to save density information for
const THRESHOLD: usize = 10000;
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
    None,
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
    pub fn gen(self, mut rng: &mut impl Rng) -> Vec<usize> {
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
    /// biased Configuration Model
    CMBiased(DegreeDist),
    /// Configuration Model
    CM(DegreeDist),
    /// square lattice
    SquareLattice(usize),
    /// Watts Strogatz
    WS(usize, f64),
    /// Watts Strogatz on a square lattice
    WSlat(usize, f64),
    /// BA + Triangles
    BAT(usize, f64),
    /// uniform HyperER
    HyperER(f64, usize),
    /// uniform HyperER simplical complex
    HyperERSC(f64, usize),
    /// uniform HyperBA
    HyperBA(usize, usize),
    /// uniform HyperER with two sizes of hyperedges
    HyperER2(f64, f64, usize, usize),
    /// HyperER with Gaussian distributed degrees for all orders
    HyperERGaussian(f64, f64, f64),
    /// Hypergraph with a spatial structure
    HyperLattice_3_12,
    /// Hypergraph with a spatial structure
    HyperLattice_5_15,
    /// Watts Strogatz Hypergraphs on a 3_12 lattice
    HyperWSlat(f64),
}

#[derive(Clone, Debug)]
pub enum TopologyRealization {
    Graph(Graph<usize, u32, Undirected>),
    Hypergraph(Hypergraph),
    None
}


#[derive(Clone, Debug)]
pub struct Agent {
    pub opinion: f32,
    pub tolerance: f32,
    pub initial_opinion: f32,
    pub resources: f32,
}

impl Agent {
    pub fn new(opinion: f32, tolerance: f32, resources: f32) -> Agent {
        Agent {
            opinion,
            tolerance,
            initial_opinion: opinion,
            resources,
        }
    }
}

impl PartialEq for Agent {
    fn eq(&self, other: &Agent) -> bool {
        (self.opinion - other.opinion).abs() < EPS
            && (self.tolerance - other.tolerance).abs() < EPS
    }
}


fn stretch(x: f32, low: f32, high: f32) -> f32 {
    x*(high-low)+low
}

#[derive(Clone, Debug)]
pub struct ABMinternals {
    acc_change: f32,
    dynamic_density: Vec<Vec<u64>>,
    entropies_acc: Vec<f32>,
    max_betweenness: f64,

    density_slice: Vec<u64>,
}

impl ABMinternals {
    pub fn new() -> ABMinternals {
        ABMinternals {
            acc_change: 0.,
            dynamic_density: Vec::new(),
            density_slice: vec![0; DENSITYBINS+1],
            entropies_acc: Vec::new(),
            max_betweenness: 0.,
        }
    }
}

pub trait ABM {
    fn sweep(&mut self, rng: &mut impl Rng);
    fn reset(&mut self, rng: &mut impl Rng);

    // getter methods to access fields
    fn get_population_model(&self) -> PopulationModel;
    fn get_resource_model(&self) -> ResourceModel;
    fn get_topology_model(&self) -> TopologyModel;
    fn get_topology(&self) -> &TopologyRealization;
    fn get_agents(&self) -> &Vec<Agent>;
    fn get_time(&self) -> usize;

    fn get_mut_abm_internals(&mut self) -> &mut ABMinternals;
    fn get_abm_internals(&self) -> &ABMinternals;

    fn get_acc_change(&mut self) -> f32 {
        self.get_abm_internals().acc_change
    }

    fn acc_change(&mut self, delta: f32) {
        self.get_mut_abm_internals().acc_change += delta;
    }

    fn acc_change_reset(&mut self) {
        self.get_mut_abm_internals().acc_change = 0.;
    }

    fn get_num_agents(&self) -> u32 {
        self.get_agents().len() as u32
    }

    fn relax(&mut self, mut rng: &mut impl Rng) {
        self.get_mut_abm_internals().acc_change = ACC_EPS;

        // println!("{:?}", self.agents);
        while self.get_abm_internals().acc_change >= ACC_EPS {
            self.get_mut_abm_internals().acc_change = 0.;
            self.sweep(&mut rng);
        }
    }

    fn gen_init_opinion(&mut self, rng: &mut impl Rng) -> f32 {
        match self.get_population_model() {
            PopulationModel::Bridgehead(x_init, x_spread, frac, _eps_init, _eps_spread, _eps_min, _eps_max) => {
                if rng.gen::<f32>() > frac {
                    rng.gen()
                } else {
                    stretch(rng.gen(), x_init-x_spread, x_init+x_spread)
                }
            },
            _ => rng.gen(),
        }
    }

    fn gen_init_tolerance(&mut self, mut rng: &mut impl Rng) -> f32 {
        match self.get_population_model() {
            PopulationModel::Uniform(min, max) => stretch(rng.gen(), min, max),
            PopulationModel::Bimodal(first, second) => if rng.gen::<f32>() < 0.5 {first} else {second},
            PopulationModel::Bridgehead(_x_init, _x_spread, frac, eps_init, eps_spread, eps_min, eps_max) => {
                if rng.gen::<f32>() > frac {
                    stretch(rng.gen(), eps_min, eps_max)
                } else {
                    stretch(rng.gen(), eps_init-eps_spread, eps_init+eps_spread)
                }
            },
            PopulationModel::Gaussian(mean, sdev) => {
                let gauss = Normal::new(mean, sdev).unwrap();
                // draw gaussian RN until you get one in range
                loop {
                    let x = gauss.sample(&mut rng);
                    if x <= 1. && x >= 0. {
                        break x
                    }
                }
            },
            PopulationModel::PowerLaw(min, exponent) => {
                let pareto = Pareto::new(min, exponent - 1.).unwrap();
                pareto.sample(&mut rng)
            }
            PopulationModel::PowerLawBound(min, max, exponent) => {
                // http://mathworld.wolfram.com/RandomNumber.html
                fn powerlaw(y: f32, low: f32, high: f32, alpha: f32) -> f32 {
                    ((high.powf(alpha+1.) - low.powf(alpha+1.))*y + low.powf(alpha+1.)).powf(1./(alpha+1.))
                }
                powerlaw(rng.gen(), min, max, exponent)
            }
        }
    }

    fn gen_init_topology(&mut self, mut rng: &mut impl Rng) -> TopologyRealization {
        match &self.get_topology_model() {
            TopologyModel::FullyConnected => TopologyRealization::None,
            TopologyModel::ER(c) => {
                let n = self.get_agents().len();
                let g = loop {
                    let tmp = build_er(n, *c as f64, &mut rng);
                    if size_largest_connected_component(&tmp).0 == 1 {
                        break tmp
                    }
                };
                // let g = build_er(n, *c as f64, &mut rng);

                TopologyRealization::Graph(g)
            },
            TopologyModel::BA(degree, m0) => {
                let n = self.get_agents().len();
                let g = build_ba(n, *degree, *m0, &mut rng);

                TopologyRealization::Graph(g)
            },
            TopologyModel::CMBiased(degree_dist) => {
                let g = loop {
                    let tmp = build_cm_biased(move |r| degree_dist.clone().gen(r), &mut rng);
                    if size_largest_connected_component(&tmp).0 == 1 {
                        break tmp
                    }
                };

                TopologyRealization::Graph(g)
            },
            TopologyModel::CM(degree_dist) => {
                let g = loop {
                    let tmp = build_cm(move |r| degree_dist.clone().gen(r), &mut rng);
                    if size_largest_connected_component(&tmp).0 == 1 {
                        break tmp
                    }
                };

                TopologyRealization::Graph(g)
            },
            TopologyModel::SquareLattice(next_neighbors) => {
                let n = self.get_agents().len();
                let g = build_lattice(n, *next_neighbors);

                TopologyRealization::Graph(g)
            },
            TopologyModel::WS(neighbors, rewiring) => {
                let n = self.get_agents().len();
                // let g = build_ws(n, *neighbors, *rewiring, &mut rng);
                let g = loop {
                    let tmp = build_ws(n, *neighbors, *rewiring, &mut rng);
                    if size_largest_connected_component(&tmp).0 == 1 {
                        break tmp
                    }
                };

                TopologyRealization::Graph(g)
            },
            TopologyModel::WSlat(neighbors, rewiring) => {
                let n = self.get_agents().len();
                // let g = build_ws(n, *neighbors, *rewiring, &mut rng);
                let g = loop {
                    let tmp = build_ws_lattice(n, *neighbors, *rewiring, &mut rng);
                    if size_largest_connected_component(&tmp).0 == 1 {
                        break tmp
                    }
                };

                TopologyRealization::Graph(g)
            },
            TopologyModel::BAT(degree, mt) => {
                let n = self.get_agents().len();
                let m0 = (*degree as f64 / 2.).ceil() as usize + mt.ceil() as usize;
                let g = build_ba_with_clustering(n, *degree, m0, *mt, &mut rng);

                TopologyRealization::Graph(g)
            },
            TopologyModel::HyperER(c, k) => {
                let n = self.get_agents().len();
                // TODO: maybe ensure connectedness
                let g = build_hyper_uniform_er(n, *c, *k, &mut rng);

                TopologyRealization::Hypergraph(g)
            },
            TopologyModel::HyperERSC(c, k) => {
                let n = self.get_agents().len();
                // TODO: maybe ensure connectedness
                let g = convert_to_simplical_complex(&build_hyper_uniform_er(n, *c, *k, &mut rng));

                TopologyRealization::Hypergraph(g)
            },
            TopologyModel::HyperBA(m, k) => {
                let n = self.get_agents().len();
                let g = build_hyper_uniform_ba(n, *m, *k, &mut rng);

                TopologyRealization::Hypergraph(g)
            },
            TopologyModel::HyperER2(c1, c2, k1, k2) => {
                let n = self.get_agents().len();
                let mut g = build_hyper_uniform_er(n, *c1, *k1, &mut rng);
                g.add_er_hyperdeges(*c2, *k2, &mut rng);

                TopologyRealization::Hypergraph(g)
            },
            TopologyModel::HyperERGaussian(c, mu, sigma) => {
                let n = self.get_agents().len();

                let g = build_hyper_gaussian_er(n, *c, *mu, *sigma, &mut rng);

                TopologyRealization::Hypergraph(g)
            },
            TopologyModel::HyperLattice_3_12 => {
                let n = self.get_agents().len();

                let g = build_hyper_uniform_lattice_3_12(n);

                TopologyRealization::Hypergraph(g)
            },
            TopologyModel::HyperLattice_5_15 => {
                let n = self.get_agents().len();

                let g = build_hyper_uniform_lattice_5_15(n);

                TopologyRealization::Hypergraph(g)
            },
            TopologyModel::HyperWSlat(p) => {
                let n = self.get_agents().len();

                let mut g = build_hyper_uniform_lattice_3_12(n);
                g.rewire(*p, &mut rng);

                TopologyRealization::Hypergraph(g)
            },
        }
    }

    fn gen_init_resources(&mut self, confidence: f32, mut rng: &mut impl Rng) -> f32 {
        match self.get_resource_model() {
            ResourceModel::Uniform(l, u) => stretch(rng.gen(), l, u),
            ResourceModel::Pareto(x0, a) => {
                let pareto = Pareto::new(x0, a - 1.).unwrap();
                pareto.sample(&mut rng)
            },
            ResourceModel::Proportional(a, offset) => (confidence - offset) * a,
            ResourceModel::Antiproportional(a, offset) => 1. - (confidence - offset) * a,
            ResourceModel::HalfGauss(sigma) => {
                let gauss = Normal::new(0., sigma).unwrap();
                gauss.sample(&mut rng).abs()
            },
            ResourceModel::None => unimplemented!()
        }
    }


    #[cfg(feature = "graphtool")]
    fn write_graph_png(&self, path: &Path, active: bool) -> std::io::Result<()> {
        let mut py = python!{g = None};
        self.write_graph_png_with_memory(path, &mut py, active)
    }

    #[cfg(not(feature = "graphtool"))]
    fn write_graph_png(&self, _path: &Path, _active: bool) -> std::io::Result<()> {
        println!("Warning: This executable was not compiled the 'graphtool' feature. Can not draw the graph representation. Will ignore this error and proceed.");
        Ok(())
    }

    #[cfg(feature = "graphtool")]
    fn write_graph_png_with_memory(&self, path: &Path, py: &mut Context, active: bool) -> std::io::Result<()> {
        let gradient = colorous::VIRIDIS;
        let out = path.to_str().unwrap();

        let edgelist: Vec<Vec<usize>>;
        let colors: Vec<Vec<f64>>;

        match &self.get_topology() {
            TopologyRealization::Graph(g) => {
                colors = self.get_agents().iter().map(|i| {
                    let gr = gradient.eval_continuous(i.opinion as f64);
                    vec![gr.r as f64 / 255., gr.g as f64 / 255., gr.b as f64 / 255., 1.]
                }).collect();
                edgelist = if active {
                    g.edge_indices()
                        .map(|e| {
                            let (u, v) = g.edge_endpoints(e).unwrap();
                            vec![u.index(), v.index()]
                        })
                        .filter(|v| (self.get_agents()[v[0]].opinion - self.get_agents()[v[1]].opinion).abs() <= self.get_agents()[v[0]].tolerance)
                        .collect()
                } else {
                    g.edge_indices()
                        .map(|e| {
                            let (u, v) = g.edge_endpoints(e).unwrap();
                            vec![u.index(), v.index()]
                        })
                        .collect()
                };
            }
            TopologyRealization::Hypergraph(h) => {
                colors = self.get_agents().iter().map(|i| {
                    let gr = gradient.eval_continuous(i.opinion as f64);
                    vec![gr.r as f64 / 255., gr.g as f64 / 255., gr.b as f64 / 255., 1.]
                }).chain(
                    h.edge_nodes.iter()
                        .map(|_| vec![1., 0., 0., 1.])
                ).collect();

                let g = &h.factor_graph;
                edgelist = if active {
                    h.edge_nodes.iter()
                        .filter(|&&e| {
                            let opin = g.neighbors(e).map(|n| OrderedFloat(self.get_agents()[*g.node_weight(n).unwrap()].opinion));
                            let opix = g.neighbors(e).map(|n| OrderedFloat(self.get_agents()[*g.node_weight(n).unwrap()].opinion));
                            let tol = g.neighbors(e).map(|n| OrderedFloat(self.get_agents()[*g.node_weight(n).unwrap()].tolerance));
                            opix.max().unwrap().into_inner() - opin.min().unwrap().into_inner() < tol.min().unwrap().into_inner()
                        })
                        .flat_map(|&e| {
                            g.edges(e).map(|edge| {
                                vec![edge.source().index(), edge.target().index()]
                            })
                        })
                        .collect()
                } else {
                    g.edge_indices()
                        .map(|e| {
                            let (u, v) = g.edge_endpoints(e).unwrap();
                            vec![u.index(), v.index()]
                        })
                        .collect()
                };
            }
            _ => {
                edgelist = Vec::new();
                colors = Vec::new();
            }
        }

        py.run(python! {
            import graph_tool.all as gt

            if len('edgelist) == 0:
                print("Warning: There are no edges in this graph, do not render anything!")

            g = None
            if g is None:
                g = gt.Graph(directed=False)
                g.add_edge_list('edgelist)
                pos = gt.sfdp_layout(g)
            else:
                g.clear_edges()
                g.add_edge_list('edgelist)

            colors = g.new_vp("vector<double>")
            sizes = g.new_vp("double")
            shapes = g.new_vp("int")
            for n, c in enumerate('colors):
                colors[n] = c
                sizes[n] = 3 if c == [1., 0., 0., 1.] else 20
                shapes[n] = 0 if c == [1., 0., 0., 1.] else 2

            gt.graph_draw(
                g,
                pos=pos,
                vertex_shape=shapes,
                vertex_size=sizes,
                vertex_fill_color=colors,
                edge_color=[0.7, 0.7, 0.7, 0.5],
                bg_color=[1., 1., 1., 1.],
                output_size=(1920, 1920),
                adjust_aspect=False,
                output='out,
            )
        });

        Ok(())
    }

    /// A cluster are agents whose distance is less than EPS
    fn list_clusters(&self) -> Vec<Vec<Agent>> {
        let mut clusters: Vec<Vec<Agent>> = Vec::new();
        'agent: for i in self.get_agents() {
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

    fn list_clusters_nopoor(&self) -> Vec<Vec<Agent>> {
        let mut clusters: Vec<Vec<Agent>> = Vec::new();
        'agent: for i in self.get_agents() {
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

    fn cluster_sizes(&self) -> Vec<usize> {
        let clusters = self.list_clusters();
        clusters.iter()
            .map(|c| c.len() as usize)
            .collect()
    }

    fn cluster_max(&self) -> usize {
        let clusters = self.list_clusters();
        clusters.iter()
            .map(|c| c.len() as usize)
            .max()
            .unwrap()
    }

    fn write_cluster_sizes(&self, file: &mut File) -> std::io::Result<()> {
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

    fn write_cluster_sizes_nopoor(&self, file: &mut File) -> std::io::Result<()> {
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

    fn add_state_to_density(&mut self) {
        if self.get_time() > THRESHOLD {
            return
        }

        for i in 0..DENSITYBINS {
            self.get_mut_abm_internals().density_slice[i] = 0;
        }

        for i in 0..self.get_num_agents() {
            let op = self.get_agents()[i as usize].opinion;
            self.get_mut_abm_internals().density_slice[(op*DENSITYBINS as f32) as usize] += 1;
        }

        let t = self.get_time();
        if self.get_abm_internals().dynamic_density.len() <= t {
            let slice = self.get_abm_internals().density_slice.clone();
            self.get_mut_abm_internals().dynamic_density.push(slice);
        } else {
            for i in 0..DENSITYBINS {
                self.get_mut_abm_internals().dynamic_density[t][i] += self.get_abm_internals().density_slice[i];
            }
        }

        let entropy = self.get_abm_internals().density_slice.iter().map(|x| {
            let p = *x as f32 / self.get_num_agents() as f32;
            if x > &0 {-p * p.ln()} else {0.}
        }).sum();

        if self.get_abm_internals().entropies_acc.len() <= self.get_time() {
            self.get_mut_abm_internals().entropies_acc.push(entropy)
        } else {
            let t = self.get_time();
            self.get_mut_abm_internals().entropies_acc[t] += entropy;
        }
    }

    fn fill_density(&mut self) {
        let mut j = self.get_time();
        while j < THRESHOLD {
            if self.get_abm_internals().dynamic_density.len() <= j {
                let slice = self.get_abm_internals().density_slice.clone();
                self.get_mut_abm_internals().dynamic_density.push(slice);
            } else {
                for i in 0..DENSITYBINS {
                    self.get_mut_abm_internals().dynamic_density[j][i] += self.get_abm_internals().density_slice[i];
                }
            }

            let entropy = self.get_abm_internals().density_slice.iter().map(|x| {
                let p = *x as f32 / self.get_num_agents() as f32;
                if x > &0 {-p * p.ln()} else {0.}
            }).sum();
            if self.get_abm_internals().entropies_acc.len() <= j {
                self.get_mut_abm_internals().entropies_acc.push(entropy);
            } else {
                self.get_mut_abm_internals().entropies_acc[j] += entropy;
            }

            j += 1;
        }
    }

    fn write_density(&self, file: &mut File) -> std::io::Result<()> {
        let string_list = self.get_abm_internals().dynamic_density.iter()
            .map(|x| x.iter().join(" "))
            .join("\n");
        writeln!(file, "{}", string_list)
    }

    fn write_entropy(&self, file: &mut File) -> std::io::Result<()> {
        let string_list = self.get_abm_internals().entropies_acc.iter()
            .map(|x| x.to_string())
            .join("\n");
        writeln!(file, "{}", string_list)
    }

    fn write_state(&self, file: &mut File) -> std::io::Result<()> {
        let string_list = self.get_agents().iter()
            .map(|j| j.opinion.to_string())
            .join(" ");
        writeln!(file, "{}", string_list)
    }

    fn write_gp(&self, file: &mut File, outfilename: &str) -> std::io::Result<()> {
        writeln!(file, "set terminal pngcairo")?;
        writeln!(file, "set output '{}.png'", outfilename)?;
        writeln!(file, "set xl 't'")?;
        writeln!(file, "set yl 'x_i'")?;
        write!(file, "p '{}' u 0:1 w l not, ", outfilename)?;

        let string_list = (2..self.get_num_agents())
            .map(|j| format!("'' u 0:{} w l not,", j))
            .join(" ");
        write!(file, "{}", string_list)
    }

    fn betweenness_active(&self, g: &Graph<usize, u32, Undirected>) -> f64 {
        use petgraph::graph::NodeIndex;
        let n = g.node_count();
        // construct a new graph with only active edges
        let edgelist: Vec<Vec<usize>> = g.edge_indices()
            .map(|e| {
                let (u, v) = g.edge_endpoints(e).unwrap();
                vec![u.index(), v.index()]
            })
            .filter(|v| (self.get_agents()[v[0]].opinion - self.get_agents()[v[1]].opinion).abs() <= self.get_agents()[v[0]].tolerance)
            .collect();

        let mut h = Graph::new_undirected();
        let node_array: Vec<NodeIndex<u32>> = (0..n).map(|i| h.add_node(i)).collect();

        for e in edgelist {
            h.add_edge(node_array[e[0]], node_array[e[1]], 1);
        }

        max_betweenness_approx(&h)
    }

    fn update_max_betweenness(&mut self) {
        match self.get_topology() {
            TopologyRealization::None => (),
            TopologyRealization::Graph(g) => {
                let cur_b = self.betweenness_active(g);
                if cur_b > self.get_abm_internals().max_betweenness {
                    self.get_mut_abm_internals().max_betweenness = cur_b;
                }
            },
            TopologyRealization::Hypergraph(_) => (),
        }
    }

    fn write_topology_info(&self, file: &mut File) -> std::io::Result<()> {
        let (num_components, lcc_num, lcc, mean_degree, max_betweenness) = match self.get_topology() {
            TopologyRealization::None => (1, 1, self.get_num_agents() as usize, self.get_num_agents() as f64 - 1., 0.),
            TopologyRealization::Graph(g) => {
                let (num, size) = size_largest_connected_component(&g);

                let d = 2. * g.edge_count() as f64 / g.node_count() as f64;

                let betweenness = self.get_abm_internals().max_betweenness;

                (connected_components(&g), num, size, d, betweenness)
            },
            TopologyRealization::Hypergraph(g) => {
                (0, 0, 0, g.mean_deg(), 0.)
            },
        };

        writeln!(file, "{} {} {} {} {}", num_components, lcc_num, lcc, mean_degree, max_betweenness)
        // println!("n {}, c {}, p {}, m {}, num components: {:?}", n, c, p, m, components);
    }

    fn write_state_png(&self, path: &Path) -> std::io::Result<()> {
        let file = File::create(path).unwrap();

        let w = &mut BufWriter::new(file);
        let gradient = colorous::VIRIDIS;

        let n = self.get_num_agents();
        let m = (n as f64).sqrt() as u32;
        assert!(m*m == n);

        let mut encoder = png::Encoder::new(w, m, m); // Width is 2 pixels and height is 1.
        encoder.set_color(png::ColorType::RGB);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();

        let data: Vec<Vec<u8>> = self.get_agents().iter().map(|i| {
            let gr = gradient.eval_continuous(i.opinion as f64);
            vec![gr.r, gr.g, gr.b]
        }).collect();

        let data: Vec<u8> = data.into_iter().flatten().collect();

        writer.write_image_data(&data).unwrap();

        Ok(())
    }
}


pub struct ABMBuilder {
    pub(super) num_agents: u32,

    pub(super) cost_model: CostModel,
    pub(super) resource_model: ResourceModel,
    pub(super) population_model: PopulationModel,
    pub(super) topology_model: TopologyModel,

    pub(super) rng: Pcg64,
}

impl ABMBuilder {
    pub fn new(num_agents: u32) -> ABMBuilder {
        let rng = Pcg64::seed_from_u64(42);
        ABMBuilder {
            num_agents,

            cost_model: CostModel::Free,
            resource_model: ResourceModel::Uniform(0., 1.),
            population_model: PopulationModel::Uniform(0., 1.),
            topology_model: TopologyModel::FullyConnected,

            rng,
        }
    }

    pub fn cost_model(&mut self, cost_model: CostModel) -> &mut ABMBuilder {
        self.cost_model = cost_model;
        self
    }

    pub fn resource_model(&mut self, resource_model: ResourceModel) -> &mut ABMBuilder {
        self.resource_model = resource_model;
        self
    }

    pub fn population_model(&mut self, population_model: PopulationModel) -> &mut ABMBuilder {
        self.population_model = population_model;
        self
    }

    pub fn topology_model(&mut self, topology_model: TopologyModel) -> &mut ABMBuilder {
        self.topology_model = topology_model;
        self
    }

    pub fn seed(&mut self, seed: u64) -> &mut ABMBuilder {
        self.rng = Pcg64::seed_from_u64(seed);
        self
    }
}
