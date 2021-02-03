
use std::path::Path;

use rand::{Rng, SeedableRng};
use rand_distr::{Normal, Pareto, Distribution};
use rand_pcg::Pcg64;

#[cfg(feature = "graphtool")]
use inline_python::{python,Context};

// TODO: integrate Builder here with two


use petgraph::graph::Graph;
use petgraph::Undirected;
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

pub const EPS: f32 = 2e-3;

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

pub trait ABM {

    // getter methods to access fields
    fn get_population_model(&self) -> PopulationModel;
    fn get_topology_model(&self) -> TopologyModel;
    fn get_resource_model(&self) -> ResourceModel;
    fn get_agents(&self) -> &Vec<Agent>;
    fn get_rng(&mut self) -> &mut Pcg64;

    fn gen_init_opinion(&mut self) -> f32 {
        match self.get_population_model() {
            PopulationModel::Bridgehead(x_init, x_spread, frac, _eps_init, _eps_spread, _eps_min, _eps_max) => {
                if self.get_rng().gen::<f32>() > frac {
                    self.get_rng().gen()
                } else {
                    stretch(self.get_rng().gen(), x_init-x_spread, x_init+x_spread)
                }
            },
            _ => self.get_rng().gen(),
        }
    }

    fn gen_init_tolerance(&mut self) -> f32 {
        match self.get_population_model() {
            PopulationModel::Uniform(min, max) => stretch(self.get_rng().gen(), min, max),
            PopulationModel::Bimodal(first, second) => if self.get_rng().gen::<f32>() < 0.5 {first} else {second},
            PopulationModel::Bridgehead(_x_init, _x_spread, frac, eps_init, eps_spread, eps_min, eps_max) => {
                if self.get_rng().gen::<f32>() > frac {
                    stretch(self.get_rng().gen(), eps_min, eps_max)
                } else {
                    stretch(self.get_rng().gen(), eps_init-eps_spread, eps_init+eps_spread)
                }
            },
            PopulationModel::Gaussian(mean, sdev) => {
                let gauss = Normal::new(mean, sdev).unwrap();
                // draw gaussian RN until you get one in range
                loop {
                    let x = gauss.sample(&mut self.get_rng());
                    if x <= 1. && x >= 0. {
                        break x
                    }
                }
            },
            PopulationModel::PowerLaw(min, exponent) => {
                let pareto = Pareto::new(min, exponent - 1.).unwrap();
                pareto.sample(&mut self.get_rng())
            }
            PopulationModel::PowerLawBound(min, max, exponent) => {
                // http://mathworld.wolfram.com/RandomNumber.html
                fn powerlaw(y: f32, low: f32, high: f32, alpha: f32) -> f32 {
                    ((high.powf(alpha+1.) - low.powf(alpha+1.))*y + low.powf(alpha+1.)).powf(1./(alpha+1.))
                }
                powerlaw(self.get_rng().gen(), min, max, exponent)
            }
        }
    }

    fn gen_init_topology(&mut self) -> TopologyRealization {
        match &self.get_topology_model() {
            TopologyModel::FullyConnected => TopologyRealization::None,
            TopologyModel::ER(c) => {
                let n = self.get_agents().len();
                let g = loop {
                    let tmp = build_er(n, *c as f64, &mut self.get_rng());
                    if size_largest_connected_component(&tmp).0 == 1 {
                        break tmp
                    }
                };

                TopologyRealization::Graph(g)
            },
            TopologyModel::BA(degree, m0) => {
                let n = self.get_agents().len();
                let g = build_ba(n, *degree, *m0, &mut self.get_rng());

                TopologyRealization::Graph(g)
            },
            TopologyModel::CMBiased(degree_dist) => {
                let g = loop {
                    let tmp = build_cm_biased(move |r| degree_dist.clone().gen(r), &mut self.get_rng());
                    if size_largest_connected_component(&tmp).0 == 1 {
                        break tmp
                    }
                };

                TopologyRealization::Graph(g)
            },
            TopologyModel::CM(degree_dist) => {
                let g = loop {
                    let tmp = build_cm(move |r| degree_dist.clone().gen(r), &mut self.get_rng());
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
                // let g = build_ws(n, *neighbors, *rewiring, &mut self.rng);
                let g = loop {
                    let tmp = build_ws(n, *neighbors, *rewiring, &mut self.get_rng());
                    if size_largest_connected_component(&tmp).0 == 1 {
                        break tmp
                    }
                };

                TopologyRealization::Graph(g)
            },
            TopologyModel::WSlat(neighbors, rewiring) => {
                let n = self.get_agents().len();
                // let g = build_ws(n, *neighbors, *rewiring, &mut self.get_rng());
                let g = loop {
                    let tmp = build_ws_lattice(n, *neighbors, *rewiring, &mut self.get_rng());
                    if size_largest_connected_component(&tmp).0 == 1 {
                        break tmp
                    }
                };

                TopologyRealization::Graph(g)
            },
            TopologyModel::BAT(degree, mt) => {
                let n = self.get_agents().len();
                let m0 = (*degree as f64 / 2.).ceil() as usize + mt.ceil() as usize;
                let g = build_ba_with_clustering(n, *degree, m0, *mt, &mut self.get_rng());

                TopologyRealization::Graph(g)
            },
            TopologyModel::HyperER(c, k) => {
                let n = self.get_agents().len();
                // TODO: maybe ensure connectedness
                let g = build_hyper_uniform_er(n, *c, *k, &mut self.get_rng());

                TopologyRealization::Hypergraph(g)
            },
            TopologyModel::HyperERSC(c, k) => {
                let n = self.get_agents().len();
                // TODO: maybe ensure connectedness
                let g = convert_to_simplical_complex(&build_hyper_uniform_er(n, *c, *k, &mut self.get_rng()));

                TopologyRealization::Hypergraph(g)
            },
            TopologyModel::HyperBA(m, k) => {
                let n = self.get_agents().len();
                let g = build_hyper_uniform_ba(n, *m, *k, &mut self.get_rng());

                TopologyRealization::Hypergraph(g)
            },
            TopologyModel::HyperER2(c1, c2, k1, k2) => {
                let n = self.get_agents().len();
                let mut g = build_hyper_uniform_er(n, *c1, *k1, &mut self.get_rng());
                g.add_er_hyperdeges(*c2, *k2, &mut self.get_rng());

                TopologyRealization::Hypergraph(g)
            },
        }
    }

    fn gen_init_resources(&mut self, confidence: f32) -> f32 {
        match self.get_resource_model() {
            ResourceModel::Uniform(l, u) => stretch(self.get_rng().gen(), l, u),
            ResourceModel::Pareto(x0, a) => {
                let pareto = Pareto::new(x0, a - 1.).unwrap();
                pareto.sample(&mut self.get_rng())
            },
            ResourceModel::Proportional(a, offset) => (confidence - offset) * a,
            ResourceModel::Antiproportional(a, offset) => 1. - (confidence - offset) * a,
            ResourceModel::HalfGauss(sigma) => {
                let gauss = Normal::new(0., sigma).unwrap();
                gauss.sample(&mut self.get_rng()).abs()
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

        // TODO: hypergraph
        let edgelist: Vec<Vec<usize>>;
        let colors: Vec<Vec<f64>>;

        match &self.topology {
            TopologyRealization::Graph(g) => {
                colors = self.agents.iter().map(|i| {
                    let gr = gradient.eval_continuous(i.opinion as f64);
                    vec![gr.r as f64 / 255., gr.g as f64 / 255., gr.b as f64 / 255., 1.]
                }).collect();
                edgelist = if active {
                    g.edge_indices()
                        .map(|e| {
                            let (u, v) = g.edge_endpoints(e).unwrap();
                            vec![u.index(), v.index()]
                        })
                        .filter(|v| (self.agents[v[0]].opinion - self.agents[v[1]].opinion).abs() <= self.agents[v[0]].tolerance)
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
                colors = self.agents.iter().map(|i| {
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
                            let opin = g.neighbors(e).map(|n| OrderedFloat(self.agents[*g.node_weight(n).unwrap()].opinion));
                            let opix = g.neighbors(e).map(|n| OrderedFloat(self.agents[*g.node_weight(n).unwrap()].opinion));
                            let tol = g.neighbors(e).map(|n| OrderedFloat(self.agents[*g.node_weight(n).unwrap()].tolerance));
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
}