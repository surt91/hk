use std::collections::BTreeSet as Set;
use itertools::Itertools;
use std::iter::FromIterator;

use rand::Rng;
use rand_distr::{Binomial, Normal, Distribution};
use rand::seq::SliceRandom;

use petgraph::graph::{UnGraph, Graph, NodeIndex};

#[derive(Clone, Debug)]
pub struct Hypergraph {
    pub factor_graph: UnGraph<usize, u32>,
    // helper vectors conatining lists of
    // all nodes of the factor graph representing nodes of the hypergraph
    pub node_nodes: Vec<NodeIndex<u32>>,
    // and all nodes of the factor graph representing edges of the hypergraph
    pub edge_nodes: Vec<NodeIndex<u32>>,
    pub edge_set: Set<Set<usize>>,
    pub ctr: usize,
}

impl Hypergraph {
    pub fn mean_deg(&self) -> f64 {
        self.edge_set.iter().map(|s| s.len()).sum::<usize>() as f64 / self.node_nodes.len() as f64
    }

    pub fn degrees(&self) -> Vec<usize> {
        self.factor_graph.node_indices()
            .filter(|i| i.index() < self.node_nodes.len())  // do not count nodes corresponding to hyperedges
            .map(|i| self.factor_graph.neighbors(i).count())
            .collect()
    }


    pub fn add_er_hyperdeges(&mut self, c: f64, k: usize, mut rng: &mut impl Rng) {
        // roll dice how many edges there should be
        let g = &mut self.factor_graph;
        let n = self.node_nodes.len();

        // roll dice how many edges there should be
        let num = n_choose_k(n, k);
        let p = c / (k as f64 * num as f64 / n as f64);
        let np = c / (k as f64 / n as f64);
        // if `num` is too large, use a Gaussian approximation, otherwise use binomial
        let num_edges = if np < 1e3 {
            let binom = Binomial::new(num as u64, p).unwrap();
            binom.sample(&mut rng) as usize
        } else {
            // assert that our error will be small
            assert!(np > 9.*(1.-p));
            let gauss = Normal::new(np, (np*(1.-p)).sqrt()).unwrap();
            let rn = gauss.sample(&mut rng);
            // this should not happen during the lifetime of our universe
            // so we do not need to handle it.
            assert!(rn > 0.);
            rn as usize
        };

        let start_ctr = self.ctr;

        // draw `num_edges` many `k`-tuples of nodes and add them as edges (=factor-nodes)
        while self.ctr < num_edges + start_ctr {
            let idx: Set<usize> = (0..k).map(|_| rng.gen_range(0, n)).collect();

            if idx.len() == k && !self.edge_set.contains(&idx) {
                let j = g.add_node(n + self.ctr);
                self.edge_nodes.push(j);
                for i in &idx {
                    g.add_edge(self.node_nodes[*i], j, 1);
                }
                self.ctr += 1;
                self.edge_set.insert(idx);
            }
        }
    }
}

fn n_choose_k(n: usize, k: usize) -> usize {
    (1..=k).map(|i| (n+1-i) as f64 / i as f64).product::<f64>() as usize
}

pub fn build_hyper_empty(n: usize) -> Hypergraph {
    let mut g = Graph::new_undirected();
    let node_array: Vec<NodeIndex<u32>> = (0..n).map(|i| g.add_node(i)).collect();
    let edge_array: Vec<NodeIndex<u32>> = Vec::new();
    let edge_set: Set<Set<usize>> = Set::new();

    Hypergraph {
        factor_graph: g,
        node_nodes: node_array,
        edge_nodes: edge_array,
        edge_set,
        ctr: 0,
    }
}

pub fn build_hyper_uniform_er(n: usize, c: f64, k: usize, mut rng: &mut impl Rng) -> Hypergraph {
    // ensure that the edges connect at least two nodes
    // note that k = 2 is just a regular graph, which might be handy for testing
    assert!(k > 1);

    let mut h = build_hyper_empty(n);

    h.add_er_hyperdeges(c, k, &mut rng);

    h
}

pub fn build_hyper_uniform_ba(n: usize, m: usize, k: usize, mut rng: &mut impl Rng) -> Hypergraph {
    // ensure that the edges connect at least two nodes
    // note that k = 2 is just a regular graph, which might be handy for testing
    assert!(k > 1);

    let mut g = Graph::new_undirected();
    let node_array: Vec<NodeIndex<u32>> = (0..n).map(|i| g.add_node(i)).collect();
    let mut edge_array: Vec<NodeIndex<u32>> = Vec::new();
    let mut edge_set: Set<Set<usize>> = Set::new();

    let mut weighted_node_list: Vec<NodeIndex<u32>> = Vec::new();
    let m0 = std::cmp::max(m-1, k);

    let mut ctr = 0;

    // build starting core of m0 = m-1 nodes
    // we therefore choose all subsets of size k
    for subset in (0..m0).combinations(k) {
        let he = g.add_node(n + ctr);
        edge_array.push(he);

        for &i in &subset {
            g.add_edge(node_array[i], he, 1);
            weighted_node_list.push(node_array[i]);
        }
        edge_set.insert(subset.into_iter().collect::<Set<usize>>());

        ctr += 1;
    }

    // preferential attachment
    for &i in node_array.iter().skip(m0) {
        // add new node and connect it with `m` hyperedges to (k-1)m other nodes
        for _ in 0..m {
            let (neighbors, edge) = loop {
                let hyper_edge_members = weighted_node_list
                    .choose_multiple(&mut rng, k-1)
                    .cloned()
                    .chain(std::iter::once(i))
                    .collect::<Set<NodeIndex<u32>>>();

                // ensure members are unique and
                // check for double edges and self loops
                let edge = hyper_edge_members.iter()
                    .map(|x| x.index())
                    .collect::<Set<usize>>();

                if hyper_edge_members.len() == k
                    && !edge_set.contains(&edge) {
                    break (hyper_edge_members, edge);
                }
            };

            // add the edge
            // and update the weights
            let he = g.add_node(n + ctr);
            edge_array.push(he);
            for &i in &neighbors {
                g.add_edge(i, he, 1);
                weighted_node_list.push(i);
            }
            edge_set.insert(edge);
            ctr += 1;
        }
    }

    Hypergraph {
        factor_graph: g,
        node_nodes: node_array,
        edge_nodes: edge_array,
        edge_set,
        ctr,
    }
}

pub fn build_hyper_gaussian_er(n: usize, c: f64, mu: f64, sigma: f64, mut rng: &mut impl Rng) -> Hypergraph {
    // empty hypergraph
    let mut h = build_hyper_empty(n);

    // distribute the order of the edges like c * Gaussian(mu, sigma), but rounded and bounded by 2 and N-1
    for k in 2..=n {
        let c_k = c/(sigma*2.*std::f64::consts::PI) * (-1./2.*(k as f64 - mu).powi(2)/sigma.powi(2)).exp();
        h.add_er_hyperdeges(c_k, k, &mut rng);
    }

    h
}

pub fn convert_to_simplical_complex(g: &Hypergraph) -> Hypergraph {
    // currently I copy, maybe I should just modify the graph in place...
    let mut new = g.clone();
    let mut ctr = g.ctr;

    // iterate over edges and insert edges between all combinations of all lengths
    for e in &g.edge_set {
        for k in 2..e.len() {
            for comb in e.iter().cloned().combinations(k) {
                let sub_edge = Set::from_iter(comb.iter().cloned());
                if ! new.edge_set.contains(&sub_edge) {
                    ctr += 1;
                    let idx = new.factor_graph.add_node(ctr);
                    for i in &sub_edge {
                        new.factor_graph.add_edge(new.node_nodes[*i], idx, 1);
                    }
                    new.edge_nodes.push(idx);
                    new.edge_set.insert(sub_edge);
                }
            }
        }
    }

    new.ctr = ctr;
    new
}