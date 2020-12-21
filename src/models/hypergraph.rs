use std::collections::BTreeSet as Set;

use rand::Rng;
use rand_distr::{Binomial, Distribution};

use petgraph::graph::{UnGraph, Graph, NodeIndex};

use super::graph;

#[derive(Clone, Debug)]
pub struct Hypergraph {
    factor_graph: UnGraph<usize, u32>,
    // helper vectors conatining lists of
    // all nodes of the factor graph representing nodes of the hypergraph
    node_nodes: Vec<NodeIndex<u32>>,
    // and all nodes of the factor graph representing edges of the hypergraph
    edge_nodes: Vec<NodeIndex<u32>>,
    edge_set: Set<Set<usize>>,
}

impl Hypergraph {

}


pub fn build_hyper_uniform_er(n: usize, c: f64, k: usize, mut rng: &mut impl Rng) -> Hypergraph {
    // ensure that the edges connect at least two nodes
    // note that k = 2 is just a regular graph, which might be handy for testing
    assert!(k > 1);

    let mut g = Graph::new_undirected();
    let node_array: Vec<NodeIndex<u32>> = (0..n).map(|i| g.add_node(i)).collect();
    let mut edge_array: Vec<NodeIndex<u32>> = Vec::new();
    let mut edge_set: Set<Set<usize>> = Set::new();

    // roll dice how many edges there should be
    let p = c / n as f64;
    let binom = Binomial::new((n*(n-1) / 2) as u64, p).unwrap();
    let m = binom.sample(&mut rng) as usize;

    // draw m many k-tuples of nodes and add them as edges (=factor-nodes)
    let mut ctr = 0;
    while ctr < m {
        let idx: Set<usize> = (0..k).map(|_| rng.gen_range(0, n)).collect();

        if idx.len() != k && !edge_set.contains(&idx) {
            let j = g.add_node(n + ctr);
            edge_array.push(j);
            for i in &idx {
                g.add_edge(node_array[*i], j, 1);

            }
            ctr += 1;
        }
        edge_set.insert(idx);
    }

    Hypergraph {
        factor_graph: g,
        node_nodes: node_array,
        edge_nodes: edge_array,
        edge_set
    }
}