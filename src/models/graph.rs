use std::collections::HashMap;

use rand::Rng;
use rand::seq::SliceRandom;
use rand_distr::{Binomial, Distribution};

use petgraph::graph::{DiGraph, UnGraph, Graph, NodeIndex};
use petgraph::algo::{tarjan_scc, condensation};
use petgraph::dot;
use petgraph::unionfind::UnionFind;
use petgraph::Undirected;

use super::HegselmannKrause;
use super::CostModel;
// use super::hk_vanilla::HKAgent;

pub fn from_hk(hk: &HegselmannKrause) -> DiGraph::<i32, f32> {
    let mut g = DiGraph::<i32, f32>::new();

    let nodes: Vec<_> = hk.agents.iter().enumerate()
        .map(|(n, _)| {
            g.add_node(n as i32)
        })
        .collect();
    let neighbors: Vec<Vec<_>> = hk.agents.iter().enumerate()
        .map(|(n, a)| {
            hk.agents.iter().enumerate()
                // our neighbors are all agents that we interact with
                // if we do not have any resources left, we are isolated
                .filter(|(_, a2)| (a.resources > 0. || hk.cost_model == CostModel::Free) && (a2.opinion - a.opinion).abs() < a.tolerance)
                .map(|(m, _)| (nodes[n], nodes[m], 1.))
                .collect()
        })
        .collect();

    for n in neighbors {
        g.extend_with_edges(&n);
    }

    g
}

pub fn clustersizes(g: &DiGraph::<i32, f32>) -> Vec<usize> {
    tarjan_scc(&g).iter().map(|v| v.len()).collect()
}

pub fn dot(g: &DiGraph::<i32, f32>) -> String {
    dot::Dot::with_config(&g, &[dot::Config::EdgeNoLabel]).to_string()
}

pub fn condense(g: &DiGraph::<i32, f32>) -> DiGraph::<i32, f32> {
    condensation(g.clone(), true).map(|_, x| x.len() as i32, |_, x| *x)
}

pub fn size_largest_connected_component(g : &UnGraph<usize, u32>) -> (usize, usize) {
    let mut uf = UnionFind::new(g.node_count());
    let mut reverse_lookup: HashMap<NodeIndex<_>, usize> = HashMap::new();
    for (i, n1) in g.node_indices().enumerate() {
        reverse_lookup.insert(n1, i);
    }
    for (i, n1) in g.node_indices().enumerate() {
        for n2 in g.neighbors(n1) {
            let j = reverse_lookup[&n2];
            uf.union(i, j);
        }
    }

    let mut counter = HashMap::new();
    let labels = uf.into_labeling();
    for l in labels {
        *counter.entry(l).or_insert(0) += 1;
    }

    use petgraph::algo::connected_components;
    assert_eq!(connected_components(&g), counter.values().len());

    (counter.values().len(), *counter.values().max().unwrap())
}

pub fn build_er(n: usize, c: f64, mut rng: &mut impl Rng) -> Graph<usize, u32, Undirected> {
    let mut g = Graph::new_undirected();
    let node_array: Vec<NodeIndex<u32>> = (0..n).map(|i| g.add_node(i)).collect();
    // draw `m' from binomial distribution how many edges the ER should have
    let p = c / n as f64;
    let binom = Binomial::new((n*(n-1) / 2) as u64, p).unwrap();
    let m = binom.sample(&mut rng);
    // draw `m' unconnected pairs of agents and connect them
    let mut ctr = 0;
    while ctr < m {
        let idx1 = rng.gen_range(0, n);
        let node1 = node_array[idx1];
        let idx2 = rng.gen_range(0, n);
        let node2 = node_array[idx2];

        if idx1 != idx2 && g.find_edge(node1, node2) == None {
            g.add_edge(node1, node2, 1);
            ctr += 1;
        }
    }

    g
}

pub fn build_ba(n: usize, degree: f64, m0: usize, mut rng: &mut impl Rng) -> Graph<usize, u32, Undirected> {
    let m = degree / 2.;
    let mut g = Graph::new_undirected();
    let nodes: Vec<NodeIndex<u32>> = (0..n).map(|i| g.add_node(i)).collect();

    let mut weighted_node_list: Vec<NodeIndex<u32>> = Vec::new();

    // starting core
    for i in 0..m0 {
        for j in 0..i {
            let n1 = nodes[i];
            let n2 = nodes[j];
            g.add_edge(n1, n2, 1);
            weighted_node_list.push(n1);
            weighted_node_list.push(n2);
        }
    }

    for &i in nodes.iter().skip(m0) {
        // add new node and connect to `m` nodes
        for j in 0..m.ceil() as usize {
            // if we have a fractional m, only add a node with the probability of the fractional part
            if (m - j as f64) < 1. && rng.gen::<f64>() > m - j as f64 {
                continue
            }

            let neighbor = loop {
                let neighbor = *weighted_node_list.choose(&mut rng).unwrap();

                // check for double edges and self loops
                if neighbor != i && g.find_edge(neighbor, i) == None {
                    break neighbor;
                }
            };

            weighted_node_list.push(neighbor);
            weighted_node_list.push(i);
            g.add_edge(i, neighbor, 1);
        }
    }

    g
}