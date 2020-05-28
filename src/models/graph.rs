use std::collections::HashMap;

use rand::Rng;
use rand_distr::Binomial;

use petgraph::graph::{DiGraph, UnGraph, Graph};
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
    for (i, n1) in g.node_indices().enumerate() {
        for (j, _n2) in g.neighbors(n1).enumerate() {
            uf.union(i, j);
        }
    }

    let mut counter = HashMap::new();
    let labels = uf.into_labeling();
    for l in labels {
        *counter.entry(l).or_insert(0) += 1;
    }

    (counter.values().len(), *counter.values().max().unwrap())
}
