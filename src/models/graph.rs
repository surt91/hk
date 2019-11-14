use petgraph::graph::DiGraph;
use petgraph::algo::{tarjan_scc, condensation};
use petgraph::dot;

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
