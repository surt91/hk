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

pub fn dot2(g: &Graph::<usize, u32, Undirected>) -> String {
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

// we need a function which gives us a new degree vector given an rng
// all parameters mus already be incorporated inside this factory
// So, passing a closure is recommended
pub fn build_cm<R: Rng>(degree_vector_factory: impl Fn(&mut R) -> Vec<usize>, mut rng: &mut R) -> Graph<usize, u32, Undirected> {
    'main: loop {
        let degrees = degree_vector_factory(&mut rng);
        let n = degrees.len();
        let num_stubs: usize = degrees.iter().sum();

        // if there is an uneven number of stubs, creation of a graph is impossible
        if num_stubs % 2 != 0 {
            continue
        }

        let mut g = Graph::new_undirected();
        let nodes: Vec<NodeIndex<u32>> = (0..n).map(|i| g.add_node(i)).collect();

        // generate stubs
        let mut stubs = Vec::new();
        for (i, &d) in degrees.iter().enumerate() {
            for _ in 0..d {
                stubs.push(i)
            }
        }

        // connect stubs
        while !stubs.is_empty() {
            // draw two random stubs and swap the last element to preserve a contiguous vector
            let n = stubs.len();
            let i1 = rng.gen_range(0, n);
            stubs.swap(i1, n-1);
            let stub1 = stubs.pop().unwrap();

            let n = stubs.len();
            let i2 = rng.gen_range(0, n);
            stubs.swap(i2, n-1);
            let stub2 = stubs.pop().unwrap();

            // self loops and multi edges are not allowed
            if stub1 == stub2 || g.find_edge(nodes[stub1], nodes[stub2]) != None {
                // for correct statistics, we have to start from scratch
                continue 'main
            }

            g.add_edge(nodes[stub1], nodes[stub2], 1);
        }

        break g
    }
}

// we need a function which gives us a new degree vector given an rng
// all parameters mus already be incorporated inside this factory
// So, passing a closure is recommended
pub fn build_cm_biased<R: Rng>(degree_vector_factory: impl Fn(&mut R) -> Vec<usize>, mut rng: &mut R) -> Graph<usize, u32, Undirected> {
    'main: loop {
        let degrees = degree_vector_factory(&mut rng);
        let n = degrees.len();
        let num_stubs: usize = degrees.iter().sum();

        // if there is an uneven number of stubs, creation of a graph is impossible
        if num_stubs % 2 != 0 {
            continue
        }

        // if there are more stubs than 2*edges, the graph is too dense to work
        if num_stubs > 2*n*(n-1) {
            continue
        }

        // if there is a node with more stubs than neighbors, double edges are unavoidable
        if *degrees.iter().max().unwrap() > n - 1 {
            continue
        }

        let mut g = Graph::new_undirected();
        let nodes: Vec<NodeIndex<u32>> = (0..n).map(|i| g.add_node(i)).collect();

        // generate stubs
        let mut stubs = Vec::new();
        for (i, &d) in degrees.iter().enumerate() {
            for _ in 0..d {
                stubs.push(i)
            }
        }

        // connect stubs
        while !stubs.is_empty() {
            // draw two random stubs and swap the last element to preserve a contiguous vector
            let n = stubs.len();
            let i1 = rng.gen_range(0, n);
            stubs.swap(i1, n-1);
            let stub1 = stubs.pop().unwrap();

            // draw up to 10 different stubs to try and avoid self loops and multi edges
            // then abort to avoid infinite loops for the case that, e.g., the last two stubs are on the same node
            let mut ctr = 0;
            let stub2 = loop {
                ctr += 1;
                if ctr > 10 {
                    continue 'main
                }

                let n = stubs.len();
                let i2 = rng.gen_range(0, n);
                let s2 = stubs[i2];
                if s2 != stub1 && g.find_edge(nodes[stub1], nodes[s2]) == None {
                    stubs.swap(i2, n-1);
                    break stubs.pop().unwrap();
                }
            };

            g.add_edge(nodes[stub1], nodes[stub2], 1);
        }

        break g
    }
}

pub fn build_lattice(n: usize, next_neighbors: usize) -> Graph<usize, u32, Undirected> {
    let m = (n as f64).sqrt() as usize;
    assert!(m*m == n);
    let mut g = Graph::new_undirected();
    let node_array: Vec<NodeIndex<u32>> = (0..n).map(|i| g.add_node(i)).collect();

    assert!(next_neighbors >= 1);
    assert!(next_neighbors <= 6);

    for i in 0..m {
        for j in 0..m {
            let idx = i*m+j;
            let node = node_array[idx];

            // nearest neighbors
            let bot = (idx + m) % n;
            let right = i*m + (j+1) % m;

            g.add_edge(node, node_array[bot], 1);
            g.add_edge(node, node_array[right], 1);

            // second nearest neighbors
            if next_neighbors >= 2 {
                let bl = (i+1)*m % n + (j-1) % m;
                let br = (i+1)*m % n + (j+1) % m;

                g.add_edge(node, node_array[bl], 1);
                g.add_edge(node, node_array[br], 1);
            }

            // third nearest neighbors
            if next_neighbors >= 3 {
                let bb = (idx + 2*m) % n;
                let rr = i*m + (j+2) % m;

                g.add_edge(node, node_array[bb], 1);
                g.add_edge(node, node_array[rr], 1);
            }

            // fourth nearest neighbors
            if next_neighbors >= 4 {
                let bbl = (i+2)*m % n + (j-1) % m;
                let bbr = (i+2)*m % n + (j+1) % m;
                let brr = (i+1)*m % n + (j+2) % m;
                let bll = (i+1)*m % n + (j-2) % m;

                g.add_edge(node, node_array[bbl], 1);
                g.add_edge(node, node_array[bbr], 1);
                g.add_edge(node, node_array[brr], 1);
                g.add_edge(node, node_array[bll], 1);
            }

            // fifth nearest neighbors
            if next_neighbors >= 5 {
                let bbll = (i+2)*m % n + (j-2) % m;
                let bbrr = (i+2)*m % n + (j+2) % m;

                g.add_edge(node, node_array[bbll], 1);
                g.add_edge(node, node_array[bbrr], 1);
            }

            // sixth nearest neighbors
            if next_neighbors >= 6 {
                let bbb = (i+3)*m % n + j % m;
                let rrr = i*m % n + (j+3) % m;

                g.add_edge(node, node_array[bbb], 1);
                g.add_edge(node, node_array[rrr], 1);
            }
        }
    }

    g
}

pub fn build_ws(n: usize, k: usize, p: f64, rng: &mut impl Rng) -> Graph<usize, u32, Undirected> {
    let mut g = Graph::new_undirected();
    let node_array: Vec<NodeIndex<u32>> = (0..n).map(|i| g.add_node(i)).collect();
    // connect every node to its k right neigbors (periodic)
    for i in 0..n {
        for j in 1..=k {
            // with a probability of p, do not wire to the neighbor, but to a random node
            if rng.gen::<f64>() < p {
                // To avoid multi- or self-loops, we just have to avoid the k left and right
                // neighbors and the node itself. Therefore calculate the random number a
                // bit clever (note that the range is exclusive the right bound)
                let m = rng.gen_range(k+1, n-k);
                g.add_edge(node_array[i], node_array[(i+m) % n], 1);
            } else {
                g.add_edge(node_array[i], node_array[(i+j) % n], 1);
            }
        }
    }

    g
}

pub fn build_ws_lattice(n: usize, k: usize, p: f64, rng: &mut impl Rng) -> Graph<usize, u32, Undirected> {
    let mut g = build_lattice(n, k);
    let node_array: Vec<NodeIndex<u32>> = g.node_indices().collect();
    // iterate over all edges
    // with a probability of p, rewire it, avoiding self- and multi-edges
    for e in g.edge_indices() {
        if rng.gen::<f64>() < p {
            let (u, v) = g.edge_endpoints(e).unwrap();
            g.remove_edge(e);
            let mut vp;
            while {
                vp = node_array[rng.gen_range(0, n)];
                g.contains_edge(u, vp) && vp != u
            } {}
            g.add_edge(u, vp, 1);
        }
    }

    g
}
