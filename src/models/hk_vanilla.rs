use std::collections::HashSet;
use std::collections::BTreeMap;
use std::ops::Bound::Included;
use std::fmt;
use std::fs::File;
use std::io::prelude::*;

use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use itertools::Itertools;

use ordered_float::OrderedFloat;

const EPS: f32 = 1e-6;

#[derive(Clone, Debug)]
struct HKAgent {
    opinion: f32,
    tolerance: f32,
}

impl HKAgent {
    fn new(opinion: f32, tolerance: f32) -> HKAgent {
        HKAgent {
            opinion,
            tolerance,
        }
    }
}

impl PartialEq for HKAgent {
    fn eq(&self, other: &HKAgent) -> bool {
        (self.opinion - other.opinion).abs() < EPS
            && (self.tolerance - other.tolerance).abs() < EPS
    }
}

#[derive(Clone, Debug)]
struct Cell {
    total: f32,
    count: u32,
    agent_ids: HashSet<usize>,
}

impl Cell {
    fn new() -> Cell {
        Cell {
            total: 0.,
            count: 0,
            agent_ids: HashSet::new(),
        }
    }

    fn insert(&mut self, agent_id: usize, val: f32) {
        assert!(!self.agent_ids.contains(&agent_id));
        self.agent_ids.insert(agent_id);
        self.total += val;
        self.count += 1;
    }

    fn remove(&mut self, agent_id: usize, val: f32) {
        assert!(self.agent_ids.contains(&agent_id));
        assert!(self.count > 0);
        self.agent_ids.remove(&agent_id);
        self.total -= val;
        self.count -= 1;
    }
}

struct CellList {
    cells: Vec<Cell>,
    borders: Vec<OrderedFloat<f32>>,
}

impl CellList {
    fn new(n: u32, lower: f32, upper: f32) -> CellList {
        let width = (upper - lower) / n as f32;
        let borders = (0..(n+1)).map(|i| width * i as f32)
            .map(|x| OrderedFloat(x))
            .collect();
        CellList {
            cells: vec![Cell::new(); n as usize],
            borders
        }
    }

    fn get_cell_idx(&self, val: f32) -> usize {
        // we also use this function with added/subtracted tolerance
        // so we have to handle values outside of [0,1]

        if val <= 0. {
            return 0
        }
        if val >= 1. {
            return self.cells.len() - 1
        }

        let val = OrderedFloat(val);
        let idx = match self.borders.binary_search(&val) {
            Ok(x) => x,
            Err(x) => x,
        } - 1;

        idx
    }

    fn fill(&mut self, agents: &[HKAgent]) {
        for (n, a) in agents.iter().enumerate() {
            let idx = self.get_cell_idx(a.opinion);
            self.cells[idx].insert(n, a.opinion);
        }
    }

    fn change(&mut self, agent_id: usize, val_old: f32, val_new: f32) {
        let idx_old = self.get_cell_idx(val_old);
        let idx_new = self.get_cell_idx(val_new);

        // often we will stay in the same cell
        if idx_old == idx_new {
            self.cells[idx_old].total += val_new - val_old;
            return
        }

        self.cells[idx_old].remove(agent_id, val_old);
        self.cells[idx_new].insert(agent_id, val_new);
    }
}

pub struct HegselmannKrause {
    num_agents: u32,
    agents: Vec<HKAgent>,

    opinion_set: BTreeMap<OrderedFloat<f32>, u32>,
    cell_list: CellList,
    // we need many, good (but not crypto) random numbers
    // we will use here the pcg generator
    rng: Pcg64,
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
    pub fn new(n: u32, min_tolerance: f32, max_tolerance: f32, seed: u64) -> HegselmannKrause {
        let mut rng = Pcg64::seed_from_u64(seed);
        let stretch = |x: f32| x*(max_tolerance-min_tolerance)+min_tolerance;
        let agents: Vec<HKAgent> = (0..n).map(|_| HKAgent::new(
            rng.gen(),
            stretch(rng.gen())
        )).collect();

        // datastructure for `step_bisect`
        let mut opinion_set = BTreeMap::new();
        for i in agents.iter() {
            opinion_set.insert(OrderedFloat(i.opinion), 1);
        }
        assert!(opinion_set.len() == n as usize);

        // datastructure for `step_cells`
        let mut cell_list = CellList::new((n as f32).sqrt().round() as u32, 0., 1.);
        cell_list.fill(&agents);

        HegselmannKrause {
            num_agents: n,
            agents,
            opinion_set,
            cell_list,
            rng,
        }
    }

    pub fn step_naive(&mut self) {
        // get a random agent
        let idx = self.rng.gen_range(0, self.num_agents) as usize;
        let i = &self.agents[idx];

        let (sum, count) = self.agents.iter()
            .map(|j| j.opinion)
            .filter(|j| (i.opinion - j).abs() < i.tolerance)
            .fold((0., 0), |(sum, count), i| (sum + i, count + 1));

        self.agents[idx].opinion = sum / count as f32;
    }

    pub fn step_bisect(&mut self) {
        // get a random agent
        let idx = self.rng.gen_range(0, self.num_agents) as usize;
        let i = &self.agents[idx];

        let (sum, count) = self.opinion_set
            .range((Included(&OrderedFloat(i.opinion-i.tolerance)), Included(&OrderedFloat(i.opinion+i.tolerance))))
            .map(|(j, ctr)| (j.into_inner(), ctr))
            .fold((0., 0), |(sum, count), (j, ctr)| (sum + *ctr as f32 * j, count + ctr));

        let new_opinion = sum / count as f32;

        // often, nothing changes -> optimize for this converged case
        if i.opinion == new_opinion {
            return
        }

        *self.opinion_set.entry(OrderedFloat(i.opinion)).or_insert_with(|| panic!("todo")) -= 1;
        if self.opinion_set[&OrderedFloat(i.opinion)] == 0 {
            self.opinion_set.remove(&OrderedFloat(i.opinion));
        }
        *self.opinion_set.entry(OrderedFloat(new_opinion)).or_insert(0) += 1;

        self.agents[idx].opinion = new_opinion;
    }

    pub fn step_cells(&mut self) {
        let idx = self.rng.gen_range(0, self.num_agents) as usize;
        let i = &self.agents[idx];

        let mut sum = 0.;
        let mut count = 0;

        let upper_border = self.cell_list.get_cell_idx(i.opinion + i.tolerance);
        let lower_border = self.cell_list.get_cell_idx(i.opinion - i.tolerance);

        if upper_border - lower_border > 1 {
            let (sum_bulk, count_bulk) = self.cell_list.cells[lower_border+1..upper_border].iter()
                .map(|c| (c.total, c.count))
                .fold((0., 0), |(sum, total_count), (bin_val, bin_count)| (sum + bin_val, total_count + bin_count));
            sum += sum_bulk;
            count += count_bulk;
        }

        for id in &self.cell_list.cells[upper_border].agent_ids {
            let j = &self.agents[*id];
            // if upper == lower, may not count those below the lower tolerance
            if i.opinion + i.tolerance >= j.opinion && i.opinion - i.tolerance <= j.opinion {
                sum += j.opinion;
                count += 1;
            }
        }
        // if upper == lower, we would count these agents twice
        if upper_border != lower_border {
            for id in &self.cell_list.cells[lower_border].agent_ids {
                let j = &self.agents[*id];
                if i.opinion - i.tolerance <= j.opinion {
                    sum += j.opinion;
                    count += 1;
                }
            }
        }

        let new_opinion = sum / count as f32;

        self.cell_list.change(idx, i.opinion, new_opinion);

        self.agents[idx].opinion = new_opinion;
    }

    pub fn sweep(&mut self) {
        for _ in 0..self.num_agents {
            // self.step_naive();
            self.step_bisect();
            // self.step_cells();
        }
    }

    pub fn write_state(&self, file: &mut File) -> std::io::Result<()> {
        let string_list = self.agents.iter()
            .map(|j| j.opinion.to_string())
            .join(" ");
        write!(file, "{}\n", string_list)
    }

    pub fn write_gp(&self, file: &mut File, outfilename: &str) -> std::io::Result<()> {
        write!(file, "set terminal pngcairo\n")?;
        write!(file, "set output '{}.png'\n", outfilename)?;
        write!(file, "set xl 't'\n")?;
        write!(file, "set yl 'x_i'\n")?;
        write!(file, "p '{}' u 0:1 w l not, ", outfilename)?;

        let string_list = (2..self.num_agents)
            .map(|j| format!("'' u 0:{} w l not,", j))
            .join(" ");
        write!(file, "{}", string_list)
    }
}
