/// we want and annealing function, which takes a model
/// i.e., a struct, which has the model trait with the
/// functions
///
/// fn change()
/// fn energy() -> f32
/// fn undo()
///
/// also we want to be flexible in regard to the cooling schedule

use rand::{Rng, SeedableRng};

use ordered_float::OrderedFloat;

use std::ops::Bound::Included;


use super::HegselmannKrause;

pub trait Model {
    fn size(&self) -> usize;
    fn energy(&self) -> f32;
    fn change<R>(&mut self, rng: &mut R) -> (usize, f32) where R: Rng;
    fn undo(&mut self, undo_info: (usize, f32));
    fn notify_sweep(&mut self);
}

impl Model for HegselmannKrause {
    fn energy(&self) -> f32 {
        (0..self.num_agents as usize).map(|idx| {
            let i = &self.agents[idx];

            let (sum, count) = self.opinion_set
                .range((Included(&OrderedFloat(i.opinion-i.tolerance)), Included(&OrderedFloat(i.opinion+i.tolerance))))
                .map(|(j, ctr)| (j.into_inner(), ctr))
                .fold((0., 0), |(sum, count), (j, ctr)| (sum + *ctr as f32 * (j-i.opinion).powf(2.), count + ctr));

            sum / count as f32 + self.eta*(i.opinion - i.initial_opinion).powf(2.)
        }).sum()
    }

    fn change<R>(&mut self, rng: &mut R) -> (usize, f32) where R: Rng {
        let idx: usize = (rng.gen::<f32>() * self.size() as f32) as usize;
        let i = &self.agents[idx];

        let old_x = i.opinion;
        // let mut new_x = old_x + 0.1 * (rng.gen::<f32>() - 0.5);
        // if new_x < 0. {
        //     new_x *= -1.;
        // } else if new_x > 1. {
        //     new_x = 2. - new_x
        // }
        let new_x = rng.gen::<f32>();

        self.update_entry(i.opinion, new_x);

        self.agents[idx].opinion = new_x;

        (idx, old_x)
    }

    fn undo(&mut self, undo_info: (usize, f32)) {
        let (idx, old_x) = undo_info;
        let i = &self.agents[idx];

        self.update_entry(i.opinion, old_x);

        self.agents[idx].opinion = old_x;
    }

    fn size(&self) -> usize {
        self.num_agents as usize
    }

    fn notify_sweep(&mut self) {
        self.add_state_to_density();
        self.time += 1;
    }
}

struct Powerlaw {
    state: f32
}

pub struct Linear {
    count: usize,
    limit: usize,
    start: f32,
    factor: f32
}

impl Linear {
    pub fn new(limit: usize, start: f32) -> Linear {
        Linear {
            count: 0,
            limit,
            start,
            factor: start / limit as f32 ,
        }
    }
}

impl Iterator for Linear {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        self.count += 1;
        let state = self.start - self.factor * self.count as f32;

        if self.count < self.limit {
            Some(state)
        } else {
            None
        }
    }
}

pub struct Exponential {
    count: usize,
    limit: usize,
    state: f32,
    factor: f32
}

impl Exponential {
    pub fn new(limit: usize, start: f32, factor: f32) -> Exponential {
        Exponential {
            count: 0,
            limit,
            state: start,
            factor,
        }
    }
}

impl Iterator for Exponential {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        self.count += 1;
        self.state *= self.factor;

        if self.count < self.limit {
            Some(self.state)
        } else {
            None
        }
    }
}



pub fn anneal<T, S, R>(model: &mut T, schedule: S, mut rng: &mut R)
        where T: Model, S: Iterator<Item = f32>, R: Rng {
    let mut e_before = model.energy();

    for t in schedule {
        let mut tries = 0;
        let mut reject = 0;
        for _ in 0..model.size() {
            let undo_info = model.change(&mut rng);
            let e_after = model.energy();
            tries += 1;
            if ((e_after - e_before) / t).exp() < rng.gen() {
                model.undo(undo_info);
                reject += 1;
            } else {
                e_before = e_after;
            }
            // println!("{} -> {}", e_before, e_after);
        }
        // println!("{}: {:.0}%", t, reject as f32 / tries as f32 * 100.);
        model.notify_sweep();
    }
}
