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
    fn energy_incremental(
        &mut self,
        current: usize,
        old: f32,
        new: f32,
    ) -> f32;
    fn init_ji(&mut self);
    fn change<R>(&mut self, rng: &mut R) -> (usize, f32, f32) where R: Rng;
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

    fn init_ji(&mut self) {
        self.ji = (0..self.num_agents as usize).map(|idx| {
            let i = &self.agents[idx];

            let (sum, count) = self.opinion_set
                .range((Included(&OrderedFloat(i.opinion-i.tolerance)), Included(&OrderedFloat(i.opinion+i.tolerance))))
                .map(|(j, ctr)| (j.into_inner(), ctr))
                .fold((0., 0), |(sum, count), (j, ctr)| (sum + *ctr as f32 * (j-i.opinion).powf(2.), count + ctr));

            sum / count as f32 + self.eta*(i.opinion - i.initial_opinion).powf(2.)
        }).collect();

        self.jin = (0..self.num_agents as usize).map(|idx| {
            let i = &self.agents[idx];

            self.opinion_set
                .range((Included(&OrderedFloat(i.opinion-i.tolerance)), Included(&OrderedFloat(i.opinion+i.tolerance))))
                .map(|(_, ctr)| *ctr as i32)
                .sum()
        }).collect();
    }

    // if we cache the single energies ji and the number of neighbors jin
    // we can update J in linear time instead of quadratic (or cubic?)
    // given the old and new opinion of a changed agent
    fn energy_incremental(
        &mut self,
        current: usize,
        old: f32,
        new: f32,
    ) -> f32 {
        self.ji[current] = 0.;
        self.jin[current] = 1;

        for (idx, i) in self.agents.iter().enumerate() {
            // also we influenced ourself before, so we have to remove ourself in any case
            if idx == current {
                continue
            }

            // first we need to remove the cost of being too far away from the start
            self.ji[idx] -= (i.opinion - i.initial_opinion).powf(2.) * self.eta;

            // if we had influence before, remove it
            if (i.opinion - old).abs() < i.tolerance {
                self.ji[idx] -= (i.opinion - old).powf(2.) / self.jin[idx] as f32;
                self.ji[idx] *= self.jin[idx] as f32 / (self.jin[idx] - 1) as f32;
                self.jin[idx] -= 1;
            }

            // if we have influence now, add it
            if (i.opinion - new).abs() < i.tolerance {
                self.jin[idx] += 1;
                self.ji[idx] *= (self.jin[idx] - 1) as f32 / self.jin[idx] as f32;
                self.ji[idx] += (i.opinion - new).powf(2.) / self.jin[idx] as f32;
            }

            // and calculate the energy for the agent which changes
            if (i.opinion - new).abs() < self.agents[current].tolerance {
                self.ji[current] += (i.opinion - new).powf(2.);
                self.jin[current] += 1;
            }

            // in the end we have to reintroduce the cost of being too far away from the start
            self.ji[idx] += (i.opinion - i.initial_opinion).powf(2.) * self.eta;
        }


        self.ji[current] /= self.jin[current] as f32;
        self.ji[current] += self.eta*(new - self.agents[current].initial_opinion).powf(2.);

        self.ji.iter().sum::<f32>()
    }

    fn change<R>(&mut self, rng: &mut R) -> (usize, f32, f32) where R: Rng {
        let idx: usize = (rng.gen::<f32>() * self.size() as f32) as usize;

        let old_x = self.agents[idx].opinion;
        // let mut new_x = old_x + 0.1 * (rng.gen::<f32>() - 0.5);
        // if new_x < 0. {
        //     new_x *= -1.;
        // } else if new_x > 1. {
        //     new_x = 2. - new_x
        // }
        let new_x = rng.gen::<f32>();

        self.update_entry(old_x, new_x);

        self.agents[idx].opinion = new_x;

        (idx, old_x, new_x)
    }

    fn undo(&mut self, undo_info: (usize, f32)) {
        let (idx, old_x) = undo_info;
        let new_x = self.agents[idx].opinion;

        self.update_entry(new_x, old_x);

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
    model.init_ji();

    for t in schedule {
        // let mut tries = 0;
        // let mut reject = 0;
        for _ in 0..model.size() {
            let (idx, old, new) = model.change(&mut rng);
            let undo_info = (idx, old);
            // let e_after = model.energy();
            let e_after = model.energy_incremental(idx, old, new);
            // tries += 1;
            if (-(e_after - e_before) / t).exp() < rng.gen() {
                model.energy_incremental(idx, new, old);
                model.undo(undo_info);
                // reject += 1;
            } else {
                e_before = e_after;
            }
        }
        // println!("{}: {:.0}%", t, reject as f32 / tries as f32 * 100.);
        model.notify_sweep();
    }
}
