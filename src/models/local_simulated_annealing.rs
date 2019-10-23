/// we want and annealing function, which takes a model
/// i.e., a struct, which has the model trait with the
/// functions
///
/// fn change()
/// fn energy() -> f32
/// fn undo()
///
/// also we want to be flexible in regard to the cooling schedule

use rand::Rng;

use ordered_float::OrderedFloat;

use std::ops::Bound::Included;


use super::HegselmannKrause;

pub trait LocalModel {
    fn size(&self) -> usize;
    fn local_energy(&self, idx: usize) -> f32;
    fn energy_incremental(
        &mut self,
        current: usize,
        old: f32,
        new: f32,
    ) -> f32;
    fn init_ji(&mut self);
    fn change<R>(&mut self, idx: usize, rng: &mut R) -> (usize, f32, f32) where R: Rng;
    fn undo(&mut self, undo_info: (usize, f32));
    fn notify_sweep(&mut self);
}

impl LocalModel for HegselmannKrause {
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

    fn local_energy(&self, idx: usize) -> f32 {
        self.ji[idx]
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

        self.ji[current]
    }

    fn change<R>(&mut self, idx: usize, rng: &mut R) -> (usize, f32, f32) where R: Rng {
        let old_x = self.agents[idx].opinion;
        if self.agents[idx].resources < 0. && self.eta > 0. {
            return (idx, old_x, old_x)
        }

        // let mut new_x = old_x + 0.1 * (rng.gen::<f32>() - 0.5);
        // if new_x < 0. {
        //     new_x *= -1.;
        // } else if new_x > 1. {
        //     new_x = 2. - new_x
        // }

        let new_x = rng.gen::<f32>();

        self.update_entry(old_x, new_x);

        self.agents[idx].opinion = new_x;
        self.agents[idx].resources -= (self.agents[idx].opinion - self.agents[idx].initial_opinion).abs();

        (idx, old_x, new_x)
    }

    fn undo(&mut self, undo_info: (usize, f32)) {
        let (idx, old_x) = undo_info;
        let new_x = self.agents[idx].opinion;

        if old_x == new_x {
            return
        }

        self.update_entry(new_x, old_x);

        self.agents[idx].resources += (self.agents[idx].opinion - self.agents[idx].initial_opinion).abs();
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


pub fn local_anneal<T, S, R>(model: &mut T, schedule: S, mut rng: &mut R)
        where T: LocalModel, S: Iterator<Item = f32>, R: Rng {
    model.init_ji();

    for t in schedule {
        // let mut tries = 0;
        // let mut reject = 0;
        for _ in 0..model.size() {
            let idx = (rng.gen::<f32>() * model.size() as f32) as usize;
            let e_before = model.local_energy(idx);
            let (_, old, new) = model.change(idx, &mut rng);
            let undo_info = (idx, old);
            // let e_after = model.energy();
            let e_after = model.energy_incremental(idx, old, new);
            // tries += 1;
            if (-(e_after - e_before) / t).exp() < rng.gen() {
                model.energy_incremental(idx, new, old);
                model.undo(undo_info);
                // reject += 1;
            }
        }
        // println!("{}: {:.0}%", t, reject as f32 / tries as f32 * 100.);
        model.notify_sweep();
    }
}
