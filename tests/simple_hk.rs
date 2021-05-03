extern crate hk;
use hk::ABMBuilder;

use rand::SeedableRng;
use rand_pcg::Pcg64;

use hk::PopulationModel;

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    fn test_cmp_naive_bisect() {
        let mut hk1 = ABMBuilder::new(100)
            .population_model(PopulationModel::Uniform(0., 1.))
            .seed(13)
            .hk();
        let mut hk2 = ABMBuilder::new(100)
            .population_model(PopulationModel::Uniform(0., 1.))
            .seed(13)
            .hk();

        let mut rng1 = Pcg64::seed_from_u64(42);
        let mut rng2 = Pcg64::seed_from_u64(42);

        for _ in 0..100 {
            // println!("{}", i);
            hk1.step_naive(&mut rng1);
            hk2.step_bisect(&mut rng2);
            // println!("naive:  {:?}", hk1);
            // println!("bisect: {:?}", hk2);
            assert!(hk1 == hk2);
        }
    }

    #[test]
    fn test_cmp_sync() {
        let mut hk1 = ABMBuilder::new(100)
            .population_model(PopulationModel::Uniform(0., 1.))
            .seed(13)
            .hk();
        let mut hk2 = ABMBuilder::new(100)
            .population_model(PopulationModel::Uniform(0., 1.))
            .seed(13)
            .hk();

        for _ in 0..100 {
            hk1.sweep_synchronous_naive();
            hk2.sweep_synchronous_bisect();
            // println!("naive:  {:?}", hk1);
            // println!("bisect: {:?}", hk2);
            assert!(hk1 == hk2);
        }
    }
}
