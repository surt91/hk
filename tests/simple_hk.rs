extern crate hk;
use hk::HegselmannKrauseBuilder;

use hk::Model;

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    fn test_cmp_naive_bisect() {
        let mut hk1 = HegselmannKrauseBuilder::new(100, 0., 1.)
            .seed(13)
            .build();
        let mut hk2 = HegselmannKrauseBuilder::new(100, 0., 1.)
            .seed(13)
            .build();

        for _ in 0..100 {
            // println!("{}", i);
            hk1.step_naive();
            hk2.step_bisect();
            // println!("naive:  {:?}", hk1);
            // println!("bisect: {:?}", hk2);
            assert!(hk1 == hk2);
        }
    }

    #[test]
    fn test_cmp_sync() {
        let mut hk1 = HegselmannKrauseBuilder::new(100, 0., 1.)
            .seed(13)
            .build();
        let mut hk2 = HegselmannKrauseBuilder::new(100, 0., 1.)
            .seed(13)
            .build();

        for _ in 0..100 {
            hk1.sweep_synchronous_naive();
            hk2.sweep_synchronous_bisect();
            // println!("naive:  {:?}", hk1);
            // println!("bisect: {:?}", hk2);
            assert!(hk1 == hk2);
        }
    }

    #[test]
    fn test_cmp_energy() {
        use rand::SeedableRng;
        use rand_pcg::Pcg64;
        let mut rng = Pcg64::seed_from_u64(42);

        let mut hk = HegselmannKrauseBuilder::new(100, 0., 1.)
            .seed(13)
            .eta(0.5)
            .build();

        hk.init_ji();
        for i in 0..100 {
            let (idx, old, new) = hk.change(&mut rng);
            let e1 = hk.energy();
            let e2 = hk.energy_incremental(idx, old, new);
            // println!("naive:  {:?}", hk1);
            // println!("bisect: {:?}", hk2);
            println!("{}: {}: {} -> {}", i, idx, old, new);
            println!("{} {}", e1, e2);
            assert!((e1 - e2).abs() < 1e-4);
        }
    }
}
