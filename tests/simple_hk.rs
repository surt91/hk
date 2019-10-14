extern crate hk;
use hk::HegselmannKrause;

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    fn test_cmp_naive_bisect() {
        let mut hk1 = HegselmannKrause::new(100, 0., 1.0, 0., 13);
        let mut hk2 = HegselmannKrause::new(100, 0., 1.0, 0., 13);
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
        let mut hk1 = HegselmannKrause::new(100, 0., 1., 0., 13);
        let mut hk2 = HegselmannKrause::new(100, 0., 1., 0., 13);
        for _ in 0..100 {
            hk1.sweep_synchronous_naive();
            hk2.sweep_synchronous_bisect();
            // println!("naive:  {:?}", hk1);
            // println!("bisect: {:?}", hk2);
            assert!(hk1 == hk2);
        }
    }
}
