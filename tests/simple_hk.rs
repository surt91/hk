extern crate hk;
use hk::HegselmannKrause;

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    fn test_cmp_naive_bisect() {
        let mut hk1 = HegselmannKrause::new(100, 13);
        let mut hk2 = HegselmannKrause::new(100, 13);
        for i in 0..1000*200 {
            println!("{}", i);
            println!("naive:  {:?}", hk1);
            println!("bisect: {:?}", hk2);
            hk1.step_naive();
            hk2.step_bisect();
            assert!(hk1 == hk2);
        }
    }
}
