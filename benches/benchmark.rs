#[macro_use]
extern crate criterion;

use criterion::Criterion;
use criterion::black_box;

extern crate hk;
use hk::HegselmannKrauseBuilder;
use hk::{Model, anneal, Linear};

fn criterion_benchmark(c: &mut Criterion) {
    // let mut hk = HegselmannKrause::new(1000, 0., 1., 13);
    // c.bench_function("hk N=10 sweep", |b| b.iter(|| hk.sweep()));
    // let mut hk = HegselmannKrause::new(100, 0., 1., 13);
    // c.bench_function("hk N=100 sweep", |b| b.iter(|| hk.sweep()));
    let mut hk = HegselmannKrauseBuilder::new(100, 0., 1.).seed(13).build();
    c.bench_function("hk N=1000 sweep", |b| b.iter(|| hk.sweep()));

    let mut hk = HegselmannKrauseBuilder::new(1000, 0., 1.).seed(13).build();
    c.bench_function("hk N=1000 sync sweep", |b| b.iter(|| hk.sweep_synchronous_naive()));

    let mut hk = HegselmannKrauseBuilder::new(1000, 0., 1.).seed(13).build();
    c.bench_function("hk N=1000 sync btree sweep", |b| b.iter(|| hk.sweep_synchronous_bisect()));


    use rand::SeedableRng;
    use rand_pcg::Pcg64;
    let mut hk = HegselmannKrauseBuilder::new(100, 0., 1.).seed(13).eta(0.5).build();
    let mut rng = Pcg64::seed_from_u64(42);
    c.bench_function("hk annealing", |b| b.iter(|| anneal(&mut hk, Linear::new(20, 0.), &mut rng)));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
