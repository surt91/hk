#[macro_use]
extern crate criterion;

use criterion::Criterion;

extern crate hk;
use hk::{ABM, ABMBuilder};
use hk::PopulationModel;

use rand::SeedableRng;
use rand_pcg::Pcg64;

fn criterion_benchmark(c: &mut Criterion) {
    let mut hk = ABMBuilder::new(100)
        .seed(13)
        .population_model(PopulationModel::Uniform(0., 1.))
        .hk();

    let mut rng1 = Pcg64::seed_from_u64(42);
    let mut rng2 = Pcg64::seed_from_u64(42);
    let mut rng3 = Pcg64::seed_from_u64(42);

    hk.reset(&mut rng1);
    c.bench_function("hk N=1000 sweep", |b| b.iter(|| hk.sweep(&mut rng1)));

    hk.reset(&mut rng2);
    c.bench_function("hk N=1000 sync sweep", |b| b.iter(|| hk.sweep_synchronous_naive()));

    hk.reset(&mut rng3);
    c.bench_function("hk N=1000 sync btree sweep", |b| b.iter(|| hk.sweep_synchronous_bisect()));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
