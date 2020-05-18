#[macro_use]
extern crate criterion;

use criterion::Criterion;

extern crate hk;
use hk::HegselmannKrauseBuilder;
use hk::PopulationModel;

fn criterion_benchmark(c: &mut Criterion) {
    let mut hk = HegselmannKrauseBuilder::new(100)
        .seed(13)
        .population_model(PopulationModel::Uniform(0., 1.))
        .build();

    hk.reset();
    c.bench_function("hk N=1000 sweep", |b| b.iter(|| hk.sweep()));

    hk.reset();
    c.bench_function("hk N=1000 sync sweep", |b| b.iter(|| hk.sweep_synchronous_naive()));

    hk.reset();
    c.bench_function("hk N=1000 sync btree sweep", |b| b.iter(|| hk.sweep_synchronous_bisect()));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
