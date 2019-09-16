#[macro_use]
extern crate criterion;

use criterion::Criterion;
use criterion::black_box;

extern crate hk;
use hk::HegselmannKrause;

fn criterion_benchmark(c: &mut Criterion) {
    let mut hk = HegselmannKrause::new(10, 0., 1., 13);
    c.bench_function("hk N=10 sweep", |b| b.iter(|| hk.sweep()));
    let mut hk = HegselmannKrause::new(100, 0., 1., 13);
    c.bench_function("hk N=100 sweep", |b| b.iter(|| hk.sweep()));
    let mut hk = HegselmannKrause::new(1000, 0., 1., 13);
    c.bench_function("hk N=1000 sweep", |b| b.iter(|| hk.sweep()));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
