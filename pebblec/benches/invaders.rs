use criterion::{Criterion, criterion_group, criterion_main};
use pebblec::ir::ctx::Ctx;
use pebblec::{air, ir};
use pebblec_parse::lex::source::SourceMap;
use std::time::Instant;

const INVADERS: &str = "../demo/invaders.peb";

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("source map parse", |b| {
        b.iter_with_large_drop(|| {
            let mut source_map = SourceMap::from_path(INVADERS).unwrap();
            source_map.parse().unwrap();
        });
    })
    .bench_function("ir parse", |b| {
        b.iter_batched(
            || {
                let mut source_map = SourceMap::from_path(INVADERS).unwrap();
                let items = source_map.parse().unwrap();
                (source_map, items)
            },
            |(source_map, items)| {
                let start = Instant::now();
                let ctx = Ctx::new(source_map);
                ir::lower_items(ctx, items).unwrap();
                start.elapsed()
            },
            criterion::BatchSize::LargeInput,
        );
    })
    .bench_function("gen bytecode", |b| {
        b.iter_batched(
            || {
                let source_map = SourceMap::from_path(INVADERS).unwrap();
                ir::lower(source_map).unwrap()
            },
            |ir| air::lower(ir),
            criterion::BatchSize::LargeInput,
        );
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
