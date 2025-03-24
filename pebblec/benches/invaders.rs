use std::ffi::OsString;

use criterion::{Criterion, criterion_group, criterion_main};
use pebblec::air::ctx::AirCtx;
use pebblec::comp::CompUnit;
use pebblec::ir::ctx::Ctx;
use pebblec::{air, ir};
use pebblec_parse::lex::source::{Source, SourceMap};
use pebblec_parse::lex::*;

fn criterion_benchmark(c: &mut Criterion) {
    let source = Source::new("../demo/invaders.peb").unwrap();
    c.bench_function("lex invaders", |b| {
        b.iter_batched(
            || source.clone(),
            |source| Lexer::new(source).lex().unwrap(),
            criterion::BatchSize::LargeInput,
        )
    });

    let buf = Lexer::new(source).lex().unwrap();
    let mut map = SourceMap::default();
    map.insert(buf);
    let origin = OsString::from("../demo/invaders.peb");
    c.bench_function("parse invaders", |b| {
        b.iter_batched(
            || map.clone(),
            |mut map| CompUnit::parse(&origin, &mut map).unwrap(),
            criterion::BatchSize::LargeInput,
        )
    });

    let mut items = CompUnit::parse(&origin, &mut map).unwrap();
    c.bench_function("ir invaders", |b| {
        b.iter_batched(
            || (map.clone(), items.clone()),
            |(map, mut items)| {
                let mut ctx = Ctx::new(map);
                _ = ir::lower(&mut ctx, &mut items).unwrap();
            },
            criterion::BatchSize::LargeInput,
        )
    });

    let mut ctx = Ctx::new(map);
    let (key, const_eval_order) = ir::lower(&mut ctx, &mut items).unwrap();
    let mut air_ctx = AirCtx::new(&ctx, &key);
    c.bench_function("air invaders", |b| {
        b.iter_batched(
            || const_eval_order.clone(),
            |const_eval_order| {
                _ = air::lower(&ctx, &key, const_eval_order);
            },
            criterion::BatchSize::LargeInput,
        );
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
