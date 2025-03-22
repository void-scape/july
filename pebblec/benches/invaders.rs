use criterion::{Criterion, criterion_group, criterion_main};
use pebblec::air::ctx::AirCtx;
use pebblec::{air, ir};
use pebblec_parse::lex::source::Source;
use pebblec_parse::{Parser, lex::*};

fn criterion_benchmark(c: &mut Criterion) {
    let source = Source::new("../demo/invaders.jy").unwrap();
    c.bench_function("lexing invaders", |b| {
        b.iter_with_large_drop(|| Lexer::new(&source).lex().unwrap())
    });

    let buf = Lexer::new(&source).lex().unwrap();
    c.bench_function("parsing invaders", |b| {
        b.iter_with_large_drop(|| Parser::parse(&buf).unwrap())
    });

    let mut items = Parser::parse(&buf).unwrap();
    c.bench_function("lowering invaders", |b| {
        b.iter_batched_ref(
            || items.clone(),
            |items| ir::lower(&buf, items).unwrap(),
            criterion::BatchSize::LargeInput,
        )
    });

    let (ctx, key, const_eval_order) = ir::lower(&buf, &mut items).unwrap();
    let mut air_ctx = AirCtx::new(&ctx, &key);
    c.bench_function("bytecode gen invaders", |b| {
        b.iter_with_large_drop(|| {
            let consts = const_eval_order
                .iter()
                .cloned()
                .flat_map(|id| air::lower_const(&mut air_ctx, ctx.tys.get_const(id).unwrap()))
                .collect::<Vec<_>>();
            let air_funcs = ctx
                .funcs
                .iter()
                .map(|func| air::lower_func(&mut air_ctx, func))
                .collect::<Vec<_>>();
            air_ctx = AirCtx::new(&ctx, &key);
            (air_funcs, consts)
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
