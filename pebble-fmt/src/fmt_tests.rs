use super::*;
use pebblec::air::ctx::AirCtx;
use pebblec::air::{self, Air};
use pebblec::comp::CompUnit;
use pebblec::ir;
use pebblec_parse::Parser;
use pebblec_parse::lex::Lexer;
use pebblec_parse::lex::source::Source;

const INVADERS: &str = "res/invaders_unformatted.peb";
const TESTS: &str = "res/tests_unformatted.peb";

#[test]
fn deterministic() {
    let outputs = (0..100)
        .map(|_| fmt::fmt(INVADERS).unwrap().unwrap())
        .collect::<Vec<_>>();
    let first = outputs.first().unwrap();
    assert!(outputs.iter().all(|o| o == first));
}

#[test]
fn fmt_the_fmt() {
    let first = fmt::fmt(INVADERS).unwrap().unwrap();
    let second = fmt::fmt_string(first.clone()).unwrap();
    let third = fmt::fmt_string(second.clone()).unwrap();
    assert_eq!(first, second);
    assert_eq!(second, third);
}

#[test]
fn codegen() {
    codegen_with(INVADERS);
    codegen_with(TESTS);
}

#[test]
fn language_tests() {
    assert_eq!(0, CompUnit::new(TESTS).unwrap().compile(false).unwrap());
    assert_eq!(
        0,
        CompUnit::with_source(Source::from_string(
            "tests",
            fmt::fmt(TESTS).unwrap().unwrap()
        ))
        .compile(false)
        .unwrap()
    );
}

fn codegen_with(path: &str) {
    let source = Source::new(path).unwrap();
    let buf = Lexer::new(&source).lex().unwrap();
    let mut items = Parser::parse(&buf).unwrap();
    let (first_ctx, first_key, first_const_eval_order) = ir::lower(&buf, &mut items).unwrap();

    let mut first_air_ctx = AirCtx::new(&first_ctx, &first_key);
    let (first_air_funcs, first_consts) = {
        let consts = first_const_eval_order
            .clone()
            .into_iter()
            .flat_map(|id| {
                air::lower_const(&mut first_air_ctx, first_ctx.tys.get_const(id).unwrap())
            })
            .collect::<Vec<_>>();
        let air_funcs = first_ctx
            .funcs
            .iter()
            .map(|func| air::lower_func(&mut first_air_ctx, func))
            .collect::<Vec<_>>();
        (air_funcs, consts)
    };

    let source = Source::from_string(path, fmt::fmt(path).unwrap().unwrap());
    let buf = Lexer::new(&source).lex().unwrap();
    let mut items = Parser::parse(&buf).unwrap();
    let (ctx, key, const_eval_order) = ir::lower(&buf, &mut items).unwrap();

    let mut air_ctx = AirCtx::new(&ctx, &key);
    let (air_funcs, consts) = {
        let consts = const_eval_order
            .clone()
            .into_iter()
            .flat_map(|id| air::lower_const(&mut air_ctx, ctx.tys.get_const(id).unwrap()))
            .collect::<Vec<_>>();
        let air_funcs = ctx
            .funcs
            .iter()
            .map(|func| air::lower_func(&mut air_ctx, func))
            .collect::<Vec<_>>();
        (air_funcs, consts)
    };

    assert_eq!(first_air_funcs.len(), air_funcs.len());
    for (first_func, func) in first_air_funcs.iter().zip(air_funcs.iter()) {
        let first_instrs = first_func.instrs();
        let instrs = func.instrs();

        assert_eq!(first_instrs.len(), instrs.len());
        for (first_instr, instr) in first_instrs.iter().zip(instrs.iter()) {
            match (first_instr, instr) {
                (Air::Call(first_sig, first_args), Air::Call(sig, args)) => {
                    assert_eq!(
                        first_ctx.expect_ident(first_sig.ident),
                        ctx.expect_ident(sig.ident),
                    );
                    assert_eq!(first_args, args);
                }
                _ => assert_eq!(first_instr, instr),
            }
        }
    }

    assert_eq!(first_consts, consts);
}
