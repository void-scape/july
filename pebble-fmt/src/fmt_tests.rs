use std::ffi::OsString;

use super::*;
use pebblec::air::ctx::AirCtx;
use pebblec::air::{self, Air};
use pebblec::comp::CompUnit;
use pebblec::ir;
use pebblec::ir::ctx::Ctx;
use pebblec_parse::lex::Lexer;
use pebblec_parse::lex::source::{Source, SourceMap};

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
    assert_eq!(0, CompUnit::default().path(TESTS).compile(false));
    assert_eq!(
        0,
        CompUnit::default()
            .source(Source::from_string(
                "tests",
                fmt::fmt(TESTS).unwrap().unwrap()
            ))
            .compile(false)
    );
}

fn codegen_with(path: &str) {
    let origin = OsString::from(&TESTS);
    let mut first_source_map = SourceMap::from_paths([TESTS]).unwrap();
    let mut first_items = CompUnit::parse(&origin, &mut first_source_map).unwrap();
    let mut first_ctx = Ctx::new(first_source_map.clone());
    let (first_key, first_const_eval_order) = ir::lower(&mut first_ctx, &mut first_items).unwrap();
    let (first_air_funcs, first_consts) =
        air::lower(&first_ctx, &first_key, first_const_eval_order);

    let mut source_map =
        SourceMap::from_strings([("fmt-tests", fmt::fmt(TESTS).unwrap().unwrap())]).unwrap();
    let mut items = CompUnit::parse(&origin, &mut source_map).unwrap();
    let mut ctx = Ctx::new(source_map);
    let (key, const_eval_order) = ir::lower(&mut ctx, &mut items).unwrap();
    let (air_funcs, consts) = air::lower(&ctx, &key, const_eval_order);

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
