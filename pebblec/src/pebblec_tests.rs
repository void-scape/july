use std::ffi::OsString;

use super::*;
use crate::air::ctx::AirCtx;
use crate::air::{self, Air};
use crate::comp::CompUnit;
use crate::ir;
use crate::ir::ctx::Ctx;
use pebblec_parse::lex::Lexer;
use pebblec_parse::lex::source::{Source, SourceMap};

const TESTS: &str = "../demo/tests.peb";

#[test]
fn language_tests() {
    for _ in 0..100 {
        assert_eq!(0, CompUnit::default().path(TESTS).compile(false));
    }
}

#[test]
fn deterministic() {
    let mut source_map = SourceMap::from_paths([TESTS]).unwrap();

    let mut first_items = CompUnit::parse(&OsString::from(TESTS), &mut source_map).unwrap();
    let mut items = first_items.clone();
    assert_eq!(first_items, items);

    let mut first_ctx = Ctx::new(source_map.clone());
    let mut ctx = Ctx::new(source_map);
    assert_eq!(first_ctx, ctx);

    let (first_key, first_const_eval_order) = ir::lower(&mut first_ctx, &mut first_items).unwrap();
    let (key, const_eval_order) = ir::lower(&mut ctx, &mut items).unwrap();
    assert_eq!(first_key, key);
    assert_eq!(first_const_eval_order, const_eval_order);

    let (first_air_funcs, first_consts) =
        air::lower(&first_ctx, &first_key, first_const_eval_order);
    let (air_funcs, consts) = air::lower(&ctx, &key, const_eval_order);
    assert_eq!(first_consts, consts);
    assert_eq!(first_air_funcs, air_funcs);
}
