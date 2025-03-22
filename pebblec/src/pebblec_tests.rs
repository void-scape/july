use super::*;
use crate::air::ctx::AirCtx;
use crate::air::{self, Air};
use crate::comp::CompUnit;
use crate::ir;
use pebblec_parse::Parser;
use pebblec_parse::lex::Lexer;
use pebblec_parse::lex::source::Source;

const TESTS: &str = "../demo/tests.peb";

#[test]
fn language_tests() {
    assert_eq!(0, CompUnit::new(TESTS).unwrap().compile(false).unwrap());
}

#[test]
fn deterministic() {
    let source = Source::new(TESTS).unwrap();
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

    assert_eq!(first_const_eval_order, const_eval_order);
    assert_eq!(first_key, key);
    assert_eq!(first_ctx, ctx);

    assert_eq!(first_consts, consts);
    assert_eq!(first_air_ctx, air_ctx);
    assert_eq!(first_air_funcs, air_funcs);
}
