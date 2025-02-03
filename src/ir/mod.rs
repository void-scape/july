use self::ctx::Ctx;
use self::func::Func;
use crate::diagnostic;
use crate::lex::buffer::TokenBuffer;
use crate::parse::Item;

pub mod block;
pub mod ctx;
pub mod expr;
pub mod func;
pub mod ident;
pub mod lit;
pub mod stmt;
pub mod ty;

pub const SYM_DEF: &str = "undefined symbol";

pub fn lower<'a>(tokens: &'a TokenBuffer<'a>, items: &[Item]) -> Ctx<'a> {
    let mut ctx = Ctx::new(tokens);

    match items
        .iter()
        .filter_map(|i| match i {
            Item::Struct(strukt) => Some(strukt),
            _ => None,
        })
        .map(|_| Ok(()))
        .collect::<Result<(), _>>()
    {
        Ok(_) => {}
        Err(diag) => {
            diagnostic::report(diag);
            panic!()
        }
    }

    match items
        .iter()
        .filter_map(|i| match i {
            Item::Func(func) => Some(func),
            _ => None,
        })
        .map(|func| Func::sig(&mut ctx, func))
        .collect::<Result<(), _>>()
    {
        Ok(_) => {}
        Err(diag) => {
            diagnostic::report(diag);
            panic!()
        }
    }

    match items
        .iter()
        .filter_map(|i| match i {
            Item::Func(func) => Some(func),
            _ => None,
        })
        .map(|func| Func::lower(&mut ctx, func))
        .collect::<Result<(), _>>()
    {
        Ok(_) => {}
        Err(diag) => {
            diagnostic::report(diag);
            panic!()
        }
    }

    ctx
}
