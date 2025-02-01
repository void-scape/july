use super::{Next, ParserRule, RResult};
use crate::lex::buffer::*;
use crate::parse::{combinator::prelude::*, matc::*, rules::prelude::*, stream::TokenStream};

/// High level operation performed in blocks.
#[derive(Debug)]
pub enum Stmt {
    Ret(Expr),
    Let {
        name: TokenId,
        ty: Option<TokenId>,
        assign: Expr,
    },
}

pub struct TyBinding {
    pub ident: TokenId,
    pub ty: TokenId,
}

/// `<ident>: <type>`
pub struct TyBindingRule;

impl<'a> ParserRule<'a> for TyBindingRule {
    type Output = TyBinding;

    fn parse(
        buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        stack: &mut Vec<TokenId>,
    ) -> RResult<'a, Self::Output> {
        let (ident, _, ty) =
            <(Next<Ident>, Next<Colon>, Next<Ident>) as ParserRule>::parse(buffer, stream, stack)?;

        Ok(TyBinding { ident, ty })
    }
}

/// `return <expr>;`
pub struct RetRule;

impl<'a> ParserRule<'a> for RetRule {
    type Output = Stmt;

    fn parse(
        buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        stack: &mut Vec<TokenId>,
    ) -> RResult<'a, Self::Output> {
        let (_, expr, _) =
            <(Next<Ret>, ExprRule, Next<Semi>) as ParserRule>::parse(buffer, stream, stack)?;

        Ok(Stmt::Ret(expr))
    }
}

/// `let <ident>[: <type>] = <expr>;`
pub struct LetRule;

impl<'a> ParserRule<'a> for LetRule {
    type Output = Stmt;

    fn parse(
        buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        stack: &mut Vec<TokenId>,
    ) -> RResult<'a, Self::Output> {
        let (_let, name, ty, _equals, expr, _semi) =
            <(
                Next<Let>,
                Next<Ident>,
                Opt<(Next<Colon>, Next<Ident>)>,
                Next<Equals>,
                ExprRule,
                Next<Semi>,
            ) as ParserRule>::parse(buffer, stream, stack)?;

        Ok(Stmt::Let {
            ty: ty.map(|(_, t)| t),
            assign: expr,
            name,
        })
    }
}
