use super::{Next, ParserRule, RResult};
use crate::lex::buffer::*;
use crate::parse::{combinator::prelude::*, matc::*, rules::prelude::*, stream::TokenStream};

/// High level operation performed in blocks.
#[derive(Debug)]
pub enum Stmt {
    Let {
        name: TokenId,
        ty: Option<TokenId>,
        assign: Expr,
    },
    Semi(Expr),
    Open(Expr),
}

/// `<expr>[;]`
pub struct StmtRule;

impl<'a> ParserRule<'a> for StmtRule {
    type Output = Stmt;

    fn parse(
        buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        stack: &mut Vec<TokenId>,
    ) -> RResult<'a, Self::Output> {
        let (expr, semi) =
            <(ExprRule, Opt<Next<Semi>>) as ParserRule>::parse(buffer, stream, stack)?;

        if semi.is_some() {
            Ok(Stmt::Semi(expr))
        } else {
            Ok(Stmt::Open(expr))
        }
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
