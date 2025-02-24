use super::{Next, ParserRule, RResult};
use crate::lex::buffer::*;
use crate::lex::kind::TokenKind;
use crate::parse::{combinator::prelude::*, matc::*, rules::prelude::*, stream::TokenStream};

/// High level operation performed in blocks.
#[derive(Debug, Clone)]
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
        let (expr, semi) = <(Alt<(CntrlFlowRule, ExprRule, AssignRule)>, Opt<Next<Semi>>)>::parse(
            buffer, stream, stack,
        )?;

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

/// `if { <stmts> }`
pub struct CntrlFlowRule;

impl<'a> ParserRule<'a> for CntrlFlowRule {
    type Output = Expr;

    fn parse(
        buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        stack: &mut Vec<TokenId>,
    ) -> RResult<'a, Self::Output> {
        let (iff, expr, block, otherwise) = <(
            Next<If>,
            ExprRule,
            BlockRules,
            Opt<(Next<Else>, BlockRules)>,
        ) as ParserRule>::parse(buffer, stream, stack)?;

        Ok(if let Some((_else, otherwise)) = otherwise {
            Expr::If(iff, Box::new(expr), block, Some(otherwise))
        } else {
            Expr::If(iff, Box::new(expr), block, None)
        })

        //match (buffer.kind(iff), expr) {
        //    (TokenKind::If, Some(expr)) => Ok(Expr::If(iff, Box::new(expr), block)),
        //    (TokenKind::If, None) => {
        //        Err(stream.full_error("expected expression following `if`", buffer.span(iff), ""))
        //    }
        //    (TokenKind::Else, None) => Ok(Expr::Else(iff, block)),
        //    (TokenKind::Else, Some(_)) => {
        //        Err(stream.full_error("expected block following `else`", buffer.span(iff), ""))
        //    }
        //    _ => unreachable!(),
        //}
    }
}
