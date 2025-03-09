use super::{Next, ParserRule, RResult};
use crate::lex::buffer::*;
use crate::parse::{combinator::prelude::*, matc::*, rules::prelude::*, stream::TokenStream};

/// High level operation performed in blocks.
#[derive(Debug, Clone)]
pub enum Stmt {
    Let {
        name: TokenId,
        ty: Option<PType>,
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
        let (expr, semi) = <(
            Alt<(CntrlFlowRule, ExprRule, AssignRule, LoopRule)>,
            Opt<Next<Semi>>,
        )>::parse(buffer, stream, stack)?;

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
                Opt<(Next<Colon>, TypeRule)>,
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

/// `if <condition> { <stmts> } [else { <stmts> }]`
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
            IfExprRule,
            BlockRules,
            Opt<(Next<Else>, BlockRules)>,
        ) as ParserRule>::parse(buffer, stream, stack)?;

        Ok(if let Some((_else, otherwise)) = otherwise {
            Expr::If(iff, Box::new(expr), block, Some(otherwise))
        } else {
            Expr::If(iff, Box::new(expr), block, None)
        })
    }
}

pub struct IfExprRule;

impl<'a> ParserRule<'a> for IfExprRule {
    type Output = Expr;

    fn parse(
        buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        stack: &mut Vec<TokenId>,
    ) -> RResult<'a, Self::Output> {
        let chk = *stream;
        let mut slice = stream.slice(stream.find_offset::<OpenCurly>());
        stream.eat_until::<OpenCurly>();
        match ExprRule::parse(buffer, &mut slice, stack) {
            Ok(expr) => Ok(expr),
            Err(diag) => {
                *stream = chk;
                Err(diag)
            }
        }
    }
}

/// `loop { <stmts> .. }`
pub struct LoopRule;

impl<'a> ParserRule<'a> for LoopRule {
    type Output = Expr;

    fn parse(
        buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        stack: &mut Vec<TokenId>,
    ) -> RResult<'a, Self::Output> {
        let (loop_, block) =
            <(Next<Loop>, BlockRules) as ParserRule>::parse(buffer, stream, stack)?;
        Ok(Expr::Loop(loop_, block))
    }
}
