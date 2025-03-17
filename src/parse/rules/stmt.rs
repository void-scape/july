use super::{Next, PErr, ParserRule, RResult};
use crate::ir::AssignKind;
use crate::lex::buffer::*;
use crate::lex::kind::TokenKind;
use crate::parse::{combinator::prelude::*, matc::*, rules::prelude::*, stream::TokenStream};

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

pub struct StmtRule;

impl<'a, 's> ParserRule<'a, 's> for StmtRule {
    type Output = Stmt;

    fn parse(stream: &mut TokenStream<'a, 's>) -> RResult<'s, Self::Output> {
        let chk = *stream;
        match LetRule::parse(stream) {
            Ok(let_) => Ok(let_),
            Err(err) => {
                if !err.recoverable() {
                    return Err(err);
                }

                *stream = chk;
                let (expr, semi) = <(
                    Alt<(CntrlFlowRule, ForRule, LoopRule, ExprRule, AssignRule)>,
                    Opt<Next<Semi>>,
                )>::parse(stream)?;

                if semi.is_some() {
                    Ok(Stmt::Semi(expr))
                } else {
                    Ok(Stmt::Open(expr))
                }
            }
        }
    }
}

pub struct LetRule;

impl<'a, 's> ParserRule<'a, 's> for LetRule {
    type Output = Stmt;

    fn parse(stream: &mut TokenStream<'a, 's>) -> RResult<'s, Self::Output> {
        if !stream.match_peek::<Let>() {
            return Err(PErr::Recover(stream.error("expected `let`")));
        }

        let (_let, name, ty, _equals, expr, _semi) = <(
            Next<Let>,
            Next<Ident>,
            Opt<(Next<Colon>, TypeRule)>,
            Next<Equals>,
            ExprRule,
            Next<Semi>,
        ) as ParserRule>::parse(stream)
        .map_err(PErr::fail)?;

        Ok(Stmt::Let {
            ty: ty.map(|(_, t)| t),
            assign: expr,
            name,
        })
    }
}

#[derive(Debug, Default)]
pub struct AssignRule;

impl<'a, 's> ParserRule<'a, 's> for AssignRule {
    type Output = Expr;

    fn parse(stream: &mut TokenStream<'a, 's>) -> RResult<'s, Self::Output> {
        let mut slice = stream.slice(stream.find_offset::<Any<(Equals, Plus, Hyphen)>>());
        let expr = ExprRule::parse(&mut slice)?;
        stream.eat_until::<Any<(Equals, Plus, Hyphen)>>();

        match stream.peek_kind() {
            Some(TokenKind::Equals) => {
                _ = stream.expect();
                if stream.match_peek::<Equals>() {
                    Err(stream.recover("expected assignment, got equality"))
                } else {
                    Ok(Expr::Assign(Assign {
                        kind: AssignKind::Equals,
                        lhs: Box::new(expr),
                        rhs: Box::new(ExprRule::parse(stream).map_err(PErr::fail)?),
                    }))
                }
            }
            Some(TokenKind::Plus) => {
                let _plus = stream.expect();
                let next = stream.next();
                match next.map(|next| stream.kind(next)) {
                    Some(TokenKind::Equals) => Ok(Expr::Assign(Assign {
                        kind: AssignKind::Add,
                        lhs: Box::new(expr),
                        rhs: Box::new(ExprRule::parse(stream).map_err(PErr::fail)?),
                    })),
                    _ => Err(stream.recover("expected `+`")),
                }
            }
            Some(TokenKind::Hyphen) => {
                let _plus = stream.expect();
                let next = stream.next();
                match next.map(|next| stream.kind(next)) {
                    Some(TokenKind::Equals) => Ok(Expr::Assign(Assign {
                        kind: AssignKind::Sub,
                        lhs: Box::new(expr),
                        rhs: Box::new(ExprRule::parse(stream).map_err(PErr::fail)?),
                    })),
                    _ => Err(stream.recover("expected `-`")),
                }
            }
            _ => Err(stream.recover("expected assignment")),
        }
    }
}

pub struct CntrlFlowRule;

impl<'a, 's> ParserRule<'a, 's> for CntrlFlowRule {
    type Output = Expr;

    fn parse(stream: &mut TokenStream<'a, 's>) -> RResult<'s, Self::Output> {
        if !stream.match_peek::<If>() {
            return Err(PErr::Recover(stream.error("expected `if`")));
        }

        let (iff, expr, block, otherwise) = <(
            Next<If>,
            ToFirstOpenCurlyExprButSubjectToChangeInOtherWordsPleaseFixMe,
            BlockRules,
            Opt<(Next<Else>, BlockRules)>,
        ) as ParserRule>::parse(stream)
        .map_err(PErr::fail)?;

        Ok(if let Some((_else, otherwise)) = otherwise {
            Expr::If(iff, Box::new(expr), block, Some(otherwise))
        } else {
            Expr::If(iff, Box::new(expr), block, None)
        })
    }
}

pub struct ToFirstOpenCurlyExprButSubjectToChangeInOtherWordsPleaseFixMe;

impl<'a, 's> ParserRule<'a, 's> for ToFirstOpenCurlyExprButSubjectToChangeInOtherWordsPleaseFixMe {
    type Output = Expr;

    fn parse(stream: &mut TokenStream<'a, 's>) -> RResult<'s, Self::Output> {
        let offset = stream.find_offset::<OpenCurly>();
        let mut slice = stream.slice(offset);
        stream.eat_n(offset);
        assert!(stream.match_peek::<OpenCurly>());
        match ExprRule::parse(&mut slice) {
            Ok(expr) => {
                if slice.remaining() > 0 {
                    Err(slice.fail("left over tokens"))
                } else {
                    Ok(expr)
                }
            }
            Err(diag) => Err(diag.fail()),
        }
    }
}

pub struct LoopRule;

impl<'a, 's> ParserRule<'a, 's> for LoopRule {
    type Output = Expr;

    fn parse(stream: &mut TokenStream<'a, 's>) -> RResult<'s, Self::Output> {
        if !stream.match_peek::<Loop>() {
            return Err(PErr::Recover(stream.error("expected `loop`")));
        }

        let (loop_, block) =
            <(Next<Loop>, BlockRules) as ParserRule>::parse(stream).map_err(PErr::fail)?;
        Ok(Expr::Loop(loop_, block))
    }
}

pub struct ForRule;

impl<'a, 's> ParserRule<'a, 's> for ForRule {
    type Output = Expr;

    fn parse(stream: &mut TokenStream<'a, 's>) -> RResult<'s, Self::Output> {
        if !stream.match_peek::<For>() {
            return Err(PErr::Recover(stream.error("expected `for`")));
        }

        let spanned = Spanned::<(
            Next<For>,
            Next<Ident>,
            Next<In>,
            ToFirstOpenCurlyExprButSubjectToChangeInOtherWordsPleaseFixMe,
            BlockRules,
        )>::parse(stream)
        .map_err(PErr::fail)?;
        let span = spanned.span();
        let (_four, iter, _inn, iterable, block) = spanned.into_inner();
        Ok(Expr::For {
            span,
            iter,
            iterable: Box::new(iterable),
            block,
        })
    }
}
