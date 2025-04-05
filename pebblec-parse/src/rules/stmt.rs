use super::{Next, PErr, ParserRule, RResult};
use crate::AssignKind;
use crate::lex::buffer::*;
use crate::lex::kind::*;
use crate::{combinator::prelude::*, matc::*, rules::prelude::*, stream::TokenStream};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stmt {
    Let {
        span: Span,
        let_: TokenId,
        name: TokenId,
        ty: Option<PType>,
        assign: Expr,
    },
    Semi(Expr),
    Open(Expr),
}

pub struct StmtRule;

impl<'a, 's> ParserRule<'a> for StmtRule {
    type Output = Stmt;

    fn parse(stream: &mut TokenStream<'a>) -> RResult<Self::Output> {
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

impl<'a, 's> ParserRule<'a> for LetRule {
    type Output = Stmt;

    fn parse(stream: &mut TokenStream<'a>) -> RResult<Self::Output> {
        if !stream.match_peek::<Let>() {
            return Err(PErr::Recover(stream.error("expected `let`")));
        }

        let (let_, name) = <(Next<Let>, Next<Ident>)>::parse(stream).map_err(PErr::fail)?;
        let ty = if Opt::<Next<Colon>>::parse(stream)?.is_some() {
            Some(TypeRule::parse(stream).map_err(PErr::fail)?)
        } else {
            None
        };

        let (_equals, expr, _semi) =
            <(Next<Equals>, ExprRule, Next<Semi>) as ParserRule>::parse(stream)
                .map_err(PErr::fail)?;

        Ok(Stmt::Let {
            span: Span::from_spans(stream.span(let_), stream.span(_semi)),
            let_,
            ty,
            assign: expr,
            name,
        })
    }
}

#[derive(Debug, Default)]
pub struct AssignRule;

impl<'a, 's> ParserRule<'a> for AssignRule {
    type Output = Expr;

    fn parse(stream: &mut TokenStream<'a>) -> RResult<Self::Output> {
        let offset = stream.find_offset::<Equals>();
        let mut slice = stream.slice(offset);
        stream.eat_n(offset);

        while Any::<(
            Hyphen,
            Plus,
            Asterisk,
            Slash,
            Percent,
            Caret,
            Ampersand,
            Pipe,
            CloseAngle,
            OpenAngle,
        )>::matches(Some(stream.kind(stream.prev())))
        {
            stream.back();
            slice.shrink();
        }

        let expr = ExprRule::parse(&mut slice)?;
        match stream.peek_kind() {
            Some(TokenKind::Equals) => {
                _ = stream.expect();
                if stream.match_peek::<Equals>() {
                    Err(stream.recover("expected assignment, got equality"))
                } else {
                    let lhs = Box::new(expr);
                    let rhs = Box::new(ExprRule::parse(stream).map_err(PErr::fail)?);
                    Ok(Expr::Assign(Assign {
                        span: Span::from_spans(
                            lhs.span(stream.token_buffer()),
                            rhs.span(stream.token_buffer()),
                        ),
                        kind: AssignKind::Equals,
                        lhs,
                        rhs,
                    }))
                }
            }
            Some(TokenKind::Hyphen)
            | Some(TokenKind::Plus)
            | Some(TokenKind::Asterisk)
            | Some(TokenKind::Slash)
            | Some(TokenKind::Percent)
            | Some(TokenKind::Caret)
            | Some(TokenKind::Ampersand)
            | Some(TokenKind::Pipe)
            | Some(TokenKind::CloseAngle)
            | Some(TokenKind::OpenAngle) => {
                let first = stream.expect();
                let next = stream.next();
                Ok(match next.map(|next| stream.kind(next)) {
                    Some(TokenKind::Equals) => {
                        let lhs = Box::new(expr);
                        let rhs = Box::new(ExprRule::parse(stream).map_err(PErr::fail)?);

                        let kind = match stream.kind(first) {
                            TokenKind::Hyphen => AssignKind::Sub,
                            TokenKind::Plus => AssignKind::Add,
                            TokenKind::Asterisk => AssignKind::Mul,
                            TokenKind::Slash => AssignKind::Div,
                            TokenKind::Percent => AssignKind::Rem,
                            TokenKind::Caret => AssignKind::Xor,
                            TokenKind::Ampersand => AssignKind::And,
                            TokenKind::Pipe => AssignKind::Or,
                            TokenKind::OpenAngle => {
                                return Err(PErr::Fail(stream.report_error(
                                    "expected assignment `=` or shift left assignment `<<=`",
                                    Span::from_spans(
                                        stream.span(first),
                                        stream.span(next.unwrap()),
                                    ),
                                )));
                            }
                            TokenKind::CloseAngle => {
                                return Err(PErr::Fail(stream.report_error(
                                    "expected assignment `=` or shift right assignment `>>=`",
                                    Span::from_spans(
                                        stream.span(first),
                                        stream.span(next.unwrap()),
                                    ),
                                )));
                            }
                            _ => unreachable!(),
                        };

                        Expr::Assign(Assign {
                            span: Span::from_spans(
                                lhs.span(stream.token_buffer()),
                                rhs.span(stream.token_buffer()),
                            ),
                            kind,
                            lhs,
                            rhs,
                        })
                    }
                    Some(TokenKind::OpenAngle) => {
                        if stream.match_peek::<Equals>()
                            && stream.kind(first) != TokenKind::OpenAngle
                        {
                            return Err(PErr::Fail(stream.report_error(
                                "expected assignment `=` or shift left assignment `<<=`",
                                Span::from_spans(stream.span(first), stream.span(next.unwrap())),
                            )));
                        }

                        stream.eat();
                        let lhs = Box::new(expr);
                        let rhs = Box::new(ExprRule::parse(stream).map_err(PErr::fail)?);

                        Expr::Assign(Assign {
                            span: Span::from_spans(
                                lhs.span(stream.token_buffer()),
                                rhs.span(stream.token_buffer()),
                            ),
                            kind: AssignKind::Shl,
                            lhs,
                            rhs,
                        })
                    }
                    Some(TokenKind::CloseAngle) => {
                        if stream.match_peek::<Equals>()
                            && stream.kind(first) != TokenKind::CloseAngle
                        {
                            return Err(PErr::Fail(stream.report_error(
                                "expected assignment `=` or shift right assignment `>>=`",
                                Span::from_spans(stream.span(first), stream.span(next.unwrap())),
                            )));
                        }

                        stream.eat();
                        let lhs = Box::new(expr);
                        let rhs = Box::new(ExprRule::parse(stream).map_err(PErr::fail)?);

                        Expr::Assign(Assign {
                            span: Span::from_spans(
                                lhs.span(stream.token_buffer()),
                                rhs.span(stream.token_buffer()),
                            ),
                            kind: AssignKind::Shr,
                            lhs,
                            rhs,
                        })
                    }
                    _ => return Err(stream.recover("expected assignment")),
                })
            }
            _ => Err(stream.recover("expected assignment")),
        }
    }
}

pub struct CntrlFlowRule;

impl<'a, 's> ParserRule<'a> for CntrlFlowRule {
    type Output = Expr;

    fn parse(stream: &mut TokenStream<'a>) -> RResult<Self::Output> {
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

        let span = if let Some(otherwise) = &otherwise {
            Span::from_spans(stream.span(iff), otherwise.1.span)
        } else {
            Span::from_spans(stream.span(iff), block.span)
        };

        Ok(if let Some((_else, otherwise)) = otherwise {
            Expr::If {
                span,
                condition: Box::new(expr),
                block,
                otherwise: Some(otherwise),
            }
        } else {
            Expr::If {
                span,
                condition: Box::new(expr),
                block,
                otherwise: None,
            }
        })
    }
}

pub struct ToFirstOpenCurlyExprButSubjectToChangeInOtherWordsPleaseFixMe;

impl<'a, 's> ParserRule<'a> for ToFirstOpenCurlyExprButSubjectToChangeInOtherWordsPleaseFixMe {
    type Output = Expr;

    fn parse(stream: &mut TokenStream<'a>) -> RResult<Self::Output> {
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

impl<'a, 's> ParserRule<'a> for LoopRule {
    type Output = Expr;

    fn parse(stream: &mut TokenStream<'a>) -> RResult<Self::Output> {
        if !stream.match_peek::<Loop>() {
            return Err(PErr::Recover(stream.error("expected `loop`")));
        }

        let (loop_, block) =
            <(Next<Loop>, BlockRules) as ParserRule>::parse(stream).map_err(PErr::fail)?;
        Ok(Expr::Loop(loop_, block))
    }
}

pub struct ForRule;

impl<'a, 's> ParserRule<'a> for ForRule {
    type Output = Expr;

    fn parse(stream: &mut TokenStream<'a>) -> RResult<Self::Output> {
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
