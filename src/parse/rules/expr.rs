use super::arr::{ArrDef, ArrDefRule};
use super::block::Block;
use super::func::ArgsRule;
use super::strukt::StructDef;
use super::{ParserRule, RResult};
use crate::ir::{AssignKind, BinOpKind, UOpKind};
use crate::lex::{buffer::*, kind::*};
use crate::parse::combinator::opt::Opt;
use crate::parse::combinator::spanned::Spanned;
use crate::parse::rules::strukt::StructDefRule;
use crate::parse::rules::PErr;
use crate::parse::{matc::*, stream::TokenStream};

#[derive(Debug, Clone)]
pub enum Expr {
    Break(TokenId),
    Continue(TokenId),
    Ident(TokenId),
    Lit(TokenId),
    Str(TokenId),
    Bool(TokenId),
    Bin(BinOpKind, Box<Expr>, Box<Expr>),
    Paren(Box<Expr>),
    Ret(Span, Option<Box<Expr>>),
    Assign(Assign),
    StructDef(StructDef),
    //EnumDef(EnumDef),
    Array(ArrDef),
    Access {
        span: Span,
        lhs: Box<Expr>,
        field: TokenId,
    },
    IndexOf {
        span: Span,
        array: Box<Expr>,
        index: Box<Expr>,
    },
    Call {
        span: Span,
        func: TokenId,
        args: Vec<Expr>,
    },
    If(TokenId, Box<Expr>, Block, Option<Block>),
    For {
        span: Span,
        iter: TokenId,
        iterable: Box<Expr>,
        block: Block,
    },
    Range {
        span: Span,
        start: Option<Box<Expr>>,
        end: Option<Box<Expr>>,
        inclusive: bool,
    },
    Loop(TokenId, Block),
    Unary(TokenId, UOpKind, Box<Expr>),
}

impl Expr {
    pub fn span(&self, token_buffer: &TokenBuffer) -> Span {
        match self {
            Self::Break(t) => token_buffer.span(*t),
            Self::Continue(t) => token_buffer.span(*t),
            Self::Ident(t) => token_buffer.span(*t),
            Self::Lit(t) => token_buffer.span(*t),
            Self::Str(t) => token_buffer.span(*t),
            Self::Bool(t) => token_buffer.span(*t),
            Self::Paren(inner) => inner.span(token_buffer),
            Self::Bin(_, lhs, rhs) => {
                Span::from_spans(lhs.span(token_buffer), rhs.span(token_buffer))
            }
            Self::Ret(span, _) => *span,
            Self::Assign(assign) => {
                Span::from_spans(assign.lhs.span(token_buffer), assign.rhs.span(token_buffer))
            }
            Self::StructDef(def) => def.span,
            //Self::EnumDef(def) => def.span,
            Self::Array(def) => def.span,
            Self::Access { span, .. } => *span,
            Self::IndexOf { span, .. } => *span,
            Self::Call { span, .. } => *span,
            Self::If(t, _, block, otherwise) => {
                if let Some(otherwise) = otherwise {
                    Span::from_spans(token_buffer.span(*t), otherwise.span)
                } else {
                    Span::from_spans(token_buffer.span(*t), block.span)
                }
            }
            Self::For { span, .. } => *span,
            Self::Range { span, .. } => *span,
            Self::Loop(t, block) => Span::from_spans(token_buffer.span(*t), block.span),
            Self::Unary(t, _, expr) => {
                Span::from_spans(token_buffer.span(*t), expr.span(token_buffer))
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Assign {
    pub kind: AssignKind,
    pub lhs: Box<Expr>,
    pub rhs: Box<Expr>,
}

pub struct BinOpKindRule;

trait Precedence {
    fn precedence(&self) -> usize;
}

impl Precedence for BinOpKind {
    fn precedence(&self) -> usize {
        match self {
            Self::Eq | Self::Ne => 1,
            Self::Xor => 2,
            Self::Add | Self::Sub => 3,
            Self::Mul | Self::Div => 4,
        }
    }
}

// TODO: ranges should be operators will low precedence
impl<'a, 's> ParserRule<'a, 's> for BinOpKindRule {
    type Output = BinOpKind;

    fn parse(stream: &mut TokenStream<'a, 's>) -> RResult<'s, Self::Output> {
        match stream.peek_kind() {
            Some(TokenKind::Plus)
            | Some(TokenKind::Hyphen)
            | Some(TokenKind::Asterisk)
            | Some(TokenKind::Slash) => {
                let mut tmp = *stream;
                let plus = tmp.expect();
                if tmp.peek_kind() == Some(TokenKind::Equals) {
                    let equals = tmp.expect();
                    return Err(PErr::Fail(stream.full_error(
                        "cannot assign expression",
                        Span::from_spans(stream.span(plus), stream.span(equals)),
                    )));
                }
            }
            _ => {}
        }

        let chk = *stream;
        let op = Ok(match stream.peek_kind() {
            Some(TokenKind::Equals) => {
                let equals = stream.expect();
                if stream.match_peek::<Equals>() {
                    BinOpKind::Eq
                } else {
                    *stream = chk;
                    return Err(PErr::Fail(
                        stream.full_error("cannot assign expression", stream.span(equals)),
                    ));
                }
            }
            Some(TokenKind::Bang) => {
                let bang = stream.expect();
                if stream.match_peek::<Equals>() {
                    BinOpKind::Ne
                } else {
                    *stream = chk;
                    return Err(PErr::Fail(
                        stream.full_error("cannot assign expression", stream.span(bang)),
                    ));
                }
            }
            Some(TokenKind::Plus) => BinOpKind::Add,
            Some(TokenKind::Hyphen) => BinOpKind::Sub,
            Some(TokenKind::Asterisk) => BinOpKind::Mul,
            Some(TokenKind::Caret) => BinOpKind::Xor,
            Some(TokenKind::Slash) => BinOpKind::Div,
            kind => {
                return Err(PErr::Recover(stream.error(format!(
                    "expected binary operator, got `{}`",
                    kind.map(|k| k.as_str()).unwrap_or("???")
                ))));
            }
        });

        stream.expect();
        op
    }
}

pub struct TermRule;

impl<'a, 's> ParserRule<'a, 's> for TermRule {
    type Output = Expr;

    fn parse(stream: &mut TokenStream<'a, 's>) -> RResult<'s, Self::Output> {
        let Some(start_span) = stream.peek().map(|t| stream.span(t)) else {
            return Err(stream.recover("expected expression term"));
        };

        if stream.match_peek::<DoubleDot>() {
            let dots = stream.expect();
            let dot_span = stream.span(dots);

            let inclusive = if stream.match_peek::<Equals>() {
                stream.expect();
                true
            } else {
                false
            };

            if let Some(spanned) = Opt::<Spanned<TermRule>>::parse(stream).unwrap() {
                let end_span = spanned.span();
                let end_expr = spanned.into_inner();

                return Ok(Expr::Range {
                    span: Span::from_spans(dot_span, end_span),
                    start: None,
                    end: Some(Box::new(end_expr)),
                    inclusive,
                });
            } else {
                return Ok(Expr::Range {
                    span: Span::from_spans(dot_span, stream.span(dots)),
                    start: None,
                    end: None,
                    inclusive,
                });
            }
        }

        let term = match stream.peek_kind() {
            Some(TokenKind::Ident) => match stream.peekn(1).map(|t| stream.kind(t)) {
                Some(TokenKind::OpenParen) => {
                    let ident = stream.expect();
                    let (span, args) = ArgsRule::parse(stream).map_err(PErr::fail)?;
                    Ok(Expr::Call {
                        func: ident,
                        span,
                        args,
                    })
                }
                Some(TokenKind::OpenCurly) => Ok(Expr::StructDef(
                    StructDefRule::parse(stream).map_err(PErr::fail)?,
                )),
                _ => Ok(Expr::Ident(stream.expect())),
            },
            Some(TokenKind::Float) | Some(TokenKind::Int) => Ok(Expr::Lit(stream.expect())),
            Some(TokenKind::True) | Some(TokenKind::False) => Ok(Expr::Bool(stream.expect())),
            Some(TokenKind::Str) => Ok(Expr::Str(stream.expect())),
            Some(TokenKind::Bang) => Ok(Expr::Unary(
                stream.expect(),
                UOpKind::Not,
                Box::new(TermRule::parse(stream)?),
            )),
            Some(TokenKind::Ampersand) => Ok(Expr::Unary(
                stream.expect(),
                UOpKind::Ref,
                Box::new(TermRule::parse(stream)?),
            )),
            Some(TokenKind::OpenBracket) => Ok(Expr::Array(ArrDefRule::parse(stream)?)),
            Some(TokenKind::Break) => Ok(Expr::Break(stream.expect())),
            Some(TokenKind::Continue) => Ok(Expr::Continue(stream.expect())),
            Some(TokenKind::OpenParen) => {
                let open = stream.expect();
                let offset = stream.find_matched_delim_offset::<Paren>();
                let mut slice = stream.slice(offset);
                stream.eat_n(offset);
                let inner = ExprRule::parse(&mut slice).map_err(|_| {
                    PErr::Fail(
                        stream
                            .full_error("expected expression within delimiters", stream.span(open)),
                    )
                })?;

                if stream.is_empty() {
                    return Err(PErr::Fail(
                        stream.full_error("mismatched delimiter", stream.span(open)),
                    ));
                }

                assert!(stream.match_peek::<CloseParen>());
                stream.expect();
                Ok(Expr::Paren(Box::new(inner)))
            }
            _ => Err(PErr::Recover(stream.full_error(
                "expected term",
                stream.span(stream.peek().unwrap()),
            ))),
        };

        let mut term_result = term?;
        loop {
            if stream.match_peek::<DoubleDot>() {
                let dots = stream.expect();
                let dot_span = stream.span(dots);

                let inclusive = if stream.match_peek::<Equals>() {
                    stream.expect();
                    true
                } else {
                    false
                };

                if let Some(spanned) = Opt::<Spanned<TermRule>>::parse(stream).unwrap() {
                    let end_span = spanned.span();
                    let end_expr = spanned.into_inner();

                    return Ok(Expr::Range {
                        span: Span::from_spans(dot_span, end_span),
                        start: Some(Box::new(term_result)),
                        end: Some(Box::new(end_expr)),
                        inclusive,
                    });
                } else {
                    return Ok(Expr::Range {
                        span: Span::from_spans(dot_span, stream.span(dots)),
                        start: Some(Box::new(term_result)),
                        end: None,
                        inclusive,
                    });
                }
            } else if stream.match_peek::<Dot>() {
                let dot = stream.expect();

                if !stream.match_peek::<Ident>() {
                    return Err(PErr::Fail(stream.full_error(
                        "invalid access: expected identifier after `.`",
                        stream.span(dot),
                    )));
                }

                let field = stream.expect();
                term_result = Expr::Access {
                    span: Span::from_spans(start_span, stream.span(field)),
                    lhs: Box::new(term_result),
                    field,
                };
            } else if stream.match_peek::<OpenBracket>() {
                let open_bracket = stream.expect();
                let index_expr = ExprRule::parse(stream)?;

                if !stream.match_peek::<CloseBracket>() {
                    return Err(PErr::Fail(stream.full_error(
                        "unclosed array index: expected `]`",
                        stream.span(open_bracket),
                    )));
                }

                let close_bracket = stream.expect();
                term_result = Expr::IndexOf {
                    span: Span::from_spans(start_span, stream.span(close_bracket)),
                    array: Box::new(term_result),
                    index: Box::new(index_expr),
                };
            } else if stream.match_peek::<Asterisk>() {
                let chk = *stream;
                let _caret = stream.expect();
                match TermRule::parse(stream) {
                    Ok(_) => {
                        *stream = chk;
                        break;
                    }
                    Err(_) => {
                        *stream = chk;
                        term_result =
                            Expr::Unary(stream.expect(), UOpKind::Deref, Box::new(term_result))
                    }
                }
            } else {
                break;
            }
        }

        Ok(term_result)
    }
}

#[derive(Debug, Default)]
pub struct ExprRule;

impl<'a, 's> ParserRule<'a, 's> for ExprRule {
    type Output = Expr;

    fn parse(stream: &mut TokenStream<'a, 's>) -> RResult<'s, Self::Output> {
        if stream.match_peek::<Ret>() {
            let t = stream.expect();
            return Ok(Expr::Ret(
                stream.span(t),
                Opt::<ExprRule>::parse(stream)?.map(Box::new),
            ));
        }

        let mut bin = None;
        let mut lhs = Some(TermRule::parse(stream).map_err(PErr::fail)?);
        loop {
            let op = match BinOpKindRule::parse(stream) {
                Ok(op) => op,
                Err(err) => {
                    if !err.recoverable() {
                        // fails in the case it is doing an assign, but we want to recover if that is
                        // the case
                        return Err(err.recover());
                    } else {
                        break;
                    }
                }
            };

            let rhs = TermRule::parse(stream)?;
            match bin {
                None => {
                    assert!(lhs.is_some());
                    bin = Some(Expr::Bin(op, Box::new(lhs.take().unwrap()), Box::new(rhs)));
                }
                Some(lhs) => {
                    bin = Some(Expr::Bin(op, Box::new(lhs), Box::new(rhs)));
                }
            }
        }

        match bin {
            Some(bin) => Ok(reorder_expr(bin)),
            None => Ok(reorder_expr(lhs.take().unwrap())),
        }
    }
}

fn reorder_expr(expr: Expr) -> Expr {
    match expr {
        Expr::Paren(inner) => Expr::Paren(Box::new(reorder_expr(*inner))),
        Expr::Bin(op, lhs, rhs) => {
            let lhs = reorder_expr(*lhs);
            let rhs = Box::new(reorder_expr(*rhs));

            match lhs {
                Expr::Bin(lhs_op, lhs_lhs, lhs_rhs)
                    if lhs_op.precedence() < op.precedence() && !matches!(lhs, Expr::Paren(_)) =>
                {
                    Expr::Bin(lhs_op, lhs_lhs, Box::new(Expr::Bin(op, lhs_rhs, rhs)))
                }
                _ => Expr::Bin(op, Box::new(lhs), rhs),
            }
        }
        _ => expr,
    }
}
