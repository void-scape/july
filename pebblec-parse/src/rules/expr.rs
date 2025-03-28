use super::arr::{ArrDef, ArrDefRule};
use super::block::Block;
use super::func::ArgsRule;
use super::strukt::StructDef;
use super::types::{PType, TypeRule};
use super::{ParserRule, RResult};
use crate::combinator::opt::Opt;
use crate::combinator::spanned::Spanned;
use crate::lex::{buffer::*, kind::*};
use crate::rules::PErr;
use crate::rules::strukt::StructDefRule;
use crate::{AssignKind, BinOpKind, UOpKind};
use crate::{matc::*, stream::TokenStream};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    Break(TokenId),
    Continue(TokenId),
    Ident(TokenId),
    Lit(TokenId),
    Str(TokenId),
    Bool(TokenId),
    Bin(Span, BinOpKind, Box<Expr>, Box<Expr>),
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
    MethodCall {
        span: Span,
        lhs: Box<Expr>,
        method: TokenId,
        args: Vec<Expr>,
    },
    If {
        span: Span,
        condition: Box<Expr>,
        block: Block,
        otherwise: Option<Block>,
    },
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
    Cast {
        span: Span,
        lhs: Box<Expr>,
        ty: PType,
    },
    Loop(TokenId, Block),
    Unary(Span, TokenId, UOpKind, Box<Expr>),
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
            Self::Bin(span, _, _, _) => *span,
            Self::Ret(span, _) => *span,
            Self::Assign(assign) => {
                Span::from_spans(assign.lhs.span(token_buffer), assign.rhs.span(token_buffer))
            }
            Self::StructDef(def) => def.span,
            //Self::EnumDef(def) => def.span,
            Self::Array(def) => match def {
                ArrDef::Repeated { span, .. } => *span,
                ArrDef::Elems { span, .. } => *span,
            },
            Self::Access { span, .. } => *span,
            Self::IndexOf { span, .. } => *span,
            Self::Call { span, .. } => *span,
            Self::If { span, .. } => *span,
            Self::For { span, .. } => *span,
            Self::Range { span, .. } => *span,
            Self::Loop(t, block) => Span::from_spans(token_buffer.span(*t), block.span),
            Self::Cast { span, .. } => *span,
            Self::Unary(span, _, _, _) => *span,
            Self::MethodCall { span, .. } => *span,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Assign {
    pub span: Span,
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
            Self::Mul | Self::Div => 9,
            Self::Add | Self::Sub => 8,
            Self::Shl | Self::Shr => 7,
            Self::Band => 6,
            Self::Xor => 5,
            Self::Bor => 4,
            Self::Eq | Self::Ne => 3,
            Self::Gt | Self::Lt | Self::Ge | Self::Le => 2,
            Self::And => 1,
            Self::Or => 0,
        }
    }
}

// TODO: ranges should be operators will low precedence
impl<'a, 's> ParserRule<'a> for BinOpKindRule {
    type Output = BinOpKind;

    fn parse(stream: &mut TokenStream<'a>) -> RResult<Self::Output> {
        match stream.peek_kind() {
            Some(TokenKind::Plus)
            | Some(TokenKind::Hyphen)
            | Some(TokenKind::Asterisk)
            | Some(TokenKind::Slash) => {
                let mut tmp = *stream;
                let plus = tmp.expect();
                if tmp.peek_kind() == Some(TokenKind::Equals) {
                    let equals = tmp.expect();
                    return Err(PErr::Fail(stream.report_error(
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
                        stream.report_error("cannot assign expression", stream.span(equals)),
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
                        stream.report_error("cannot assign expression", stream.span(bang)),
                    ));
                }
            }
            Some(TokenKind::Plus) => BinOpKind::Add,
            Some(TokenKind::Hyphen) => BinOpKind::Sub,
            Some(TokenKind::Asterisk) => BinOpKind::Mul,
            Some(TokenKind::Caret) => BinOpKind::Xor,
            Some(TokenKind::Slash) => BinOpKind::Div,
            Some(TokenKind::Pipe) => BinOpKind::Bor,
            Some(TokenKind::Ampersand) => BinOpKind::Band,
            Some(TokenKind::OpenAngle) => {
                stream.expect();
                if stream.match_peek::<Equals>() {
                    stream.expect();
                    return Ok(BinOpKind::Le);
                } else if stream.match_peek::<OpenAngle>() {
                    stream.expect();
                    return Ok(BinOpKind::Shl);
                } else {
                    return Ok(BinOpKind::Lt);
                }
            }
            Some(TokenKind::CloseAngle) => {
                stream.expect();
                if stream.match_peek::<Equals>() {
                    stream.expect();
                    return Ok(BinOpKind::Ge);
                } else if stream.match_peek::<CloseAngle>() {
                    stream.expect();
                    return Ok(BinOpKind::Shr);
                } else {
                    return Ok(BinOpKind::Gt);
                }
            }
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

impl<'a, 's> ParserRule<'a> for TermRule {
    type Output = Expr;

    fn parse(stream: &mut TokenStream<'a>) -> RResult<Self::Output> {
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

        let peek = stream.peek_kind();
        let term = match peek {
            Some(TokenKind::Ident) => match stream.peekn(1).map(|t| stream.kind(t)) {
                Some(TokenKind::OpenParen) => {
                    let ident = stream.expect();
                    let (span, args) = ArgsRule::parse(stream).map_err(PErr::fail)?;
                    Ok(Expr::Call {
                        func: ident,
                        span: Span::from_spans(stream.span(ident), span),
                        args,
                    })
                }
                Some(TokenKind::OpenCurly) => Ok(Expr::StructDef(
                    StructDefRule::parse(stream).map_err(PErr::fail)?,
                )),
                _ => Ok(Expr::Ident(stream.expect())),
            },
            Some(TokenKind::Slf) => Ok(Expr::Ident(stream.expect())),
            Some(TokenKind::Float) | Some(TokenKind::Int) => Ok(Expr::Lit(stream.expect())),
            Some(TokenKind::True) | Some(TokenKind::False) => Ok(Expr::Bool(stream.expect())),
            Some(TokenKind::Str) => Ok(Expr::Str(stream.expect())),
            Some(TokenKind::Hyphen) | Some(TokenKind::Bang) | Some(TokenKind::Ampersand) => Ok({
                let t = stream.expect();
                let expr = Box::new(TermRule::parse(stream)?);

                let kind = match peek.unwrap() {
                    TokenKind::Hyphen => UOpKind::Neg,
                    TokenKind::Bang => UOpKind::Not,
                    TokenKind::Ampersand => UOpKind::Ref,
                    _ => unreachable!(),
                };

                Expr::Unary(
                    Span::from_spans(stream.span(t), expr.span(stream.token_buffer())),
                    t,
                    kind,
                    expr,
                )
            }),
            Some(TokenKind::OpenBracket) => Ok(Expr::Array(ArrDefRule::parse(stream)?)),
            Some(TokenKind::Break) => Ok(Expr::Break(stream.expect())),
            Some(TokenKind::Continue) => Ok(Expr::Continue(stream.expect())),
            Some(TokenKind::OpenParen) => {
                let open = stream.expect();
                let offset = stream.find_matched_delim_offset::<Paren>();
                let mut slice = stream.slice(offset);
                stream.eat_n(offset);
                let inner =
                    ExprRule::parse(&mut slice).map_err(|_| {
                        PErr::Fail(stream.report_error(
                            "expected expression within delimiters",
                            stream.span(open),
                        ))
                    })?;

                if stream.is_empty() {
                    return Err(PErr::Fail(
                        stream.report_error("mismatched delimiter", stream.span(open)),
                    ));
                }

                assert!(stream.match_peek::<CloseParen>());
                stream.expect();
                Ok(Expr::Paren(Box::new(inner)))
            }
            _ => Err(PErr::Recover(stream.report_error(
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
                    return Err(PErr::Fail(stream.report_error(
                        "invalid access: expected identifier after `.`",
                        stream.span(dot),
                    )));
                }

                let field = stream.expect();

                if stream.match_peek::<OpenParen>() {
                    let (span, args) = ArgsRule::parse(stream).map_err(PErr::fail)?;
                    term_result = Expr::MethodCall {
                        span,
                        method: field,
                        lhs: Box::new(term_result),
                        args,
                    };
                } else {
                    term_result = Expr::Access {
                        span: Span::from_spans(start_span, stream.span(field)),
                        lhs: Box::new(term_result),
                        field,
                    };
                }
            } else if stream.match_peek::<OpenBracket>() {
                let open_bracket = stream.expect();
                let index_expr = ExprRule::parse(stream)?;

                if !stream.match_peek::<CloseBracket>() {
                    return Err(PErr::Fail(stream.report_error(
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

                        let t = stream.expect();
                        let expr = Box::new(term_result);

                        term_result = Expr::Unary(
                            Span::from_spans(stream.span(t), expr.span(stream.token_buffer())),
                            t,
                            UOpKind::Deref,
                            expr,
                        )
                    }
                }
            } else if stream.match_peek::<As>() {
                let chk = *stream;
                let _as = stream.expect();
                match TypeRule::parse(stream) {
                    Ok(ty) => {
                        term_result = Expr::Cast {
                            span: Span::from_spans(
                                term_result.span(stream.token_buffer()),
                                ty.span(),
                            ),
                            lhs: Box::new(term_result),
                            ty,
                        }
                    }
                    Err(_) => {
                        *stream = chk;
                        return Err(stream.fail("expected a type"));
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

impl<'a, 's> ParserRule<'a> for ExprRule {
    type Output = Expr;

    fn parse(stream: &mut TokenStream<'a>) -> RResult<Self::Output> {
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
            let lhs = match bin {
                None => {
                    assert!(lhs.is_some());
                    lhs.take().unwrap()
                }
                Some(lhs) => lhs,
            };

            bin = Some(Expr::Bin(
                Span::from_spans(
                    lhs.span(stream.token_buffer()),
                    rhs.span(stream.token_buffer()),
                ),
                op,
                Box::new(lhs),
                Box::new(rhs),
            ));
        }

        match bin {
            Some(bin) => Ok(reorder_expr(stream, bin)),
            None => Ok(reorder_expr(stream, lhs.take().unwrap())),
        }
    }
}

fn reorder_expr(stream: &TokenStream, expr: Expr) -> Expr {
    match expr {
        Expr::Paren(inner) => Expr::Paren(Box::new(reorder_expr(stream, *inner))),
        Expr::Bin(_, op, lhs, rhs) => {
            let lhs = reorder_expr(stream, *lhs);
            let rhs = Box::new(reorder_expr(stream, *rhs));

            match lhs {
                Expr::Bin(_, lhs_op, lhs_lhs, lhs_rhs)
                    if lhs_op.precedence() < op.precedence() && !matches!(lhs, Expr::Paren(_)) =>
                {
                    let rhs_span = Span::from_spans(
                        lhs_rhs.span(stream.token_buffer()),
                        rhs.span(stream.token_buffer()),
                    );
                    let rhs = Box::new(Expr::Bin(rhs_span, op, lhs_rhs, rhs));

                    let span = Span::from_spans(lhs_lhs.span(stream.token_buffer()), rhs_span);
                    Expr::Bin(span, lhs_op, lhs_lhs, rhs)
                }
                _ => {
                    let span = Span::from_spans(
                        lhs.span(stream.token_buffer()),
                        rhs.span(stream.token_buffer()),
                    );
                    Expr::Bin(span, op, Box::new(lhs), rhs)
                }
            }
        }
        _ => expr,
    }
}
