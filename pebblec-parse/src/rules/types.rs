use super::{Next, PErr, ParserRule};
use crate::combinator::alt::Alt;
use crate::combinator::spanned::Spanned;
use crate::lex::buffer::{Span, TokenBuffer, TokenId, TokenQuery};
use crate::lex::kind::TokenKind;
use crate::lex::kind::*;
use crate::stream::TokenStream;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PType {
    Simple(TokenId),
    Ref {
        borrow: TokenId,
        inner: Box<PType>,
    },
    Array {
        span: Span,
        size: usize,
        inner: Box<PType>,
    },
    Slice {
        span: Span,
        inner: Box<PType>,
    },
}

impl PType {
    pub fn span(&self, buffer: &TokenBuffer) -> Span {
        match self {
            Self::Simple(id) => buffer.span(*id),
            Self::Ref { borrow, inner } => {
                Span::from_spans(buffer.span(*borrow), inner.span(buffer))
            }
            Self::Array { span, .. } => *span,
            Self::Slice { span, .. } => *span,
        }
    }

    pub fn peel_refs(&self) -> &PType {
        match self {
            Self::Ref { inner, .. } => inner.peel_refs(),
            Self::Simple(_) | Self::Slice { .. } | Self::Array { .. } => self,
        }
    }
}

#[derive(Debug, Default)]
pub struct TypeRule;

impl<'a, 's> ParserRule<'a, 's> for TypeRule {
    type Output = PType;

    fn parse(stream: &mut TokenStream<'a, 's>) -> super::RResult<'s, Self::Output> {
        Alt::<(SimpleType, RefType, ArrayType)>::parse(stream).map_err(|diag| {
            if diag.recoverable() {
                stream.fail("expected type")
            } else {
                diag
            }
        })
    }
}

const BUILTIN_TYPES: &[&str] = &[
    "u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64", "f32", "f64", "bool", "str",
];

#[derive(Debug, Default)]
pub struct SimpleType;

impl<'a, 's> ParserRule<'a, 's> for SimpleType {
    type Output = PType;

    fn parse(stream: &mut TokenStream<'a, 's>) -> super::RResult<'s, Self::Output> {
        if let Some(str) = stream.peek().map(|t| stream.as_str(t)) {
            if BUILTIN_TYPES.contains(&str.as_ref()) {
                return Ok(PType::Simple(stream.expect()));
            }
        }

        if stream.match_peek::<Ident>() {
            Ok(PType::Simple(stream.expect()))
        } else {
            Err(stream.recover("expected a type"))
        }
    }
}

#[derive(Debug, Default)]
pub struct RefType;

impl<'a, 's> ParserRule<'a, 's> for RefType {
    type Output = PType;

    fn parse(stream: &mut TokenStream<'a, 's>) -> super::RResult<'s, Self::Output> {
        if stream.match_peek::<Ampersand>() {
            Ok(PType::Ref {
                borrow: stream.expect(),
                inner: Box::new(TypeRule::parse(stream).map_err(PErr::fail)?),
            })
        } else {
            Err(stream.recover("expected a type with reference"))
        }
    }
}

#[derive(Debug, Default)]
pub struct ArrayType;

impl<'a, 's> ParserRule<'a, 's> for ArrayType {
    type Output = PType;

    fn parse(stream: &mut TokenStream<'a, 's>) -> super::RResult<'s, Self::Output> {
        if !stream.match_peek::<OpenBracket>() {
            return Err(stream.recover("expected `[`"));
        }

        let chk = *stream;
        let (_open, _inner) = <(Next<OpenBracket>, TypeRule)>::parse(stream).map_err(PErr::fail)?;
        match stream.peek_kind() {
            Some(TokenKind::CloseBracket) => {
                *stream = chk;
                let spanned =
                    Spanned::<(Next<OpenBracket>, TypeRule, Next<CloseBracket>)>::parse(stream)
                        .map_err(PErr::fail)?;
                let span = spanned.span();
                let (_open, inner, _close) = spanned.into_inner();

                Ok(PType::Slice {
                    span,
                    inner: Box::new(inner),
                })
            }
            Some(TokenKind::Semi) => {
                *stream = chk;
                let spanned = Spanned::<(
                    Next<OpenBracket>,
                    TypeRule,
                    Next<Semi>,
                    Next<Int>,
                    Next<CloseBracket>,
                )>::parse(stream)
                .map_err(PErr::fail)?;
                let span = spanned.span();
                let (_open, inner, _semi, size, _close) = spanned.into_inner();
                let Ok(size) = stream.as_str(size).parse::<usize>() else {
                    return Err(PErr::Fail(stream.full_error(
                        "expected a positive integer size for an array type",
                        stream.span(size),
                    )));
                };

                Ok(PType::Array {
                    span,
                    size,
                    inner: Box::new(inner),
                })
            }
            _ => return Err(stream.fail("expected `]` or `; <size>]`")),
        }
    }
}
