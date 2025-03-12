use super::strukt::Struct;
use super::{Next, PErr, ParserRule};
use crate::ir::ctx::Ctx;
use crate::ir::ident::IdentId;
use crate::lex::buffer::{Span, TokenBuffer, TokenId, TokenQuery};
use crate::lex::kind::TokenKind;
use crate::parse::combinator::spanned::Spanned;
use crate::parse::matc::{
    Any, CloseBracket, CloseCurly, CloseParen, Comma, Equals, Int, OpenBracket, OpenCurly, Semi,
};
use crate::parse::stream::TokenStream;
use std::collections::HashMap;

#[derive(Debug, Clone)]
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
}

impl PType {
    pub fn span(&self, buffer: &TokenBuffer) -> Span {
        match self {
            Self::Simple(id) => buffer.span(*id),
            Self::Ref { borrow, inner } => {
                Span::from_spans(buffer.span(*borrow), inner.span(buffer))
            }
            Self::Array { span, .. } => *span,
        }
    }

    /// TODO: this does not consider indirection, this is strictly for descending type relationships for
    /// type sizing
    pub fn retrieve_struct<'a>(
        &self,
        ctx: &mut Ctx<'a>,
        structs: &HashMap<IdentId, &'a Struct>,
    ) -> Option<&'a Struct> {
        match self {
            Self::Simple(id) => {
                let ident = ctx.store_ident(*id).id;
                structs.get(&ident).map(|s| *s)
            }
            Self::Ref { inner, .. } => inner.retrieve_struct(ctx, structs),
            Self::Array { inner, .. } => inner.retrieve_struct(ctx, structs),
        }
    }

    pub fn peel_refs(&self) -> &PType {
        match self {
            Self::Ref { inner, .. } => inner.peel_refs(),
            Self::Simple(_) | Self::Array { .. } => self,
        }
    }
}

#[derive(Debug, Default)]
pub struct TypeRule;

impl<'a, 's> ParserRule<'a, 's> for TypeRule {
    type Output = PType;

    fn parse(stream: &mut TokenStream<'a, 's>) -> super::RResult<'s, Self::Output> {
        let mut slice =
            stream.slice(
                stream
                    .find_offset::<Any<(Semi, Comma, CloseParen, OpenCurly, CloseCurly, Equals)>>(),
            );
        stream.eat_until::<Any<(Semi, Comma, CloseParen, OpenCurly, CloseCurly, Equals)>>();

        match slice.remaining() {
            0 => {
                return Err(PErr::Recover(
                    stream.full_error("expected type", stream.span(stream.prev())),
                ))
            }
            1 => Ok(PType::Simple(slice.expect())),
            2 => match slice.peek_kind() {
                Some(TokenKind::Ampersand) => Ok(PType::Ref {
                    borrow: slice.expect(),
                    inner: Box::new(PType::Simple(slice.expect())),
                }),
                _ => Err(PErr::Fail(stream.error("expected a type"))),
            },
            _ => {
                let spanned =
                    Spanned::<(Next<OpenBracket>, Next<Int>, Next<CloseBracket>, TypeRule)>::parse(
                        &mut slice,
                    )?;
                let span = spanned.span();
                let (_open, size, _close, inner) = spanned.into_inner();
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
        }
    }
}
