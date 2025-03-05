use super::strukt::Struct;
use super::ParserRule;
use crate::ir::ctx::Ctx;
use crate::ir::ident::IdentId;
use crate::ir::ty::store::TyId;
use crate::lex::buffer::{Span, TokenBuffer, TokenId, TokenQuery};
use crate::lex::kind::TokenKind;
use crate::parse::matc::{Any, CloseParen, Comma, Equals, OpenCurly, Semi};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum PType {
    Simple(TokenId),
    Ref { borrow: TokenId, inner: Box<PType> },
}

impl PType {
    pub fn span(&self, buffer: &TokenBuffer) -> Span {
        match self {
            Self::Simple(id) => buffer.span(*id),
            Self::Ref { borrow, inner } => {
                Span::from_spans(buffer.span(*borrow), inner.span(buffer))
            }
        }
    }

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
        }
    }

    pub fn peel_refs(&self) -> &PType {
        match self {
            Self::Ref { inner, .. } => inner.peel_refs(),
            Self::Simple(_) => self,
        }
    }
}

#[derive(Debug, Default)]
pub struct TypeRule;

impl<'a> ParserRule<'a> for TypeRule {
    type Output = PType;

    fn parse(
        _: &'a crate::lex::buffer::TokenBuffer<'a>,
        stream: &mut crate::parse::stream::TokenStream<'a>,
        _: &mut Vec<crate::lex::buffer::TokenId>,
    ) -> super::RResult<'a, Self::Output> {
        let mut slice =
            stream.slice(stream.find_offset::<Any<(Semi, Comma, CloseParen, OpenCurly, Equals)>>());
        stream.eat_until::<Any<(Semi, Comma, CloseParen, OpenCurly, Equals)>>();

        match slice.remaining() {
            1 => Ok(PType::Simple(slice.expect())),
            2 => match slice.peek_kind() {
                Some(TokenKind::Ampersand) => Ok(PType::Ref {
                    borrow: slice.expect(),
                    inner: Box::new(PType::Simple(slice.expect())),
                }),
                _ => Err(stream.error("expected a type")),
            },
            _ => Err(stream.error("expected a type")),
        }
    }
}
