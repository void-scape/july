use super::types::{PType, TypeRule};
use super::{Next, ParserRule, RResult};
use crate::lex::buffer::*;
use crate::parse::matc;
use crate::parse::stream::TokenStream;
use crate::parse::{combinator::prelude::*, matc::*};

#[derive(Debug)]
pub struct Const {
    pub span: Span,
    pub name: TokenId,
    pub ty: PType,
    pub expr: TokenId,
}

/// <ident> : const <type> = <const-expr>;
#[derive(Default)]
pub struct ConstRule;

impl<'a> ParserRule<'a> for ConstRule {
    type Output = Const;

    fn parse(
        buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        stack: &mut Vec<TokenId>,
    ) -> RResult<'a, Self::Output> {
        let spanned = Spanned::<(
            //Reported<Next<Fn>, MissingFn>,
            Next<Ident>,
            Next<Colon>,
            Next<matc::Const>,
            TypeRule,
            Next<Equals>,
            Next<Int>,
            Next<Semi>,
        )>::parse(buffer, stream, stack)?;
        let span = spanned.span();
        let (name, _colon, _const, ty, _eq, expr, _semi) = spanned.into_inner();
        Ok(Const {
            span,
            name,
            ty,
            expr,
        })
    }
}
