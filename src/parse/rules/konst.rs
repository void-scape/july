use super::expr::{Expr, ExprRule};
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
    pub expr: Expr,
}

#[derive(Default)]
pub struct ConstRule;

impl<'a, 's> ParserRule<'a, 's> for ConstRule {
    type Output = Const;

    fn parse(stream: &mut TokenStream<'a, 's>) -> RResult<'s, Self::Output> {
        let spanned = Spanned::<(
            Next<Ident>,
            Next<Colon>,
            Next<matc::Const>,
            TypeRule,
            Next<Equals>,
            ExprRule,
            Next<Semi>,
        )>::parse(stream)?;
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
