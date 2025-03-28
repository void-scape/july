use super::expr::{Expr, ExprRule};
use super::types::{PType, TypeRule};
use super::{Next, ParserRule, RResult};
use crate::combinator::prelude::*;
use crate::lex::kind::*;
use crate::lex::{buffer::*, kind};
use crate::stream::TokenStream;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Const {
    pub span: Span,
    pub name: TokenId,
    pub ty: PType,
    pub expr: Expr,
}

#[derive(Default)]
pub struct ConstRule;

impl<'a, 's> ParserRule<'a> for ConstRule {
    type Output = Const;

    fn parse(stream: &mut TokenStream<'a>) -> RResult<Self::Output> {
        let spanned = Spanned::<(
            Next<Ident>,
            Next<Colon>,
            Next<kind::Const>,
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
