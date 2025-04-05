use super::expr::{Path, PathRules};
use super::{Next, ParserRule, RResult};
use crate::combinator::prelude::*;
use crate::lex::buffer::*;
use crate::lex::kind;
use crate::lex::kind::Semi;
use crate::rules::PErr;
use crate::stream::TokenStream;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Use {
    pub span: Span,
    pub uze: TokenId,
    pub path: Path,
}

#[derive(Debug, Default)]
pub struct UseRule;

impl<'a> ParserRule<'a> for UseRule {
    type Output = Use;

    fn parse(stream: &mut TokenStream<'a>) -> RResult<Self::Output> {
        if !stream.match_peek::<kind::Use>() {
            return Err(PErr::Recover(stream.error("expected `use`")));
        }

        let spanned = Spanned::<(Next<kind::Use>, PathRules<Semi>, Next<Semi>)>::parse(stream)
            .map_err(PErr::fail)?;
        let span = spanned.span();
        let (uze, path, _) = spanned.into_inner();

        Ok(Use { span, uze, path })
    }
}
