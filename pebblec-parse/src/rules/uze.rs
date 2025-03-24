use super::{Next, ParserRule, RResult};
use crate::combinator::prelude::*;
use crate::lex::buffer::*;
use crate::lex::kind;
use crate::lex::kind::Colon;
use crate::lex::kind::Ident;
use crate::lex::kind::Semi;
use crate::matc::Not;
use crate::rules::PErr;
use crate::stream::TokenStream;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Use {
    pub span: Span,
    pub uze: TokenId,
    pub path: Vec<TokenId>,
}

#[derive(Debug, Default)]
pub struct UseRule;

impl<'a, 's> ParserRule<'a, 's> for UseRule {
    type Output = Use;

    fn parse(stream: &mut TokenStream<'a, 's>) -> RResult<'s, Self::Output> {
        if !stream.match_peek::<kind::Use>() {
            return Err(PErr::Recover(stream.error("expected `use`")));
        }

        let spanned = Spanned::<(
            Next<kind::Use>,
            While<NextToken<Not<Semi>>, (Next<Ident>, Opt<(Next<Colon>, Next<Colon>)>)>,
            Next<Semi>,
        )>::parse(stream)
        .map_err(PErr::fail)?;
        let span = spanned.span();
        let (uze, source, _) = spanned.into_inner();
        let path = source.into_iter().map(|(step, _)| step).collect();

        Ok(Use { span, uze, path })
    }
}
