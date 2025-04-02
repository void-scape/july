use super::{Next, ParserRule, RResult};
use crate::lex::buffer::*;
use crate::lex::kind;
use crate::lex::kind::*;
use crate::rules::PErr;
use crate::stream::TokenStream;
use crate::{combinator::prelude::*, matc::*, rules::prelude::*};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Impl {
    pub span: Span,
    pub impul: TokenId,
    pub ident: TokenId,
    pub funcs: Vec<Func>,
}

#[derive(Debug, Default)]
pub struct ImplRule;

impl<'a, 's> ParserRule<'a> for ImplRule {
    type Output = Impl;

    fn parse(stream: &mut TokenStream<'a>) -> RResult<Self::Output> {
        if !stream.match_peek::<kind::Impl>() {
            return Err(PErr::Recover(stream.error("expected `impl`")));
        }

        let (impul, ident, _) = <(Next<kind::Impl>, Next<Ident>, Next<OpenCurly>)>::parse(stream)
            .map_err(PErr::fail)?;

        let offset = stream.find_matched_delim_offset::<Curly>();
        let mut slice = stream.slice(offset);
        stream.eat_n(offset + 1);

        let spanned =
            Spanned::<While<Atleast<1>, FnRule>>::parse(&mut slice).map_err(PErr::fail)?;
        let span = spanned.span();
        let funcs = spanned.into_inner();

        Ok(Impl { span, impul, ident, funcs })
    }
}
