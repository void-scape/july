use super::{Next, ParserRule, RResult};
use crate::lex::buffer::*;
use crate::matc;
use crate::rules::PErr;
use crate::stream::TokenStream;
use crate::{combinator::prelude::*, matc::*, rules::prelude::*};

#[allow(unused)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Impl {
    span: Span,
    ident: TokenId,
    funcs: Vec<Func>,
}

#[derive(Debug, Default)]
pub struct ImplRule;

impl<'a, 's> ParserRule<'a, 's> for ImplRule {
    type Output = Impl;

    fn parse(stream: &mut TokenStream<'a, 's>) -> RResult<'s, Self::Output> {
        if !stream.match_peek::<matc::Impl>() {
            return Err(PErr::Recover(stream.error("expected `impl`")));
        }

        let (_, ident, _) = <(Next<matc::Impl>, Next<Ident>, Next<OpenCurly>)>::parse(stream)?;

        let offset = stream.find_matched_delim_offset::<Curly>();
        let mut slice = stream.slice(offset);
        stream.eat_n(offset + 1);

        let spanned =
            Spanned::<While<Atleast<1>, FnRule>>::parse(&mut slice).map_err(PErr::fail)?;
        let span = spanned.span();
        let funcs = spanned.into_inner();

        Ok(Impl { span, ident, funcs })
    }
}
