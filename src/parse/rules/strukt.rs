use super::{Next, ParserRule, RResult};
use crate::lex::buffer::*;
use crate::parse::matc::{self, *};
use crate::parse::stream::TokenStream;

/// Structured data fields.
#[derive(Debug)]
pub struct Struct {
    name: TokenId,
}

/// <Ident> : {}
#[derive(Debug, Default)]
pub struct StructRule;

impl<'a> ParserRule<'a> for StructRule {
    type Output = Struct;

    fn parse(
        buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        stack: &mut Vec<TokenId>,
    ) -> RResult<'a, Self::Output> {
        let (name, _, _, _, _) = <(
            Next<Ident>,
            Next<Colon>,
            Next<matc::Struct>,
            Next<OpenCurly>,
            Next<CloseCurly>,
        ) as ParserRule>::parse(buffer, stream, stack)?;
        Ok(Struct { name })
    }
}
