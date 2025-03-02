use super::{ParserRule, RResult};
use crate::lex::buffer::{Span, TokenQuery};
use crate::parse::matc::Bracket;

#[derive(Debug)]
pub struct Attribute {
    pub span: Span,
    pub str: String,
}

pub struct AttributeRule;

impl<'a> ParserRule<'a> for AttributeRule {
    type Output = Attribute;

    fn parse(
        buffer: &'a crate::lex::buffer::TokenBuffer<'a>,
        stream: &mut crate::parse::stream::TokenStream<'a>,
        _: &mut Vec<crate::lex::buffer::TokenId>,
    ) -> RResult<'a, Self::Output> {
        let _pound = stream.expect();
        let _bracket = stream.expect();
        let offset = stream.find_matched_delim_offset::<Bracket>();
        let mut slice = stream.slice(offset);
        stream.eat_n(offset);
        let _bracket = stream.expect();
        assert_eq!(1, slice.remaining());
        let attribute = slice.expect();
        Ok(Attribute {
            span: buffer.span(attribute),
            str: buffer.ident(attribute).to_string(),
        })
    }
}
