use super::{ParserRule, RResult};
use crate::lex::buffer::{Span, TokenId, TokenQuery};
use crate::parse::matc::Bracket;

// TODO: parse these suckers
#[derive(Debug, Clone)]
pub struct Attribute {
    pub span: Span,
    pub tokens: Vec<TokenId>,
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

        let tokens = slice.drain();
        match tokens.len() {
            0 => {
                return Err(stream.full_error(
                    "expected atleast one argument in attribute",
                    buffer.span(stream.prev()),
                    "",
                ))
            }
            1 => Ok(Attribute {
                span: buffer.span(tokens[0]),
                tokens,
            }),
            _ => Ok(Attribute {
                span: Span::from_spans(
                    buffer.span(tokens[0]),
                    buffer.span(*tokens.last().unwrap()),
                ),
                tokens,
            }),
        }
    }
}
