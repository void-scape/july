use super::{PErr, ParserRule, RResult};
use crate::lex::buffer::{Span, TokenId, TokenQuery};
use crate::parse::matc::Bracket;
use crate::parse::stream::TokenStream;

// TODO: parse these suckers
#[derive(Debug, Clone)]
pub struct Attribute {
    pub span: Span,
    pub tokens: Vec<TokenId>,
}

pub struct AttributeRule;

impl<'a, 's> ParserRule<'a, 's> for AttributeRule {
    type Output = Attribute;

    fn parse(stream: &mut TokenStream<'a, 's>) -> RResult<'s, Self::Output> {
        let _pound = stream.expect();
        let _bracket = stream.expect();
        let offset = stream.find_matched_delim_offset::<Bracket>();
        let mut slice = stream.slice(offset);
        stream.eat_n(offset);
        let _bracket = stream.expect();

        let tokens = slice.drain();
        match tokens.len() {
            0 => {
                return Err(PErr::Fail(stream.full_error(
                    "expected atleast one argument in attribute",
                    stream.span(stream.prev()),
                )))
            }
            1 => Ok(Attribute {
                span: stream.span(tokens[0]),
                tokens,
            }),
            _ => Ok(Attribute {
                span: Span::from_spans(
                    stream.span(tokens[0]),
                    stream.span(*tokens.last().unwrap()),
                ),
                tokens,
            }),
        }
    }
}
