use super::{PErr, ParserRule, RResult};
use crate::lex::buffer::{Span, TokenId, TokenQuery};
use crate::matc::Bracket;
use crate::stream::TokenStream;

// TODO: parse these suckers
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attribute {
    pub span: Span,
    pub pound: TokenId,
    pub tokens: Vec<TokenId>,
}

pub struct AttributeRule;

impl<'a, 's> ParserRule<'a, 's> for AttributeRule {
    type Output = Attribute;

    fn parse(stream: &mut TokenStream<'a, 's>) -> RResult<'s, Self::Output> {
        let pound = stream.expect();
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
                )));
            }
            1 => Ok(Attribute {
                pound,
                span: Span::from_spans(stream.span(pound), stream.span(tokens[0])),
                tokens,
            }),
            _ => Ok(Attribute {
                pound,
                span: Span::from_spans(stream.span(pound), stream.span(*tokens.last().unwrap())),
                tokens,
            }),
        }
    }
}
