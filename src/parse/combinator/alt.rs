use crate::lex::buffer::{TokenBuffer, TokenId};
use crate::parse::{rules::*, stream::TokenStream};

/// Returns the first successful rule from `T`.
///
/// Fails if all rules failed.
#[derive(Debug, Default)]
pub struct Alt<T>(T);

impl<'a, O, A, B> ParserRule<'a> for Alt<(A, B)>
where
    A: ParserRule<'a, Output = O>,
    B: ParserRule<'a, Output = O>,
{
    type Output = O;

    fn parse(
        buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        stack: &mut Vec<TokenId>,
    ) -> RResult<'a, Self::Output> {
        let str = *stream;
        match A::parse(buffer, stream, stack) {
            Err(_) => {
                *stream = str;
                B::parse(buffer, stream, stack)
            }
            Ok(val) => Ok(val),
        }
    }
}

impl<'a, O, A, B, C> ParserRule<'a> for Alt<(A, B, C)>
where
    A: ParserRule<'a, Output = O>,
    B: ParserRule<'a, Output = O>,
    C: ParserRule<'a, Output = O>,
{
    type Output = O;

    fn parse(
        buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        stack: &mut Vec<TokenId>,
    ) -> RResult<'a, Self::Output> {
        let str = *stream;
        match A::parse(buffer, stream, stack) {
            Err(_) => {
                *stream = str;
                match B::parse(buffer, stream, stack) {
                    Err(_) => {
                        *stream = str;
                        C::parse(buffer, stream, stack)
                    }
                    Ok(val) => Ok(val),
                }
            }
            Ok(val) => Ok(val),
        }
    }
}

impl<'a, O, A, B, C, D> ParserRule<'a> for Alt<(A, B, C, D)>
where
    A: ParserRule<'a, Output = O>,
    B: ParserRule<'a, Output = O>,
    C: ParserRule<'a, Output = O>,
    D: ParserRule<'a, Output = O>,
{
    type Output = O;

    fn parse(
        buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        stack: &mut Vec<TokenId>,
    ) -> RResult<'a, Self::Output> {
        let str = *stream;
        match A::parse(buffer, stream, stack) {
            Err(_) => {
                *stream = str;
                match B::parse(buffer, stream, stack) {
                    Err(_) => {
                        *stream = str;
                        match C::parse(buffer, stream, stack) {
                            Err(_) => {
                                *stream = str;
                                D::parse(buffer, stream, stack)
                            }
                            Ok(val) => Ok(val),
                        }
                    }
                    Ok(val) => Ok(val),
                }
            }
            Ok(val) => Ok(val),
        }
    }
}
