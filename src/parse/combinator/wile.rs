use crate::lex::buffer::{TokenBuffer, TokenId, TokenQuery};
use crate::parse::matc::MatchTokenKind;
use crate::parse::{rules::*, stream::TokenStream};

/// Evaluates if the [`While`] loop should continue.
pub trait Condition<'a> {
    fn eval(
        buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        stack: &mut Vec<TokenId>,
    ) -> bool;
}

/// Evaluates `true` if the next token matches `T`.
#[derive(Debug, Default)]
pub struct NextToken<T>(T);

impl<'a, T> Condition<'a> for NextToken<T>
where
    T: MatchTokenKind,
{
    fn eval(
        buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        _stack: &mut Vec<TokenId>,
    ) -> bool {
        T::matches(stream.peek().map(|t| buffer.kind(t)))
    }
}

/// Accumulates `T` results while `C` is true.
#[derive(Debug, Default)]
pub struct While<C, T> {
    condition: C,
    rules: T,
}

impl<'a, C, T> ParserRule<'a> for While<C, T>
where
    C: Condition<'a>,
    T: ParserRule<'a>,
{
    type Output = Vec<<T as ParserRule<'a>>::Output>;

    fn parse(
        buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        stack: &mut Vec<TokenId>,
    ) -> RResult<'a, Self::Output> {
        let mut results = Vec::new();
        while C::eval(buffer, stream, stack) {
            results.push(T::parse(buffer, stream, stack)?);
        }
        Ok(results)
    }
}
