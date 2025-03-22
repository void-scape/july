use crate::lex::buffer::TokenQuery;
use crate::matc::MatchTokenKind;
use crate::{rules::*, stream::TokenStream};
use std::marker::PhantomData;

/// Evaluates if the [`While`] loop should continue.
pub trait Condition<'a, 's> {
    fn eval(stream: &mut TokenStream<'a, 's>) -> bool;
}

/// Evaluates `true` if the next token matches `T`.
#[derive(Debug, Default)]
pub struct NextToken<T>(T);

impl<'a, 's, T> Condition<'a, 's> for NextToken<T>
where
    T: MatchTokenKind,
{
    fn eval(stream: &mut TokenStream<'a, 's>) -> bool {
        T::matches(stream.peek().map(|t| stream.kind(t)))
    }
}

/// Evaluates `true` if the remaining tokens are equal to or less than `N`.
#[derive(Debug, Default)]
pub struct Remaining<const N: usize>(PhantomData<[u8; N]>);

impl<'a, 's, const N: usize> Condition<'a, 's> for Remaining<N> {
    fn eval(stream: &mut TokenStream<'a, 's>) -> bool {
        stream.remaining() <= N
    }
}

/// Evaluates `true` if the remaining tokens are greater than or equal to `N`.
#[derive(Debug, Default)]
pub struct Atleast<const N: usize>(PhantomData<[u8; N]>);

impl<'a, 's, const N: usize> Condition<'a, 's> for Atleast<N> {
    fn eval(stream: &mut TokenStream<'a, 's>) -> bool {
        stream.remaining() >= N
    }
}

/// Accumulates `T` results while `C` is true.
#[derive(Debug, Default)]
pub struct While<C, T> {
    _condition: C,
    _rules: T,
}

impl<'a, 's, C, T> ParserRule<'a, 's> for While<C, T>
where
    C: Condition<'a, 's>,
    T: ParserRule<'a, 's>,
{
    type Output = Vec<<T as ParserRule<'a, 's>>::Output>;

    #[track_caller]
    fn parse(stream: &mut TokenStream<'a, 's>) -> RResult<'s, Self::Output> {
        let mut results = Vec::new();
        while !stream.is_empty() && C::eval(stream) {
            results.push(T::parse(stream)?);
        }
        Ok(results)
    }
}
