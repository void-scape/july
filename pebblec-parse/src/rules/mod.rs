use crate::{
    diagnostic::{Diag, Msg},
    lex::buffer::Buffer,
    matc::{Any, MatchTokenKind},
};
use crate::{
    lex::buffer::{TokenBuffer, TokenId},
    {TokenQuery, stream::TokenStream},
};
use std::{marker::PhantomData, panic::Location};

mod arr;
mod attributes;
mod block;
mod enom;
mod expr;
mod func;
mod impul;
mod konst;
mod stmt;
mod strukt;
mod types;

#[allow(unused)]
pub mod prelude {
    pub use super::arr::*;
    pub use super::attributes::*;
    pub use super::block::*;
    pub use super::enom::*;
    pub use super::expr::*;
    pub use super::func::*;
    pub use super::impul::*;
    pub use super::konst::*;
    pub use super::stmt::*;
    pub use super::strukt::*;
    pub use super::types::*;
}

pub trait ParserRule<'a, 's> {
    type Output;

    fn parse(stream: &mut TokenStream<'a, 's>) -> RResult<'s, Self::Output>;
}

pub type RResult<'a, T> = Result<T, PErr<'a>>;

#[derive(Debug)]
pub enum PErr<'a> {
    Recover(Diag<'a>),
    Fail(Diag<'a>),
}

impl<'a> PErr<'a> {
    pub fn recoverable(&self) -> bool {
        matches!(self, Self::Recover(_))
    }

    pub fn recover(self) -> Self {
        match self {
            Self::Fail(diag) => Self::Recover(diag),
            Self::Recover(diag) => Self::Recover(diag),
        }
    }

    pub fn fail(self) -> Self {
        match self {
            Self::Recover(diag) => Self::Fail(diag),
            Self::Fail(diag) => Self::Fail(diag),
        }
    }

    pub fn wrap(self, other: Self) -> Self {
        match self {
            Self::Recover(diag) => Self::Recover(diag.wrap(other.into_diag())),
            Self::Fail(diag) => Self::Fail(diag.wrap(other.into_diag())),
        }
    }

    pub fn into_diag(self) -> Diag<'a> {
        match self {
            Self::Recover(diag) => diag,
            Self::Fail(diag) => diag,
        }
    }
}

/// Consumes next token if it passes the `Next` constraint. Fails otherwise.
pub type Next<Next> = Rule<Next, Any>;

/// Consuming [`ParserRule`] that applies `Instr` if the `Current` and `Next`
/// token constraints match the supplied input.
///
/// [`Resume`]s on the next token.
#[derive(Debug)]
pub struct Rule<Current, Next, Report = DefaultReport<Current, Next>>(
    PhantomData<(Current, Next, Report)>,
);

impl<C, N, R> Default for Rule<C, N, R> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<'a, 's, C, N, R> ParserRule<'a, 's> for Rule<C, N, R>
where
    C: MatchTokenKind,
    N: MatchTokenKind,
    R: ReportDiag,
{
    type Output = TokenId;

    #[track_caller]
    fn parse(stream: &mut TokenStream<'a, 's>) -> RResult<'s, TokenId> {
        match (stream.next(), stream.peek()) {
            (Some(c), Some(n)) => {
                if C::matches(Some(stream.kind(c))) && N::matches(Some(stream.kind(n))) {
                    Ok(c)
                } else {
                    Err(R::report(stream.token_buffer(), Some(c), stream.prev()))
                }
            }
            (Some(c), None) => {
                if C::matches(Some(stream.kind(c))) && N::matches(None) {
                    Ok(c)
                } else {
                    Err(R::report(stream.token_buffer(), Some(c), stream.prev()))
                }
            }
            (None, _) => Err(R::report(stream.token_buffer(), None, stream.prev())),
        }
    }
}

pub trait ReportDiag {
    fn report<'a, 's>(
        buffer: &'a TokenBuffer<'s>,
        token: Option<TokenId>,
        prev: TokenId,
    ) -> PErr<'s>;
}

#[derive(Default)]
pub struct DefaultReport<C, N>(PhantomData<(C, N)>);

impl<C, N> ReportDiag for DefaultReport<C, N>
where
    C: MatchTokenKind,
    N: MatchTokenKind,
{
    #[track_caller]
    fn report<'a, 's>(
        buffer: &'a TokenBuffer<'s>,
        token: Option<TokenId>,
        prev: TokenId,
    ) -> PErr<'s> {
        let (msg, span) = if let Some(token) = token {
            (C::expect(), buffer.span(token))
        } else {
            (C::expect(), buffer.span(prev))
        };

        PErr::Recover(
            Diag::sourced(msg, buffer.source(), Msg::error_span(span)).loc(Location::caller()),
        )
    }
}

macro_rules! impl_parser_rule {
    ($($T:ident),*) => {
        impl<'a, 's, $($T,)*> ParserRule<'a, 's> for ($($T,)*)
        where
            $($T: ParserRule<'a, 's>,)*
        {
            type Output = ($(<$T as ParserRule<'a, 's>>::Output,)*);

            #[track_caller]
            fn parse(
                stream: &mut TokenStream<'a, 's>,
            ) -> RResult<'s, Self::Output> {
                Ok(($(
                    $T::parse(stream)?,
                )*))
            }
        }
    };
}

variadics_please::all_tuples!(impl_parser_rule, 1, 10, T);
