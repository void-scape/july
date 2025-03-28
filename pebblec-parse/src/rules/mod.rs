use crate::{
    diagnostic::Diag,
    matc::{Any, MatchTokenKind},
};
use crate::{
    lex::buffer::TokenId,
    {TokenQuery, stream::TokenStream},
};
use std::marker::PhantomData;

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
mod uze;

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
    pub use super::uze::*;
}

pub trait ParserRule<'a> {
    type Output;

    fn parse(stream: &mut TokenStream<'a>) -> RResult<Self::Output>;
}

pub type RResult<T> = Result<T, PErr>;

#[derive(Debug)]
pub enum PErr {
    Recover(Diag),
    Fail(Diag),
}

impl PErr {
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
            Self::Recover(diag) => Self::Recover(diag.join(other.into_diag())),
            Self::Fail(diag) => Self::Fail(diag.join(other.into_diag())),
        }
    }

    pub fn into_diag(self) -> Diag {
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

impl<'a, C, N, R> ParserRule<'a> for Rule<C, N, R>
where
    C: MatchTokenKind,
    N: MatchTokenKind,
    R: ReportDiag,
{
    type Output = TokenId;

    #[track_caller]
    fn parse(stream: &mut TokenStream<'a>) -> RResult<TokenId> {
        match (stream.next(), stream.peek()) {
            (Some(c), Some(n)) => {
                if C::matches(Some(stream.kind(c))) && N::matches(Some(stream.kind(n))) {
                    Ok(c)
                } else {
                    Err(R::report(stream, Some(c), stream.prev()))
                }
            }
            (Some(c), None) => {
                if C::matches(Some(stream.kind(c))) && N::matches(None) {
                    Ok(c)
                } else {
                    Err(R::report(stream, Some(c), stream.prev()))
                }
            }
            (None, _) => Err(R::report(stream, None, stream.prev())),
        }
    }
}

pub trait ReportDiag {
    fn report(stream: &TokenStream, token: Option<TokenId>, prev: TokenId) -> PErr;
}

#[derive(Default)]
pub struct DefaultReport<C, N>(PhantomData<(C, N)>);

impl<C, N> ReportDiag for DefaultReport<C, N>
where
    C: MatchTokenKind,
    N: MatchTokenKind,
{
    #[track_caller]
    fn report(stream: &TokenStream, token: Option<TokenId>, prev: TokenId) -> PErr {
        let (msg, span) = if let Some(token) = token {
            (C::expect(), stream.span(token))
        } else {
            (C::expect(), stream.span(prev))
        };

        PErr::Recover(stream.report_error(msg, span))
    }
}

macro_rules! impl_parser_rule {
    ($($T:ident),*) => {
        impl<'a, 's, $($T,)*> ParserRule<'a> for ($($T,)*)
        where
            $($T: ParserRule<'a>,)*
        {
            type Output = ($(<$T as ParserRule<'a>>::Output,)*);

            #[track_caller]
            fn parse(
                stream: &mut TokenStream<'a>,
            ) -> RResult<Self::Output> {
                Ok(($(
                    $T::parse(stream)?,
                )*))
            }
        }
    };
}

variadics_please::all_tuples!(impl_parser_rule, 1, 10, T);
