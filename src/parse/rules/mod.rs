use crate::diagnostic::Diagnostic;
use crate::{
    diagnostic::{Diag, Msg},
    parse::matc::{Any, MatchTokenKind},
};
use crate::{
    lex::buffer::{TokenBuffer, TokenId},
    parse::{stream::TokenStream, TokenQuery, PARSE_ERR},
};
use std::marker::PhantomData;

mod block;
mod expr;
mod strukt;
mod func;
mod stmt;

#[allow(unused)]
pub mod prelude {
    pub use super::strukt::*;
    pub use super::block::*;
    pub use super::expr::*;
    pub use super::func::*;
    pub use super::stmt::*;
}

pub trait ParserRule<'a> {
    type Output;

    fn parse(
        buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        stack: &mut Vec<TokenId>,
    ) -> RResult<'a, Self::Output>;
}

pub type RResult<'a, T> = Result<T, RuleErr<'a>>;

#[derive(Debug)]
pub struct RuleErr<'a>(Vec<Diag<'a>>);

impl<'a> RuleErr<'a> {
    pub fn new(title: &'static str, diag: impl Diagnostic<'a>) -> Self {
        Self(vec![Diag::new(title, diag)])
    }

    pub fn from_diag(diag: Diag<'a>) -> Self {
        Self(vec![diag])
    }

    pub fn from_diags(diags: impl Iterator<Item = Diag<'a>>) -> Self {
        Self(diags.collect())
    }

    pub fn diag(&mut self, diag: Diag<'a>) {
        self.0.push(diag);
    }

    pub fn into_diags(self) -> Vec<Diag<'a>> {
        self.0
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

    fn parse(
        buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        _stack: &mut Vec<TokenId>,
    ) -> RResult<'a, TokenId> {
        match (stream.next(), stream.peek()) {
            (Some(c), Some(n)) => {
                if C::matches(Some(buffer.kind(c))) && N::matches(Some(buffer.kind(n))) {
                    Ok(c)
                } else {
                    Err(RuleErr::from_diag(R::report(buffer, Some(c))))
                }
            }
            (Some(c), None) => {
                if C::matches(Some(buffer.kind(c))) && N::matches(None) {
                    Ok(c)
                } else {
                    Err(RuleErr::from_diag(R::report(buffer, Some(c))))
                }
            }
            (None, _) => Err(RuleErr::from_diag(R::report(buffer, None))),
        }
    }
}

pub trait ReportDiag {
    fn report<'a>(buffer: &'a TokenBuffer<'a>, token: Option<TokenId>) -> Diag<'a>;
}

#[derive(Default)]
pub struct DefaultReport<C, N>(PhantomData<(C, N)>);

impl<C, N> ReportDiag for DefaultReport<C, N>
where
    C: MatchTokenKind,
    N: MatchTokenKind,
{
    fn report<'a>(buffer: &'a TokenBuffer<'a>, token: Option<TokenId>) -> Diag<'a> {
        let (msg, span) = if let Some(token) = token {
            (C::expect(), buffer.span(token))
        } else {
            panic!("not enough information");
        };

        Diag::sourced(PARSE_ERR, buffer.source(), Msg::error(span, msg))
    }
}

macro_rules! impl_parser_rule {
    ($($T:ident),*) => {
        impl<'a, $($T,)*> ParserRule<'a> for ($($T,)*)
        where
            $($T: ParserRule<'a>,)*
        {
            type Output = ($(<$T as ParserRule<'a>>::Output,)*);

            fn parse(
                buffer: &'a TokenBuffer<'a>,
                stream: &mut TokenStream<'a>,
                stack: &mut Vec<TokenId>,
            ) -> RResult<'a, Self::Output> {
                Ok(($(
                    $T::parse(buffer, stream, stack)?,
                )*))
            }
        }
    };
}

variadics_please::all_tuples!(impl_parser_rule, 1, 10, T);
