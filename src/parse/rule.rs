use crate::{
    diagnostic::{Diag, Msg},
    parse::{
        instr::{Instr, Push},
        matc::{Any, MatchTokenKind},
    },
};
use crate::{
    lex::buffer::{TokenBuffer, TokenId},
    parse::{TokenQuery, PARSE_ERR},
};
use std::{any::type_name, marker::PhantomData};

use super::Ctx;

pub trait ParserRule<'a> {
    /// Modifies the parser token stack.
    ///
    /// Determines if the input should be consumed or not.
    fn apply(
        &mut self,
        _buffer: &'a TokenBuffer<'a>,
        _token: TokenId,
        _stack: &mut Vec<TokenId>,
        _ctx: &mut Ctx<'a>,
    ) -> RuleResult<'a> {
        RuleResult::Continue
    }

    /// Should this rule run again on the next token.
    ///
    /// Queried after a first pass [`ParserRule::eval`] and [`ParserRule::apply`].
    fn finished(&self) -> bool {
        true
    }

    /// Reset any transitive state such as an index into a rule set.
    ///
    /// Used by [`crate::parse::While`] to reset rules (e.g. a [`crate::parse::Seq`] that has
    /// incremented its `index`).
    fn reset(&mut self) {}
}

/// The result of a [`ParserRule::apply`].
#[derive(Debug, Default)]
pub enum RuleResult<'a> {
    /// Consume the current token and continue.
    #[default]
    Consume,
    /// Continue to the next rule without consuming
    /// the current token.
    Continue,
    Failed(Diag<'a>),
}

impl PartialEq for RuleResult<'_> {
    fn eq(&self, other: &Self) -> bool {
        (matches!(self, Self::Consume) && matches!(self, Self::Consume))
            || (matches!(self, Self::Continue) && matches!(self, Self::Continue))
            || (self.failed() && other.failed())
    }
}

impl RuleResult<'_> {
    pub fn failed(&self) -> bool {
        matches!(self, Self::Failed(_))
    }

    pub fn inspect_diag<F>(mut self, f: F) -> Self
    where
        F: Fn(&mut Diag<'_>),
    {
        match &mut self {
            Self::Failed(diag) => {
                f(diag);
            }
            _ => {}
        }

        self
    }
}

/// Consumes next token if it passes the `Next` constraint. Fails otherwise.
pub type Next<Next, Instr = ()> = Rule<Any, Next, Instr>;

/// Consumes next token if it passes the `Next` constraint. Fails otherwise.
pub type Expect<Current, Next, Instr = ()> = Rule<Current, Next, Instr>;

/// Consuming [`ParserRule`] that applies `Instr` if the `Current` and `Next`
/// token constraints match the supplied input.
#[derive(Debug)]
pub struct Rule<Current, Next, Instr, Report = DefaultReport<Current, Next>>(
    PhantomData<(Current, Next, Instr, Report)>,
);

impl<C, N, I, R> Default for Rule<C, N, I, R> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<'a, C, N, I, R> ParserRule<'a> for Rule<C, N, I, R>
where
    C: MatchTokenKind,
    N: MatchTokenKind,
    I: Instr,
    R: ReportDiag,
{
    fn apply(
        &mut self,
        buffer: &'a TokenBuffer<'a>,
        token: TokenId,
        stack: &mut Vec<TokenId>,
        _ctx: &mut Ctx<'a>,
    ) -> RuleResult<'a> {
        let current = stack.last().map(|t| buffer.kind(*t));
        let next = Some(buffer.kind(token));
        if C::matches(current) && N::matches(next) {
            I::apply(buffer, token, stack)
                .ok()
                .map(|_| RuleResult::Consume)
                .unwrap_or_else(|| RuleResult::Failed(R::report(buffer, token, stack)))
        } else {
            RuleResult::Failed(R::report(buffer, token, stack))
        }
    }
}

pub trait ReportDiag {
    fn report<'a>(
        buffer: &'a TokenBuffer<'a>,
        token: TokenId,
        stack: &mut Vec<TokenId>,
    ) -> Diag<'a>;
}

#[derive(Default)]
pub struct DefaultReport<C, N>(PhantomData<(C, N)>);

impl<C, N> ReportDiag for DefaultReport<C, N>
where
    C: MatchTokenKind,
    N: MatchTokenKind,
{
    fn report<'a>(
        buffer: &'a TokenBuffer<'a>,
        token: TokenId,
        stack: &mut Vec<TokenId>,
    ) -> Diag<'a> {
        let (msg, span) = if !N::matches(Some(buffer.kind(token))) {
            (format!("expected {}", type_name::<N>()), buffer.span(token))
        } else {
            if let Some(token) = stack.last() {
                (
                    format!("expected {}", type_name::<C>()),
                    buffer.span(*token),
                )
            } else {
                (format!("expected {}", type_name::<C>()), buffer.span(token))
            }
        };

        Diag::sourced(PARSE_ERR, buffer.source(), Msg::error(span, msg))
    }
}

/// Maintain state that can be mutably accessed during the [`StatefulParserRule::apply_with`] step.
pub struct Stateful<T: StatefulParserRule> {
    state: Option<<T as StatefulParserRule>::State>,
    rules: T,
}

impl<'a, S, T> Default for Stateful<T>
where
    T: Default + StatefulParserRule<State = S>,
{
    fn default() -> Self {
        Self {
            state: None,
            rules: T::default(),
        }
    }
}

impl<'a, S, T> ParserRule<'a> for Stateful<T>
where
    T: StatefulParserRule<State = S>,
{
    fn apply(
        &mut self,
        buffer: &'a TokenBuffer<'a>,
        token: TokenId,
        stack: &mut Vec<TokenId>,
        ctx: &mut Ctx<'a>,
    ) -> RuleResult<'a> {
        if self.state.is_none() {
            self.state = Some(self.rules.init_state());
        }

        self.rules
            .apply_with(self.state.as_mut().unwrap(), buffer, token, stack, ctx)
    }
}

/// Apply a parser rule with some provided state.
///
/// [`StatefulParserRule::init_state`] is called before the first call to [`StatefulParserRule::apply_with`].
///
/// Wrapping a [`ParserRule`] in [`Stateful`] will always use [`StatefulParserRule::apply_with`]
/// over [`ParserRule::apply`].
pub trait StatefulParserRule {
    type State;

    fn init_state(&mut self) -> Self::State;

    fn apply_with<'a>(
        &mut self,
        state: &mut Self::State,
        buffer: &'a TokenBuffer<'a>,
        token: TokenId,
        stack: &mut Vec<TokenId>,
        ctx: &mut Ctx<'a>,
    ) -> RuleResult<'a>;
}

/// Implements the [`ParserRule`] methods for a variadic collection of [`ParserRule`]s.
///
/// ```
/// (Expect<Let, Push>, Expect<Ident, Push>).finished_nth(1);
/// ```
pub trait ParserRuleSet<'a>: ParserRuleSetLen {
    fn apply_nth(
        &mut self,
        n: usize,
        buffer: &'a TokenBuffer<'a>,
        token: TokenId,
        stack: &mut Vec<TokenId>,
        ctx: &mut Ctx<'a>,
    ) -> RuleResult<'a>;

    fn finished_nth(&self, n: usize) -> bool;

    fn reset(&mut self);
}

pub trait ParserRuleSetLen {
    fn len() -> usize;
}

impl<'a, T> ParserRuleSet<'a> for T
where
    T: ParserRule<'a>,
{
    fn apply_nth(
        &mut self,
        _n: usize,
        buffer: &'a TokenBuffer<'a>,
        token: TokenId,
        stack: &mut Vec<TokenId>,
        ctx: &mut Ctx<'a>,
    ) -> RuleResult<'a> {
        self.apply(buffer, token, stack, ctx)
    }

    fn finished_nth(&self, _n: usize) -> bool {
        self.finished()
    }

    fn reset(&mut self) {
        self.reset();
    }
}

impl<'a, T> ParserRuleSetLen for T
where
    T: ParserRule<'a>,
{
    fn len() -> usize {
        1
    }
}

macro_rules! impl_rule_set {
    ($(($n:tt, $T:ident)),*) => {
        #[allow(non_snake_case)]
        impl<'a, $($T),*> ParserRuleSet<'a> for ($($T,)*)
            where
                $($T: ParserRule<'a>),*
        {
            fn apply_nth(
                &mut self,
                n: usize,
                b: &'a TokenBuffer<'a>,
                id: TokenId,
                stk: &mut Vec<TokenId>,
                ctx: &mut Ctx<'a>,
            ) -> RuleResult<'a> {
                let ($($T,)*) = self;
                match n {
                    $($n => $T.apply(b, id, stk, ctx),)*
                    _ => unreachable!()
                }
            }

            fn finished_nth(&self, n: usize) -> bool {
                let ($($T,)*) = self;
                match n {
                    $($n => $T.finished(),)*
                    _ => unreachable!()
                }
            }

            fn reset(&mut self) {
                let ($($T,)*) = self;
                $($T.reset();)*
            }
        }
    };
}

variadics_please::all_tuples_enumerated!(impl_rule_set, 1, 15, T);

macro_rules! impl_rule_set_len {
    ($N:expr, $($T:ident),*) => {
        #[allow(non_snake_case)]
        impl<'a, $($T),*> ParserRuleSetLen for ($($T,)*)
            where
                $($T: ParserRule<'a>),*
        {
            fn len() -> usize {
                $N
            }
        }
    };
}

variadics_please::all_tuples_with_size!(impl_rule_set_len, 1, 15, T);

#[macro_export]
macro_rules! impl_parser_rule_field {
    ($T:ident, |$self:ident| $to_field:expr) => {
        impl<'a> ParserRule<'a> for $T {
            fn apply(
                &mut $self,
                buffer: &'a TokenBuffer<'a>,
                token: TokenId,
                stack: &mut Vec<TokenId>,
                ctx: &mut Ctx<'a>,
            ) -> RuleResult<'a> {
                $to_field.apply(buffer, token, stack, ctx)
            }

            fn finished(&$self) -> bool {
                $to_field.finished()
            }

            fn reset(&mut $self) {
                <_ as ParserRule<'a>>::reset(&mut $to_field);
            }
        }
    };
}
