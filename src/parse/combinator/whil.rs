use crate::parse::{Ctx, ParserRule, RuleResult, TokenQuery};
use crate::ParserRuleSet;
use crate::{
    lex::buffer::{TokenBuffer, TokenId},
    matc::MatchTokenKind,
};
use std::marker::PhantomData;

/// Evaluates whether a [`While`] loop should continue.
pub trait Condition {
    fn eval(&self, buffer: &TokenBuffer, token: TokenId, stack: &mut Vec<TokenId>) -> bool;
}

#[derive(Debug, Default)]
pub struct True;

impl Condition for True {
    fn eval(&self, _buffer: &TokenBuffer, _token: TokenId, _stack: &mut Vec<TokenId>) -> bool {
        true
    }
}

/// Evaluates true if the next input token fulfils the constraint `T`.
#[derive(Debug, Default)]
pub struct NextToken<T>(PhantomData<T>);

impl<T> Condition for NextToken<T>
where
    T: MatchTokenKind,
{
    fn eval(&self, buffer: &TokenBuffer, token: TokenId, _stack: &mut Vec<TokenId>) -> bool {
        T::matches(Some(buffer.kind(token)))
    }
}

/// Apply rules `T` until `T` fails.
pub type Loop<T> = While<True, T>;

/// Apply rule set `T` while the condition `C` is met, or until `T` failes.
#[derive(Debug, Default)]
pub struct While<C, T> {
    index: usize,
    finished: bool,
    condition: C,
    rules: T,
}

impl<'a, C, T> ParserRule<'a> for While<C, T>
where
    C: Condition + Default,
    T: ParserRuleSet<'a> + Default,
{
    fn apply(
        &mut self,
        buffer: &'a TokenBuffer<'a>,
        token: TokenId,
        stack: &mut Vec<TokenId>,
        ctx: &mut Ctx<'a>,
    ) -> RuleResult<'a> {
        assert!(!self.finished());

        if !self.condition.eval(buffer, token, stack) {
            self.finished = true;
            return RuleResult::Continue;
        }

        if self.index >= T::len() {
            self.index = 0;
            self.rules.reset();
        }

        let result = self.rules.apply_nth(self.index, buffer, token, stack, ctx);
        match result {
            RuleResult::Consume | RuleResult::Continue => {
                if self.rules.finished_nth(self.index) {
                    self.index += 1;
                }
            }
            RuleResult::Failed(diag) => {
                return RuleResult::Failed(diag);
                //println!("WASTED ERROR");
                //self.finished = true;
                //return RuleResult::Continue;
            }
        }

        result
    }

    fn finished(&self) -> bool {
        self.finished
    }

    fn reset(&mut self) {
        self.index = 0;
        self.finished = false;
    }
}
