use std::ops::Deref;

use crate::lex::buffer::{TokenBuffer, TokenId};
use crate::parse::{Ctx, ParserRule, RuleResult};

/// Track the changes to the stack since the start of a [`ParserRule`].
#[derive(Debug, Default)]
pub struct StackTrack<T> {
    start: Option<i32>,
    current: Option<i32>,
    rules: T,
}

impl<T> StackTrack<T> {
    /// Reports the number of elements pushed onto the stack since the start of `T`.
    ///
    /// Updates during [`ParserRule::apply`].
    pub fn pushed(&self) -> i32 {
        self.current.unwrap() - self.start.unwrap()
    }
}

impl<T> Deref for StackTrack<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.rules
    }
}

impl<'a, T> ParserRule<'a> for StackTrack<T>
where
    T: ParserRule<'a> + Default,
{
    fn apply(
        &mut self,
        buffer: &'a TokenBuffer<'a>,
        token: TokenId,
        stack: &mut Vec<TokenId>,
        ctx: &mut Ctx<'a>,
    ) -> RuleResult<'a> {
        if self.start.is_none() {
            self.start = Some(stack.len() as i32);
        }

        let result = self.rules.apply(buffer, token, stack, ctx);
        self.current = Some(stack.len() as i32);

        result
    }

    fn finished(&self) -> bool {
        self.rules.finished()
    }

    fn reset(&mut self) {
        self.start = None;
        self.rules.reset();
    }
}
