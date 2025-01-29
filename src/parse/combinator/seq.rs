use crate::parse::{Ctx, ParserRule, RuleResult};
use crate::{
    lex::buffer::{TokenBuffer, TokenId},
    ParserRuleSet,
};

/// Apply the rules in set `T` in sequencial order, finishing when
/// the end of set is reached.
#[derive(Debug, Default)]
pub struct Seq<T> {
    index: usize,
    rules: T,
}

impl<T> Seq<T> {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn rules(&self) -> &T {
        &self.rules
    }
}

impl<'a, T> ParserRule<'a> for Seq<T>
where
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

        let result = self.rules.apply_nth(self.index, buffer, token, stack, ctx);
        match result {
            RuleResult::Consume | RuleResult::Continue => {
                if self.rules.finished_nth(self.index) {
                    self.index += 1;
                }
            }
            _ => {}
        }

        result
    }

    fn finished(&self) -> bool {
        self.index >= T::len()
    }

    fn reset(&mut self) {
        self.index = 0;
        self.rules.reset();
    }
}
