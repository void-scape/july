use crate::lex::buffer::{TokenBuffer, TokenId};
use crate::parse::{Ctx, ParserRule, RuleResult};

/// Continues if `T` fails, otherwise applies `T`.
#[derive(Debug, Default)]
pub struct Opt<T> {
    some: bool,
    rules: T,
}

impl<'a, T> ParserRule<'a> for Opt<T>
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
        match self.rules.apply(buffer, token, stack, ctx) {
            RuleResult::Failed(diag) => {
                if self.some {
                    RuleResult::Failed(diag)
                } else {
                    RuleResult::Continue
                }
            }
            res => {
                self.some = true;
                res
            }
        }
    }

    fn finished(&self) -> bool {
        !self.some || self.rules.finished()
    }

    fn reset(&mut self) {
        self.some = false;
    }
}
