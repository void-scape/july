use crate::diagnostic::{Diag, Msg};
use crate::parse::{Ctx, ParserRule, RuleResult, TokenQuery};
use crate::PARSE_ERR;
use crate::{
    lex::buffer::{TokenBuffer, TokenId},
    ParserRuleSet,
};
use std::any::type_name;

/// Picks the first rule from `T` that does not fail.
///
/// Fails if all rules failed.
#[derive(Debug, Default)]
pub struct Alt<T> {
    chosen: Option<usize>,
    rules: T,
}

impl<'a, T> ParserRule<'a> for Alt<T>
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
        if let Some(chosen) = self.chosen {
            self.rules.apply_nth(chosen, buffer, token, stack, ctx)
        } else {
            for i in 0..T::len() {
                let result = self.rules.apply_nth(i, buffer, token, stack, ctx);
                if !result.failed() {
                    self.chosen = Some(i);
                    return result;
                }
            }

            RuleResult::Failed(Diag::sourced(
                PARSE_ERR,
                buffer.source(),
                Msg::error(
                    buffer.span(token),
                    format!("expected one of {}", type_name::<T>()),
                ),
            ))
        }
    }

    fn finished(&self) -> bool {
        self.chosen.is_none_or(|i| self.rules.finished_nth(i))
    }

    fn reset(&mut self) {
        self.chosen = None;
        self.rules.reset();
    }
}
