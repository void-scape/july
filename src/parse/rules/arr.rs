use super::expr::{Expr, ExprRule};
use super::{Next, PErr, ParserRule, RResult};
use crate::lex::buffer::*;
use crate::lex::kind::TokenKind;
use crate::parse::combinator::opt::Opt;
use crate::parse::matc::*;
use crate::parse::stream::TokenStream;

#[derive(Debug, Clone)]
pub struct ArrDef {
    pub span: Span,
    pub exprs: Vec<Expr>,
}

#[derive(Debug, Default)]
pub struct ArrDefRule;

impl<'a, 's> ParserRule<'a, 's> for ArrDefRule {
    type Output = ArrDef;

    fn parse(stream: &mut TokenStream<'a, 's>) -> RResult<'s, Self::Output> {
        let open = Next::<OpenBracket>::parse(stream)?;
        let end = stream.find_matched_delim_offset::<Bracket>();
        let mut slice = stream.slice(end);
        stream.eat_n(end);

        if slice.is_empty() {
            match stream.peek_kind() {
                Some(TokenKind::CloseBracket) => {
                    let close = stream.expect();
                    return Ok(ArrDef {
                        span: Span::from_spans(slice.span(open), slice.span(close)),
                        exprs: Vec::new(),
                    });
                }
                _ => return Err(stream.fail("expected an array literal (e.g. [..])")),
            }
        }

        let mut exprs = Vec::new();
        while !slice.is_empty() {
            let (expr, comma) =
                <(ExprRule, Opt<Next<Comma>>)>::parse(&mut slice).map_err(PErr::fail)?;

            if comma.is_none() && slice.remaining() > 1 {
                return Err(slice.fail("expected `,` after field"));
            }

            exprs.push(expr);
        }
        let close = Next::<CloseBracket>::parse(stream).map_err(PErr::fail)?;

        Ok(ArrDef {
            span: Span::from_spans(slice.span(open), slice.span(close)),
            exprs,
        })
    }
}
