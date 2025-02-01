use super::{Next, ParserRule, RResult, RuleErr};
use crate::diagnostic::Diag;
use crate::help;
use crate::lex::buffer::*;
use crate::parse::stream::TokenStream;
use crate::parse::{combinator::prelude::*, matc::*, rules::prelude::*};
use annotate_snippets::Level;

/// Function that takes a set of parameters and optionally returns a value.
#[derive(Debug)]
pub struct Func {
    name: TokenId,
    //params: Vec<LetInstr>,
    ty: Option<TokenId>,
    block: Block,
}

/// fn <ident>() [:: <type>] <block>
#[derive(Default)]
pub struct FnRule;

impl<'a> ParserRule<'a> for FnRule {
    type Output = Func;

    fn parse(
        buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        stack: &mut Vec<TokenId>,
    ) -> RResult<'a, Self::Output> {
        match <(
            //Reported<Next<Fn>, MissingFn>,
            Next<Ident>,
            Next<Colon>,
            Next<OpenParen>,
            Next<CloseParen>,
            XNor<(
                Reported<Next<Hyphen>, FnType>,
                Reported<Next<Greater>, FnType>,
                //Reported<Next<Colon>, DoubleColon>,
                Reported<Next<Ident>, MissingType>,
            )>,
        ) as ParserRule>::parse(buffer, stream, stack)
        {
            Ok((_, name, _, _, ty)) => match BlockRules::parse(buffer, stream, stack) {
                Ok(block) => Ok(Func {
                    ty: ty.map(|(_, _, t)| t),
                    name,
                    block,
                }),
                Err(e) => {
                    stream.eat_until_consume(CloseCurly);
                    Err(e)
                }
            },
            Err(e) => {
                stream.eat_until(OpenCurly);
                match BlockRules::parse(buffer, stream, stack) {
                    Ok(_) => Err(e),
                    Err(be) => Err(RuleErr::from_diags(
                        e.into_diags()
                            .into_iter()
                            .chain(be.into_diags().into_iter()),
                    )),
                }
            }
        }
    }
}

help!(
    FnType,
    "expected `->`",
    "to specify a return type, use `-> <type>`"
);

//help!(
//    DoubleColon,
//    "expected double colon `::`",
//    "to specify a return type, use `:: <type>`"
//);

help!(
    MissingType,
    "expected a type after `::`",
    "to specify a return type, use `:: <type>`"
);

//help!(MissingFn, "expected `fn` declaration", "");
