use super::{Next, ParserRule, RResult};
use crate::diagnostic::{Diag, Msg};
use crate::lex::buffer::*;
use crate::parse::stream::TokenStream;
use crate::parse::PARSE_ERR;
use crate::parse::{combinator::prelude::*, matc::*, rules::prelude::*};
use crate::parse_help;
use std::panic::Location;

/// Function that takes a set of parameters and optionally returns a value.
#[derive(Debug)]
pub struct Func {
    pub span: Span,
    pub name: TokenId,
    pub attributes: Vec<Attr>,
    pub ty: Option<TokenId>,
    pub params: Vec<Param>,
    pub block: Block,
}

impl Func {
    pub fn parse_attr<'a>(
        &mut self,
        stream: &TokenStream<'a>,
        attr: Attribute,
    ) -> Result<(), Diag<'a>> {
        match &*attr.str {
            "intrinsic" => {
                self.attributes.push(Attr::Intrinsic);
                Ok(())
            }
            _ => Err(stream.full_error(format!("invalid attribute `{}`", attr.str), attr.span, "")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Attr {
    Intrinsic,
}

#[derive(Debug)]
pub struct Param {
    pub name: TokenId,
    pub colon: TokenId,
    pub ty: TokenId,
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
        match Spanned::<(
            //Reported<Next<Fn>, MissingFn>,
            Next<Ident>,
            Next<Colon>,
            ParamsRule,
            //Next<OpenParen>,
            //Next<CloseParen>,
            XNor<(
                Reported<Next<Hyphen>, FnType>,
                Reported<Next<Greater>, FnType>,
                //Reported<Next<Colon>, DoubleColon>,
                Reported<Next<Ident>, MissingType>,
            )>,
        )>::parse(buffer, stream, stack)
        {
            Ok(res) => match BlockRules::parse(buffer, stream, stack) {
                Ok(block) => {
                    let span = res.span();
                    let (name, _, params, ty) = res.into_inner();
                    Ok(Func {
                        ty: ty.map(|(_, _, t)| t),
                        attributes: Vec::new(),
                        params,
                        span,
                        name,
                        block,
                    })
                }
                Err(e) => {
                    stream.eat_until_consume::<CloseCurly>();
                    Err(e)
                }
            },
            Err(e) => {
                stream.eat_until::<OpenCurly>();
                match BlockRules::parse(buffer, stream, stack) {
                    Ok(_) => Err(e),
                    Err(be) => Err(be.wrap(e)),
                }
            }
        }
    }
}

parse_help!(
    FnType,
    "expected `->`",
    "to specify a return type, use `-> <type>`"
);

//help!(
//    DoubleColon,
//    "expected double colon `::`",
//    "to specify a return type, use `:: <type>`"
//);

parse_help!(
    MissingType,
    "expected a type after `::`",
    "to specify a return type, use `:: <type>`"
);

//help!(MissingFn, "expected `fn` declaration", "");

struct ParamsRule;

impl<'a> ParserRule<'a> for ParamsRule {
    type Output = Vec<Param>;

    fn parse(
        buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        stack: &mut Vec<TokenId>,
    ) -> RResult<'a, Self::Output> {
        let str = *stream;
        match Next::<OpenParen>::parse(buffer, stream, stack) {
            Err(err) => {
                *stream = str;
                return Err(err);
            }
            Ok(_) => {}
        }
        if !stream.match_peek::<CloseParen>() {
            let index = stream.find_matched_delim_offset::<Paren>();
            let mut slice = stream.slice(index);
            let params = match While::<
                NextToken<Not<CloseParen>>,
                (Next<Ident>, Next<Colon>, Next<Ident>, Opt<Next<Comma>>),
            >::parse(buffer, &mut slice, stack)
            {
                Err(err) => {
                    *stream = str;
                    return Err(err);
                }
                Ok(args) => args,
            };

            stream.eat_until_consume::<CloseParen>();
            if params.len() > 1 {
                for (i, (_, _, ty, comma)) in params.iter().enumerate() {
                    if i < params.len() - 1 {
                        if comma.is_none() {
                            return Err(Diag::sourced(
                                PARSE_ERR,
                                buffer.source(),
                                Msg::error(
                                    buffer.span(*ty),
                                    "expected comma after parameter, before next parameter",
                                ),
                            )
                            .loc(Location::caller()));
                        }
                    }
                }
            }

            Ok(params
                .into_iter()
                .map(|(name, colon, ty, _)| Param { name, colon, ty })
                .collect())
        } else {
            match Next::<CloseParen>::parse(buffer, stream, stack) {
                Err(err) => {
                    *stream = str;
                    return Err(err);
                }
                Ok(_) => {}
            };

            Ok(Vec::new())
        }
    }
}
