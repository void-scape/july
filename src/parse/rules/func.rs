use super::types::{PType, TypeRule};
use super::{Next, ParserRule, RResult};
use crate::diagnostic::{Diag, Msg};
use crate::lex::buffer::*;
use crate::parse::stream::TokenStream;
use crate::parse::PARSE_ERR;
use crate::parse::{combinator::prelude::*, matc::*, rules::prelude::*};
use crate::parse_help;
use std::panic::Location;

#[derive(Debug)]
pub struct Func {
    pub span: Span,
    pub name: TokenId,
    pub attributes: Vec<Attr>,
    pub ty: Option<PType>,
    pub params: Vec<Param>,
    pub block: Block,
}

impl Func {
    pub fn parse_attr<'a>(
        &mut self,
        stream: &TokenStream<'a>,
        attr: &Attribute,
    ) -> Result<(), Diag<'a>> {
        assert!(attr.tokens.len() == 1, "add more attribute parsing");

        match stream.as_str(attr.tokens[0]) {
            "intrinsic" => {
                self.attributes.push(Attr::Intrinsic);
                Ok(())
            }
            str => Err(stream.full_error(format!("invalid attribute `{}`", str), attr.span, "")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Attr {
    Intrinsic,
}

#[derive(Debug)]
pub struct ExternFunc {
    pub span: Span,
    pub name: TokenId,
    pub ty: Option<PType>,
    pub params: Vec<Param>,
    pub convention: TokenId,
    pub link: Option<TokenId>,
}

impl ExternFunc {
    pub fn parse_attr<'a>(
        &mut self,
        stream: &TokenStream<'a>,
        attr: &Attribute,
    ) -> Result<(), Diag<'a>> {
        assert!(attr.tokens.len() == 4, "add more attribute parsing");

        if attr.tokens.first().map(|t| stream.as_str(*t)) == Some("link") {
            self.link = Some(*attr.tokens.get(2).unwrap());

            Ok(())
        } else {
            Err(stream.full_error("invalid attributes for external function", attr.span, ""))
        }
    }
}

#[derive(Debug)]
pub struct Param {
    pub name: TokenId,
    pub colon: TokenId,
    pub ty: PType,
}

/// fn <ident>() [-> <type>] <block>
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
            Next<Ident>,
            Next<Colon>,
            ParamsRule,
            XNor<(
                Reported<Next<Hyphen>, FnType>,
                Reported<Next<Greater>, FnType>,
                Reported<TypeRule, MissingType>,
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

parse_help!(
    MissingType,
    "expected a type after `->`",
    "to specify a return type, use `-> <type>`"
);

/// extern("<convention>") { fn <ident>() [-> <type>]; .. }
#[derive(Default)]
pub struct ExternFnRule;

impl<'a> ParserRule<'a> for ExternFnRule {
    type Output = Vec<ExternFunc>;

    fn parse(
        buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        stack: &mut Vec<TokenId>,
    ) -> RResult<'a, Self::Output> {
        match Spanned::<(
            Next<Extern>,
            Next<OpenParen>,
            Next<Str>,
            Next<CloseParen>,
            Next<OpenCurly>,
            While<
                NextToken<Not<CloseCurly>>,
                Spanned<(
                    Next<Ident>,
                    Next<Colon>,
                    ParamsRule,
                    XNor<(
                        Reported<Next<Hyphen>, FnType>,
                        Reported<Next<Greater>, FnType>,
                        Reported<TypeRule, MissingType>,
                    )>,
                    Next<Semi>,
                )>,
            >,
            Next<CloseCurly>,
        )>::parse(buffer, stream, stack)
        {
            Ok(res) => {
                let (_, _, convention, _, _, funcs, _) = res.into_inner();
                Ok(funcs
                    .into_iter()
                    .map(|spanned| {
                        let span = spanned.span();
                        let (name, _, params, ty, _) = spanned.into_inner();
                        ExternFunc {
                            ty: ty.map(|(_, _, t)| t),
                            params,
                            span,
                            name,
                            convention,
                            link: None,
                        }
                    })
                    .collect())
            }
            Err(e) => {
                stream.eat_until_consume::<Semi>();
                Err(e)
            }
        }
    }
}

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
                (
                    Next<Ident>,
                    Reported<Next<Colon>, MissingTyBinding>,
                    TypeRule,
                    Opt<Next<Comma>>,
                ),
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
                                    ty.span(buffer),
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

parse_help!(
    MissingTyBinding,
    "expected a type binding for parameter",
    "consider adding `: <type>`"
);
