use super::types::{PType, TypeRule};
use super::{Next, ParserRule, RResult};
use crate::diagnostic::Diag;
use crate::lex::buffer::*;
use crate::lex::kind::TokenKind;
use crate::rules::PErr;
use crate::stream::TokenStream;
use crate::{combinator::prelude::*, matc::*, rules::prelude::*};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Func {
    pub span: Span,
    pub attributes: Vec<Attr>,
    pub name: TokenId,
    pub colon: TokenId,
    pub params: Vec<Param>,
    pub ty: Option<PType>,
    pub block: Block,
}

impl Func {
    pub fn parse_attr<'a, 's>(
        &mut self,
        stream: &TokenStream<'a, 's>,
        attr: &Attribute,
    ) -> Result<(), Diag<'s>> {
        assert!(
            attr.tokens.len() == 1,
            "add more attribute parsing: {:#?}",
            attr.tokens
                .iter()
                .map(|t| stream.as_str(*t))
                .collect::<Vec<_>>()
        );

        match stream.as_str(attr.tokens[0]) {
            "intrinsic" => {
                self.attributes.push(Attr::Intrinsic);
                Ok(())
            }
            str => Err(stream.full_error(format!("invalid attribute `{}`", str), attr.span)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Attr {
    Intrinsic,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternBlock {
    pub span: Span,
    pub exturn: TokenId,
    pub convention: TokenId,
    pub funcs: Vec<ExternFunc>,
}

impl ExternBlock {
    pub fn parse_attr<'a, 's>(
        &mut self,
        stream: &TokenStream<'a, 's>,
        attr: &Attribute,
    ) -> Result<(), Diag<'s>> {
        assert!(attr.tokens.len() == 4, "add more attribute parsing");

        if attr.tokens.first().map(|t| stream.as_str(*t)) == Some("link") {
            for func in self.funcs.iter_mut() {
                func.link = Some(*attr.tokens.get(2).unwrap());
            }

            Ok(())
        } else {
            Err(stream.full_error("invalid attributes for external function", attr.span))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternFunc {
    pub span: Span,
    pub name: TokenId,
    pub ty: Option<PType>,
    pub params: Vec<Param>,
    pub convention: TokenId,
    pub link: Option<TokenId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Param {
    Slf(TokenId),
    SlfRef(TokenId),
    Named {
        name: TokenId,
        colon: TokenId,
        ty: PType,
    },
}

#[derive(Default)]
pub struct FnRule;

impl<'a, 's> ParserRule<'a, 's> for FnRule {
    type Output = Func;

    fn parse(stream: &mut TokenStream<'a, 's>) -> RResult<'s, Self::Output> {
        match Spanned::<(
            Next<Ident>,
            Next<Colon>,
            ParamsRule,
            XNor<(Next<Hyphen>, Next<CloseAngle>, TypeRule)>,
        )>::parse(stream)
        {
            Ok(res) => {
                let chk = *stream;
                match BlockRules::parse(stream) {
                    Ok(block) => {
                        let span = res.span();
                        let (name, colon, params, ty) = res.into_inner();
                        Ok(Func {
                            span,
                            attributes: Vec::new(),
                            name,
                            colon,
                            params,
                            ty: ty.map(|(_, _, ty)| ty),
                            block,
                        })
                    }
                    Err(e) => {
                        *stream = chk;
                        Next::<OpenCurly>::parse(stream).map_err(PErr::fail)?;
                        let offset = stream.find_matched_delim_offset::<Curly>();
                        stream.eat_n(offset);
                        Next::<CloseCurly>::parse(stream).map_err(PErr::fail)?;
                        Err(e.fail())
                    }
                }
            }
            Err(e) => {
                stream.eat_until::<OpenCurly>();
                match BlockRules::parse(stream) {
                    Ok(_) => Err(e),
                    Err(be) => Err(be.recover().wrap(e)),
                }
            }
        }
    }
}

//parse_help!(
//    FnType,
//    "expected `->`",
//    "to specify a return type, use `-> <type>`"
//);
//
//parse_help!(
//    MissingType,
//    "expected a type after `->`",
//    "to specify a return type, use `-> <type>`"
//);

/// extern("<convention>") { fn <ident>() [-> <type>]; .. }
#[derive(Default)]
pub struct ExternFnRule;

impl<'a, 's> ParserRule<'a, 's> for ExternFnRule {
    type Output = ExternBlock;

    #[track_caller]
    fn parse(stream: &mut TokenStream<'a, 's>) -> RResult<'s, Self::Output> {
        if !stream.match_peek::<Extern>() {
            return Err(PErr::Recover(stream.error("expected `extern`")));
        }
        let exturn = stream.expect();

        let res = Spanned::<(
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
                    XNor<(Next<Hyphen>, Next<CloseAngle>, TypeRule)>,
                    Next<Semi>,
                )>,
            >,
            Next<CloseCurly>,
        )>::parse(stream)
        .map_err(PErr::fail)?;

        let (_, convention, _, _, funcs, end) = res.into_inner();
        Ok(ExternBlock {
            span: Span::from_spans(stream.span(exturn), stream.span(end)),
            exturn,
            convention,
            funcs: funcs
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
                .collect(),
        })
    }
}

#[derive(Debug, Default)]
pub struct ParamsRule;

impl<'a, 's> ParserRule<'a, 's> for ParamsRule {
    type Output = Vec<Param>;

    fn parse(stream: &mut TokenStream<'a, 's>) -> RResult<'s, Self::Output> {
        let str = *stream;
        let _open = match Next::<OpenParen>::parse(stream) {
            Err(err) => {
                *stream = str;
                return Err(err.fail());
            }
            Ok(_) => {}
        };

        if !stream.match_peek::<CloseParen>() {
            let index = stream.find_matched_delim_offset::<Paren>();
            let mut slice = stream.slice(index);
            stream.eat_n(index);
            let _close = stream.expect();

            let mut params = Vec::new();
            loop {
                if slice.is_empty() {
                    break;
                }

                match slice.peek_kind() {
                    Some(TokenKind::Ampersand) => {
                        let _ref = slice.expect();
                        if slice.match_peek::<Slf>() {
                            let _self = slice.expect();
                            let comma = Opt::<Next<Comma>>::parse(&mut slice)?;
                            params.push((comma, Param::SlfRef(_self)));
                        } else {
                            return Err(slice.fail("expected `self` or `{identifier}`"));
                        }
                    }
                    Some(TokenKind::Slf) => {
                        let _self = slice.expect();
                        let comma = Opt::<Next<Comma>>::parse(&mut slice)?;
                        params.push((comma, Param::Slf(_self)));
                    }
                    _ => {
                        let (name, colon, ty, comma) =
                            <(Next<Ident>, Next<Colon>, TypeRule, Opt<Next<Comma>>)>::parse(
                                &mut slice,
                            )?;
                        params.push((comma, Param::Named { name, colon, ty }));
                    }
                }
            }

            if params.len() > 1 {
                for (i, (comma, _param)) in params.iter().enumerate() {
                    if i < params.len() - 1 {
                        if comma.is_none() {
                            return Err(PErr::Fail(stream.error(format!(
                                "expected comma after parameter, before next parameter"
                            ))));
                        }
                    }
                }
            }

            Ok(params.into_iter().map(|(_, param)| param).collect())
        } else {
            let _close = stream.expect();
            Ok(Vec::new())
        }
    }
}

pub struct ArgsRule;

impl<'a, 's> ParserRule<'a, 's> for ArgsRule {
    type Output = (Span, Vec<Expr>);

    fn parse(stream: &mut TokenStream<'a, 's>) -> RResult<'s, Self::Output> {
        let str = *stream;
        let open = match Next::<OpenParen>::parse(stream) {
            Err(err) => {
                *stream = str;
                return Err(err.fail());
            }
            Ok(open) => open,
        };

        if !stream.match_peek::<CloseParen>() {
            let index = stream.find_matched_delim_offset::<Paren>();
            let mut slice = stream.slice(index);
            let args =
                match While::<NextToken<Any>, (ExprRule, Opt<Next<Comma>>)>::parse(&mut slice) {
                    Err(err) => {
                        *stream = str;
                        return Err(err);
                    }
                    Ok(args) => args,
                };

            stream.eat_n(index);
            let close = if let Some(t) = stream.next() {
                t
            } else {
                stream.prev()
            };

            if args.len() > 1 {
                for (i, (_, comma)) in args.iter().enumerate() {
                    if i < args.len() - 1 {
                        if comma.is_none() {
                            return Err(PErr::Fail(stream.error(format!(
                                "expected comma after parameter, before next parameter"
                            ))));
                        }
                    }
                }
            }

            Ok((
                Span::from_spans(stream.span(open), stream.span(close)),
                args.into_iter().map(|(expr, _)| expr).collect(),
            ))
        } else {
            let close = stream.expect();
            Ok((
                Span::from_spans(stream.span(open), stream.span(close)),
                Vec::new(),
            ))
        }
    }
}

//parse_help!(
//    MissingTyBinding,
//    "expected a type binding for parameter",
//    "consider adding `: <type>`"
//);
