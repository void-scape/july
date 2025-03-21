use super::expr::{Expr, ExprRule};
use super::types::{PType, TypeRule};
use super::{Next, PErr, ParserRule, RResult};
use crate::lex::buffer::*;
use crate::combinator::opt::Opt;
use crate::combinator::spanned::Spanned;
use crate::matc::{self, *};
use crate::stream::TokenStream;

#[derive(Debug)]
pub struct Struct {
    pub span: Span,
    pub name: TokenId,
    pub fields: Vec<Field>,
}

#[derive(Debug)]
pub struct Field {
    pub span: Span,
    pub name: TokenId,
    pub ty: PType,
}

#[derive(Debug, Default)]
pub struct StructRule;

impl<'a, 's> ParserRule<'a, 's> for StructRule {
    type Output = Struct;

    fn parse(stream: &mut TokenStream<'a, 's>) -> RResult<'s, Self::Output> {
        let (name, _, _) =
            <(Next<Ident>, Next<Colon>, Next<matc::Struct>) as ParserRule>::parse(stream)?;
        let (block_span, fields) = StructBlockRule::parse(stream).map_err(PErr::fail)?;
        Ok(Struct {
            name,
            fields,
            span: Span::from_spans(stream.span(name), block_span),
        })
    }
}

#[derive(Default)]
pub struct StructBlockRule;

impl<'a, 's> ParserRule<'a, 's> for StructBlockRule {
    type Output = (Span, Vec<Field>);

    fn parse(stream: &mut TokenStream<'a, 's>) -> RResult<'s, Self::Output> {
        let chk = *stream;
        match Spanned::<(Next<OpenCurly>, StructFieldDecl, Next<CloseCurly>)>::parse(stream) {
            Ok(block) => {
                let span = block.span();
                let (_, fields, _) = block.into_inner();
                Ok((span, fields))
            }
            Err(e) => {
                *stream = chk;
                stream.consume_matched_delimiters_inclusive::<Curly>();
                Err(e.fail())
            }
        }
    }
}

/// `<ident>: <ty>[,]`
///               ^ ^ optional on last field
#[derive(Debug, Default)]
pub struct StructFieldDecl;

impl<'a, 's> ParserRule<'a, 's> for StructFieldDecl {
    type Output = Vec<Field>;

    fn parse(stream: &mut TokenStream<'a, 's>) -> RResult<'s, Self::Output> {
        let mut fields = Vec::new();

        while !stream.match_peek::<CloseCurly>() {
            let field =
                Spanned::<(Next<Ident>, Next<Colon>, TypeRule, Opt<Next<Comma>>)>::parse(stream)?;
            let span = field.span();
            let (name, _colon, ty, comma) = field.into_inner();

            if comma.is_none() && !stream.match_peek::<CloseCurly>() {
                return Err(PErr::Fail(stream.error("expected `,` after field")));
            }

            fields.push(Field { span, name, ty });
        }

        Ok(fields)
    }
}

/// Instantiation of a struct type.
#[derive(Debug, Clone)]
pub struct StructDef {
    pub span: Span,
    pub name: TokenId,
    pub fields: Vec<FieldDef>,
}

#[derive(Debug, Clone)]
pub struct FieldDef {
    pub span: Span,
    pub name: TokenId,
    pub expr: Expr,
}

#[derive(Debug, Default)]
pub struct StructDefRule;

impl<'a, 's> ParserRule<'a, 's> for StructDefRule {
    type Output = StructDef;

    fn parse(stream: &mut TokenStream<'a, 's>) -> RResult<'s, Self::Output> {
        let (name, (block_span, fields)) = <(Next<Ident>, StructDefBlockRule)>::parse(stream)?;
        Ok(StructDef {
            name,
            fields,
            span: Span::from_spans(stream.span(name), block_span),
        })
    }
}

/// `{ [<ident>: <expr>,][,] }`
#[derive(Default)]
pub struct StructDefBlockRule;

impl<'a, 's> ParserRule<'a, 's> for StructDefBlockRule {
    type Output = (Span, Vec<FieldDef>);

    fn parse(stream: &mut TokenStream<'a, 's>) -> RResult<'s, Self::Output> {
        match Spanned::<(Next<OpenCurly>, StructFieldDef, Next<CloseCurly>)>::parse(stream) {
            Ok(block) => {
                let span = block.span();
                let (_, fields, _) = block.into_inner();
                Ok((span, fields))
            }
            Err(e) => Err(e),
        }
    }
}

/// `<ident>: <expr>[,]`
///                  ^ ^ optional on last field
#[derive(Debug, Default)]
pub struct StructFieldDef;

impl<'a, 's> ParserRule<'a, 's> for StructFieldDef {
    type Output = Vec<FieldDef>;

    fn parse(stream: &mut TokenStream<'a, 's>) -> RResult<'s, Self::Output> {
        let mut fields = Vec::new();

        while !stream.match_peek::<CloseCurly>() {
            let field =
                Spanned::<(Next<Ident>, Next<Colon>, ExprRule, Opt<Next<Comma>>)>::parse(stream)?;
            let span = field.span();
            let (name, _colon, expr, comma) = field.into_inner();

            if comma.is_none() && !stream.match_peek::<CloseCurly>() {
                return Err(PErr::Fail(stream.error("expected `,` after field")));
            }

            fields.push(FieldDef { span, name, expr });
        }

        Ok(fields)
    }
}
