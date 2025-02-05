use super::expr::{Expr, ExprRule};
use super::{Next, ParserRule, RResult};
use crate::lex::buffer::*;
use crate::parse::combinator::opt::Opt;
use crate::parse::combinator::spanned::Spanned;
use crate::parse::matc::{self, *};
use crate::parse::stream::TokenStream;

/// Structured data fields.
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
    pub colon: TokenId,
    pub ty: TokenId,
}

/// <ident> : struct { [<fields>, ...] }
#[derive(Debug, Default)]
pub struct StructRule;

impl<'a> ParserRule<'a> for StructRule {
    type Output = Struct;

    fn parse(
        buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        stack: &mut Vec<TokenId>,
    ) -> RResult<'a, Self::Output> {
        let (name, _, _, (block_span, fields)) =
            <(
                Next<Ident>,
                Next<Colon>,
                Next<matc::Struct>,
                StructBlockRule,
            ) as ParserRule>::parse(buffer, stream, stack)?;
        Ok(Struct {
            name,
            fields,
            span: Span::from_spans(buffer.span(name), block_span),
        })
    }
}

/// `{ [<ident>: <type>,][,] }`
#[derive(Default)]
pub struct StructBlockRule;

impl<'a> ParserRule<'a> for StructBlockRule {
    type Output = (Span, Vec<Field>);

    fn parse(
        buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        stack: &mut Vec<TokenId>,
    ) -> RResult<'a, Self::Output> {
        match Spanned::<(Next<OpenCurly>, StructFieldDecl, Next<CloseCurly>)>::parse(
            buffer, stream, stack,
        ) {
            Ok(block) => {
                let span = block.span();
                let (_, fields, _) = block.into_inner();
                Ok((span, fields))
            }
            Err(e) => {
                stream.eat_until_consume(CloseCurly);
                Err(e)
            }
        }
    }
}

/// `<ident>: <ty>[,]`
///               ^ ^ optional on last field
#[derive(Debug, Default)]
pub struct StructFieldDecl;

impl<'a> ParserRule<'a> for StructFieldDecl {
    type Output = Vec<Field>;

    fn parse(
        buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        stack: &mut Vec<TokenId>,
    ) -> RResult<'a, Self::Output> {
        let mut fields = Vec::new();

        while !stream.match_peek::<CloseCurly>() {
            let field =
                Spanned::<(Next<Ident>, Next<Colon>, Next<Ident>, Opt<Next<Comma>>)>::parse(
                    buffer, stream, stack,
                )?;
            let span = field.span();
            let (name, colon, ty, comma) = field.into_inner();

            if comma.is_none() && !stream.match_peek::<CloseCurly>() {
                return Err(stream.error("expected `,` after field"));
            }

            fields.push(Field {
                span,
                name,
                colon,
                ty,
            });
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
    pub colon: TokenId,
    pub expr: Expr,
}

/// <ident> { [<field_def>, ...] }
#[derive(Debug, Default)]
pub struct StructDefRule;

impl<'a> ParserRule<'a> for StructDefRule {
    type Output = StructDef;

    fn parse(
        buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        stack: &mut Vec<TokenId>,
    ) -> RResult<'a, Self::Output> {
        let (name, (block_span, fields)) =
            <(Next<Ident>, StructDefBlockRule)>::parse(buffer, stream, stack)?;
        Ok(StructDef {
            name,
            fields,
            span: Span::from_spans(buffer.span(name), block_span),
        })
    }
}

/// `{ [<ident>: <expr>,][,] }`
#[derive(Default)]
pub struct StructDefBlockRule;

impl<'a> ParserRule<'a> for StructDefBlockRule {
    type Output = (Span, Vec<FieldDef>);

    fn parse(
        buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        stack: &mut Vec<TokenId>,
    ) -> RResult<'a, Self::Output> {
        match Spanned::<(Next<OpenCurly>, StructFieldDef, Next<CloseCurly>)>::parse(
            buffer, stream, stack,
        ) {
            Ok(block) => {
                let span = block.span();
                let (_, fields, _) = block.into_inner();
                Ok((span, fields))
            }
            Err(e) => {
                stream.eat_until_consume(CloseCurly);
                Err(e)
            }
        }
    }
}

/// `<ident>: <expr>[,]`
///                  ^ ^ optional on last field
#[derive(Debug, Default)]
pub struct StructFieldDef;

impl<'a> ParserRule<'a> for StructFieldDef {
    type Output = Vec<FieldDef>;

    fn parse(
        buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        stack: &mut Vec<TokenId>,
    ) -> RResult<'a, Self::Output> {
        let mut fields = Vec::new();

        while !stream.match_peek::<CloseCurly>() {
            let field = Spanned::<(Next<Ident>, Next<Colon>, ExprRule, Opt<Next<Comma>>)>::parse(
                buffer, stream, stack,
            )?;
            let span = field.span();
            let (name, colon, expr, comma) = field.into_inner();

            if comma.is_none() && !stream.match_peek::<CloseCurly>() {
                return Err(stream.error("expected `,` after field"));
            }

            fields.push(FieldDef {
                span,
                name,
                colon,
                expr,
            });
        }

        Ok(fields)
    }
}
