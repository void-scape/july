use super::{Next, ParserRule, RResult};
use crate::diagnostic::Msg;
use crate::lex::buffer::*;
use crate::parse::combinator::opt::Opt;
use crate::parse::combinator::spanned::Spanned;
use crate::parse::matc::{self, *};
use crate::parse::stream::TokenStream;

/// Tagged union.
#[derive(Debug)]
pub struct Enum {
    pub span: Span,
    pub name: TokenId,
    pub variants: Vec<Variant>,
}

#[derive(Debug, Clone, Copy)]
pub struct Variant {
    pub span: Span,
    pub name: TokenId,
    // pub ty: TokenId,
}

/// <ident> : enum { [<variant>, ...] }
#[derive(Debug, Default)]
pub struct EnumRule;

impl<'a> ParserRule<'a> for EnumRule {
    type Output = Enum;

    fn parse(
        buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        stack: &mut Vec<TokenId>,
    ) -> RResult<'a, Self::Output> {
        let (name, _, _, (block_span, variants)) =
            <(Next<Ident>, Next<Colon>, Next<matc::Enum>, EnumBlockRule) as ParserRule>::parse(
                buffer, stream, stack,
            )?;
        Ok(Enum {
            name,
            variants,
            span: Span::from_spans(buffer.span(name), block_span),
        })
    }
}

/// `{ <ident>[(<type>)][,] }`
#[derive(Default)]
pub struct EnumBlockRule;

impl<'a> ParserRule<'a> for EnumBlockRule {
    type Output = (Span, Vec<Variant>);

    fn parse(
        buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        stack: &mut Vec<TokenId>,
    ) -> RResult<'a, Self::Output> {
        let start = buffer.span(stream.prev());
        match Spanned::<(Next<OpenCurly>, VariantDecl, Next<CloseCurly>)>::parse(
            buffer, stream, stack,
        ) {
            Ok(block) => {
                let span = block.span();
                let (_, fields, _) = block.into_inner();
                Ok((span, fields))
            }
            Err(e) => {
                stream.eat_until::<CloseCurly>();
                if let Some(t) = stream.next() {
                    Err(e.msg(Msg::note(
                        Span::from_spans(start, buffer.span(t)),
                        "while parsing this enum",
                    )))
                } else {
                    Err(e)
                }
            }
        }
    }
}

#[derive(Debug, Default)]
pub struct VariantDecl;

impl<'a> ParserRule<'a> for VariantDecl {
    type Output = Vec<Variant>;

    fn parse(
        buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        stack: &mut Vec<TokenId>,
    ) -> RResult<'a, Self::Output> {
        let mut fields = Vec::new();

        while !stream.match_peek::<CloseCurly>() {
            let field = Spanned::<(Next<Ident>, Opt<Next<Comma>>)>::parse(buffer, stream, stack)?;
            let span = field.span();
            let (name, comma) = field.into_inner();

            if comma.is_none() && !stream.match_peek::<CloseCurly>() {
                return Err(stream.error("expected `,` after field"));
            }

            fields.push(Variant {
                span,
                name,
                //colon,
                //ty,
            });
        }

        Ok(fields)
    }
}

/// Instantiation of an enum type.
#[derive(Debug, Clone, Copy)]
pub struct EnumDef {
    pub span: Span,
    pub name: TokenId,
    pub variant: Variant,
}

/// <ident> { [<field_def>, ...] }
#[derive(Debug, Default)]
pub struct EnumDefRule;

impl<'a> ParserRule<'a> for EnumDefRule {
    type Output = EnumDef;

    fn parse(
        buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        stack: &mut Vec<TokenId>,
    ) -> RResult<'a, Self::Output> {
        let (name, _, _, variant) =
            <(Next<Ident>, Next<Colon>, Next<Colon>, Next<Ident>)>::parse(buffer, stream, stack)?;
        Ok(EnumDef {
            name,
            variant: Variant {
                name: variant,
                span: buffer.span(variant),
            },
            span: Span::from_spans(buffer.span(name), buffer.span(variant)),
        })
    }
}
