use self::matc::{Any, Colon, Ident, MatchTokenKind};
use self::rules::prelude::{Enum, Func, Struct};
use self::rules::ParserRule;
use crate::diagnostic::Diag;
use crate::lex::buffer::*;
use crate::lex::kind::TokenKind;

mod combinator;
mod matc;
pub mod rules;
mod stream;

pub const PARSE_ERR: &'static str = "failed to parse";

#[derive(Debug)]
pub enum Item {
    Enum(Enum),
    Struct(Struct),
    Func(Func),
}

pub struct Parser;

impl Parser {
    pub fn parse<'a>(buffer: &'a TokenBuffer<'a>) -> Result<Vec<Item>, Vec<Diag<'a>>> {
        Self::parse_buffer(buffer)
    }

    fn parse_buffer<'a>(buffer: &'a TokenBuffer<'a>) -> Result<Vec<Item>, Vec<Diag<'a>>> {
        let mut items = Vec::new();
        let mut stack = Vec::with_capacity(buffer.len());
        let mut diags = Vec::new();
        let mut stream = buffer.stream();
        let mut active_attribute = None;

        while !stream.is_empty() {
            match (
                stream.peek().map(|t| buffer.kind(t)),
                stream.peekn(1).map(|t| buffer.kind(t)),
                stream.peekn(2).map(|t| buffer.kind(t)),
            ) {
                (Some(TokenKind::Pound), Some(TokenKind::OpenBracket), _) => {
                    match rules::prelude::AttributeRule::parse(buffer, &mut stream, &mut stack) {
                        Err(diag) => {
                            diags.push(diag);
                        }
                        Ok(a) => {
                            active_attribute = Some(a);
                        }
                    }
                }
                (Some(TokenKind::Ident), Some(TokenKind::Colon), Some(TokenKind::Enum)) => {
                    if active_attribute.is_some() {
                        diags.push(stream.full_error(
                            "cannot add attributes to enums",
                            buffer.span(stream.peek().unwrap()),
                            "",
                        ))
                    }

                    match rules::prelude::EnumRule::parse(buffer, &mut stream, &mut stack) {
                        Err(diag) => {
                            diags.push(diag);
                        }
                        Ok(e) => {
                            items.push(Item::Enum(e));
                        }
                    }
                }
                (Some(TokenKind::Ident), Some(TokenKind::Colon), Some(TokenKind::Struct)) => {
                    if active_attribute.is_some() {
                        diags.push(stream.full_error(
                            "cannot add attributes to enums",
                            buffer.span(stream.peek().unwrap()),
                            "",
                        ))
                    }

                    match rules::prelude::StructRule::parse(buffer, &mut stream, &mut stack) {
                        Err(diag) => {
                            diags.push(diag);
                        }
                        Ok(s) => {
                            items.push(Item::Struct(s));
                        }
                    }
                }
                (Some(TokenKind::Ident), Some(TokenKind::Colon), Some(TokenKind::OpenParen)) => {
                    match rules::prelude::FnRule::parse(buffer, &mut stream, &mut stack) {
                        Err(diag) => {
                            diags.push(diag);
                        }
                        Ok(mut f) => {
                            if let Some(attr) = active_attribute.take() {
                                if let Err(diag) = f.parse_attr(&stream, attr) {
                                    diags.push(diag);
                                }
                            }
                            items.push(Item::Func(f));
                        }
                    }
                }
                (_t1, _t2, _t3) => {
                    // panic!("{_t1:?}, {_t2:?}, {_t3:?}");
                    diags.push(stream.error("expected a `struct`, `enum`, or `function`"));
                    while !Ident::matches(stream.peek().map(|t| buffer.kind(t)))
                        || !Colon::matches(stream.peekn(1).map(|t| buffer.kind(t)))
                        || !Any::<(matc::Enum, matc::Struct, matc::OpenParen)>::matches(
                            stream.peekn(2).map(|t| buffer.kind(t)),
                        )
                    {
                        stream.eat();
                    }
                }
            }
        }

        if !diags.is_empty() {
            Err(diags)
        } else {
            Ok(items)
        }
    }
}
