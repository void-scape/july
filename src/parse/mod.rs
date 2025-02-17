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

        while !stream.is_empty() {
            match (
                stream.peek().map(|t| buffer.kind(t)),
                stream.peekn(1).map(|t| buffer.kind(t)),
                stream.peekn(2).map(|t| buffer.kind(t)),
            ) {
                (Some(TokenKind::Ident), Some(TokenKind::Colon), Some(TokenKind::Enum)) => {
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
                        Ok(f) => {
                            items.push(Item::Func(f));
                        }
                    }
                }
                _ => {
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
