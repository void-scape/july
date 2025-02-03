use self::rules::prelude::{Func, Struct};
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
            if buffer.kind(stream.peek().unwrap()) == TokenKind::Ident
                && buffer.kind(stream.peekn(1).unwrap()) == TokenKind::Colon
                && buffer.kind(stream.peekn(2).unwrap()) == TokenKind::Struct
            {
                match rules::prelude::StructRule::parse(buffer, &mut stream, &mut stack) {
                    Err(diag) => {
                        diags.extend(diag.into_diags());
                    }
                    Ok(s) => {
                        items.push(Item::Struct(s));
                    }
                }
            } else {
                match rules::prelude::FnRule::parse(buffer, &mut stream, &mut stack) {
                    Err(diag) => {
                        diags.extend(diag.into_diags());
                    }
                    Ok(f) => {
                        items.push(Item::Func(f));
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
