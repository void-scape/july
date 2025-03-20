use self::matc::{Any, Colon, Ident, MatchTokenKind};
use self::rules::prelude::{Const, Enum, ExternFunc, Func, Impl, Param, Struct};
use self::rules::{PErr, ParserRule};
use crate::diagnostic::{self, Diag};
use crate::lex::buffer::*;
use crate::lex::kind::TokenKind;

mod combinator;
mod matc;
pub mod rules;
mod stream;

#[derive(Debug)]
pub enum Item {
    #[allow(unused)]
    Enum(Enum),
    Struct(Struct),
    Impl(Impl),
    Func(Func),
    Const(Const),
    Extern(ExternFunc),
}

pub struct Parser;

impl Parser {
    pub fn parse<'a>(buffer: &'a TokenBuffer<'a>) -> Result<Vec<Item>, ()> {
        Self::parse_buffer(buffer)
    }

    fn parse_buffer<'a>(buffer: &'a TokenBuffer<'a>) -> Result<Vec<Item>, ()> {
        let mut items = Vec::new();
        let mut diags = Vec::new();
        let mut stream = buffer.stream();
        let mut active_attribute = None;

        while !stream.is_empty() {
            match (
                stream.peek().map(|t| buffer.kind(t)),
                stream.peekn(1).map(|t| buffer.kind(t)),
                stream.peekn(2).map(|t| buffer.kind(t)),
            ) {
                (Some(TokenKind::Extern), _, _) => {
                    match rules::prelude::ExternFnRule::parse(&mut stream) {
                        Err(e) => {
                            diagnostic::report(e.into_diag());
                            return Err(());
                        }
                        Ok(mut f) => {
                            if let Some(attr) = active_attribute.take() {
                                for f in f.iter_mut() {
                                    if let Err(diag) = f.parse_attr(&stream, &attr) {
                                        diags.push(diag);
                                    }
                                }
                            }

                            for func in f.iter() {
                                if let Some(_self) = func.params.iter().find(|p| {
                                    matches!(p, Param::Slf(_)) || matches!(p, Param::SlfRef(_))
                                }) {
                                    diags.push(PErr::Fail(stream.full_error(
                                        "extern function cannot contain `self`",
                                        match _self {
                                            Param::Slf(t) => stream.span(*t),
                                            Param::SlfRef(t) => stream.span(*t),
                                            _ => unreachable!(),
                                        },
                                    )));
                                }
                            }

                            items.extend(f.into_iter().map(|f| Item::Extern(f)));
                        }
                    }
                }
                (Some(TokenKind::Pound), Some(TokenKind::OpenBracket), _) => {
                    match rules::prelude::AttributeRule::parse(&mut stream) {
                        Err(diag) => {
                            diags.push(diag);
                        }
                        Ok(a) => {
                            active_attribute = Some(a);
                        }
                    }
                }
                (Some(TokenKind::Impl), _, _) => {
                    match rules::prelude::ImplRule::parse(&mut stream) {
                        Err(diag) => {
                            diags.push(diag);
                        }
                        Ok(i) => {
                            items.push(Item::Impl(i));
                        }
                    }
                }
                //(Some(TokenKind::Ident), Some(TokenKind::Colon), Some(TokenKind::Enum)) => {
                //    if active_attribute.is_some() {
                //        diags.push(PErr::Recover(stream.full_error(
                //            "cannot add attributes to enums",
                //            buffer.span(stream.peek().unwrap()),
                //        )))
                //    }
                //
                //    match rules::prelude::EnumRule::parse(&mut stream) {
                //        Err(diag) => {
                //            diags.push(diag);
                //        }
                //        Ok(e) => {
                //            items.push(Item::Enum(e));
                //        }
                //    }
                //}
                (Some(TokenKind::Ident), Some(TokenKind::Colon), Some(TokenKind::Struct)) => {
                    if active_attribute.is_some() {
                        diags.push(PErr::Recover(stream.full_error(
                            "cannot add attributes to structs",
                            buffer.span(stream.peek().unwrap()),
                        )))
                    }

                    match rules::prelude::StructRule::parse(&mut stream) {
                        Err(diag) => {
                            diags.push(diag);
                        }
                        Ok(s) => {
                            items.push(Item::Struct(s));
                        }
                    }
                }
                (Some(TokenKind::Ident), Some(TokenKind::Colon), Some(TokenKind::OpenParen)) => {
                    match rules::prelude::FnRule::parse(&mut stream) {
                        Err(diag) => {
                            diags.push(diag);
                        }
                        Ok(mut f) => {
                            if let Some(attr) = active_attribute.take() {
                                if let Err(diag) = f.parse_attr(&stream, &attr) {
                                    diags.push(diag);
                                }
                            }
                            items.push(Item::Func(f));
                        }
                    }
                }
                (Some(TokenKind::Ident), Some(TokenKind::Colon), Some(TokenKind::Const)) => {
                    match rules::prelude::ConstRule::parse(&mut stream) {
                        Err(diag) => {
                            diags.push(diag);
                        }
                        Ok(c) => {
                            items.push(Item::Const(c));
                        }
                    }
                }
                (_t1, _t2, _t3) => {
                    // panic!("{_t1:?}, {_t2:?}, {_t3:?}");
                    diags.push(PErr::Recover(stream.error("expected an item")));
                    while !stream.is_empty()
                        && (!Ident::matches(stream.peek().map(|t| buffer.kind(t)))
                            || !Colon::matches(stream.peekn(1).map(|t| buffer.kind(t)))
                            || !Any::<(matc::Enum, matc::Struct, matc::OpenParen)>::matches(
                                stream.peekn(2).map(|t| buffer.kind(t)),
                            ))
                    {
                        stream.eat();
                    }
                }
            }
        }

        if !diags.is_empty() {
            diagnostic::report(Diag::bundle(diags.into_iter().map(PErr::into_diag)));
            Err(())
        } else {
            Ok(items)
        }
    }
}
