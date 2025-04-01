use self::diagnostic::Diag;
use self::rules::prelude::{Attribute, Const, Enum, ExternBlock, Func, Impl, Param, Struct, Use};
use self::rules::{PErr, ParserRule};
use crate::lex::buffer::*;
use crate::lex::kind::TokenKind;

pub extern crate annotate_snippets;

mod combinator;
pub mod diagnostic;
pub mod lex;
pub mod matc;
pub mod rules;
mod stream;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssignKind {
    Equals,
    Add,
    Sub,
}

impl AssignKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Equals => "=",
            Self::Add => "+=",
            Self::Sub => "-=",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOpKind {
    //
    Mul,
    Div,
    //
    Add,
    Sub,

    //
    Shl,
    Shr,
    //
    Band,
    //
    Xor,
    //
    Bor,

    //
    Eq,
    Ne,

    //
    Gt,
    Lt,
    Ge,
    Le,

    //
    And,
    //
    Or,
}

impl BinOpKind {
    pub fn output_is_input(&self) -> bool {
        match self {
            Self::Mul | Self::Div | Self::Add | Self::Sub => true,
            Self::Shl | Self::Shr | Self::Band | Self::Xor | Self::Bor => true,
            Self::Eq | Self::Ne => false,
            Self::Gt | Self::Lt | Self::Ge | Self::Le => false,
            Self::And | Self::Or => false,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Mul => "*",
            Self::Div => "/",
            Self::Add => "+",
            Self::Sub => "-",

            Self::Shl => "<<",
            Self::Shr => ">>",
            Self::Band => "&",
            Self::Xor => "^",
            Self::Bor => "|",

            Self::Eq => "==",
            Self::Ne => "!=",

            Self::Gt => ">",
            Self::Lt => "<",
            Self::Ge => ">=",
            Self::Le => "<=",

            Self::And => "&&",
            Self::Or => "||",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UOpKind {
    Deref,
    Ref,
    Not,
    Neg,
}

impl UOpKind {
    pub fn is_prefix(&self) -> bool {
        match self {
            Self::Deref => false,
            _ => true,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Deref => "*",
            Self::Ref => "&",
            Self::Not => "!",
            Self::Neg => "-",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Item {
    pub kind: ItemKind,
    pub source: usize,
}

#[allow(unused)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ItemKind {
    Func(Func),
    Struct(Struct),
    Enum(Enum),
    Const(Const),
    Use(Use),
    Impl(Impl),
    Attr(Attribute),
    Extern(ExternBlock),
}

pub fn parse<'a>(buffer: &'a TokenBuffer) -> Result<Vec<Item>, Diag> {
    let mut items = Vec::new();
    let mut diags = Vec::new();
    let mut stream = buffer.stream();

    while !stream.is_empty() {
        match (
            stream.peek().map(|t| buffer.kind(t)),
            stream.peekn(1).map(|t| buffer.kind(t)),
            stream.peekn(2).map(|t| buffer.kind(t)),
        ) {
            (Some(TokenKind::Ident), Some(TokenKind::Colon), Some(TokenKind::OpenParen)) => {
                match rules::prelude::FnRule::parse(&mut stream) {
                    Err(diag) => {
                        diags.push(diag);
                    }
                    Ok(f) => {
                        items.push(ItemKind::Func(f));
                    }
                }
            }

            (Some(TokenKind::Ident), Some(TokenKind::Colon), Some(TokenKind::Struct)) => {
                match rules::prelude::StructRule::parse(&mut stream) {
                    Err(diag) => {
                        diags.push(diag);
                    }
                    Ok(s) => {
                        items.push(ItemKind::Struct(s));
                    }
                }
            }

            (Some(TokenKind::Ident), Some(TokenKind::Colon), Some(TokenKind::Const)) => {
                match rules::prelude::ConstRule::parse(&mut stream) {
                    Err(diag) => {
                        diags.push(diag);
                    }
                    Ok(c) => {
                        items.push(ItemKind::Const(c));
                    }
                }
            }

            (Some(TokenKind::Impl), _, _) => match rules::prelude::ImplRule::parse(&mut stream) {
                Err(diag) => {
                    diags.push(diag);
                }
                Ok(i) => {
                    items.push(ItemKind::Impl(i));
                }
            },

            (Some(TokenKind::Use), _, _) => match rules::prelude::UseRule::parse(&mut stream) {
                Err(diag) => {
                    diags.push(diag);
                }
                Ok(u) => {
                    items.push(ItemKind::Use(u));
                }
            },

            (Some(TokenKind::Pound), Some(TokenKind::OpenBracket), _) => {
                match rules::prelude::AttributeRule::parse(&mut stream) {
                    Err(diag) => {
                        diags.push(diag);
                    }
                    Ok(a) => {
                        items.push(ItemKind::Attr(a));
                    }
                }
            }

            (Some(TokenKind::Extern), _, _) => {
                match rules::prelude::ExternFnRule::parse(&mut stream) {
                    Err(e) => {
                        return Err(e.into_diag());
                    }
                    Ok(e) => {
                        for func in e.funcs.iter() {
                            if let Some(_self) =
                                func.params.iter().find(|p| matches!(p, Param::Slf(_)))
                            {
                                diags.push(PErr::Fail(stream.report_error(
                                    "free function cannot contain `self`",
                                    match _self {
                                        Param::Slf(t) => stream.span(*t),
                                        _ => unreachable!(),
                                    },
                                )));
                            }
                        }

                        items.push(ItemKind::Extern(e));
                    }
                }
            }

            (_t1, _t2, _t3) => {
                return Err(stream.error("expected an item"));
            }
        }
    }

    if !diags.is_empty() {
        Err(Diag::bundle(diags.into_iter().map(PErr::into_diag)))
    } else {
        Ok(items
            .into_iter()
            .map(|kind| Item {
                kind,
                source: buffer.source_id(),
            })
            .collect())
    }
}
