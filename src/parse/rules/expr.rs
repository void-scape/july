use super::{ParserRule, RResult};
use crate::diagnostic::{Diag, Msg};
use crate::ir::prelude::Precedence;
use crate::lex::{buffer::*, kind::*};
use crate::parse::rules::RuleErr;
use crate::parse::{matc::*, stream::TokenStream, PARSE_ERR};

/// A composition of tokens that resolve into a value.
#[derive(Debug)]
pub enum Expr {
    /// return `name`;
    Ident(TokenId),
    /// return `10`;
    Lit(i64),
    /// let x = `3 * 2`;
    Bin(BinOp, Box<Expr>, Box<Expr>),
}

#[derive(Debug)]
pub enum BinOp {
    Add,
}

impl Precedence for TokenKind {
    fn precedence(&self) -> usize {
        match self {
            Self::Asterisk => 1,
            Self::Plus => 0,
            _ => 0,
        }
    }
}

/// `let x = 1 + 2;`
///          ^^^^^
pub struct ExprRule;

impl<'a> ParserRule<'a> for ExprRule {
    type Output = Expr;

    fn parse(
        buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        stack: &mut Vec<TokenId>,
    ) -> RResult<'a, Self::Output> {
        if stream.match_peek(Int) {
            let mut operators = Vec::new();
            while !stream.match_peek(Semi) {
                let token = stream.next().unwrap();
                let kind = buffer.kind(token);
                match kind {
                    TokenKind::Int => stack.push(token),
                    TokenKind::Plus | TokenKind::Asterisk => {
                        while operators
                            .last()
                            .is_some_and(|t| buffer.kind(*t).precedence() > kind.precedence())
                        {
                            stack.push(operators.pop().unwrap());
                        }
                        operators.push(token)
                    }
                    t => {
                        return Err(RuleErr::from_diag(Diag::sourced(
                            PARSE_ERR,
                            buffer.source(),
                            Msg::error(
                                buffer.span(token),
                                format!("cannot use {:?} in binary op", t),
                            ),
                        )));
                    }
                }
            }

            stack.extend(operators.drain(..).rev());
            stack.reverse();

            if stack.len() == 1 {
                return Ok(Expr::Lit(buffer.int_lit(stack.pop().unwrap())));
            } else {
                let mut op = None;
                let mut operands = Vec::new();
                while let Some(token) = stack.pop() {
                    match buffer.kind(token) {
                        TokenKind::Int => {
                            operands.push(token);
                        }
                        TokenKind::Plus => {
                            if let Some(old_op) = op {
                                op = Some(Expr::Bin(
                                    BinOp::Add,
                                    Box::new(old_op),
                                    Box::new(Expr::Lit(buffer.int_lit(operands.pop().unwrap()))),
                                ));
                            } else {
                                let o1 = operands.pop().unwrap();
                                let o2 = operands.pop().unwrap();
                                op = Some(Expr::Bin(
                                    BinOp::Add,
                                    Box::new(Expr::Lit(buffer.int_lit(o1))),
                                    Box::new(Expr::Lit(buffer.int_lit(o2))),
                                ))
                            }
                        }
                        _ => todo!(),
                    }
                }

                return Ok(op.unwrap());
            }
        } else if stream.match_peek(Ident) {
            return Ok(Expr::Ident(stream.next().unwrap()));
        }

        panic!()
    }
}
