use cranelift_codegen::timing::wasm_translate_module;

use super::{Next, ParserRule, RResult};
use crate::ir::expr::{BinOp, BinOpKind};
use crate::lex::{buffer::*, kind::*};
use crate::parse::combinator::opt::Opt;
use crate::parse::rules::RuleErr;
use crate::parse::{matc::*, stream::TokenStream};

/// A composition of tokens that resolve into a value.
#[derive(Debug, Clone)]
pub enum Expr {
    Ident(TokenId),
    Lit(TokenId),
    Bin(BinOp, Box<Expr>, Box<Expr>),
    Ret(Span, Option<Box<Expr>>),
    Binding(TyBinding),
    Call { span: Span, func: TokenId },
}

#[derive(Debug, Clone, Copy)]
pub struct TyBinding {
    pub ident: TokenId,
    pub ty: TokenId,
}

/// Produce the next [`TyBinding`].
pub struct TyBindingRule;

impl<'a> ParserRule<'a> for TyBindingRule {
    type Output = TyBinding;

    fn parse(
        buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        stack: &mut Vec<TokenId>,
    ) -> RResult<'a, Self::Output> {
        let (ident, _, ty) =
            <(Next<Ident>, Next<Colon>, Next<Ident>) as ParserRule>::parse(buffer, stream, stack)?;

        Ok(TyBinding { ident, ty })
    }
}

impl From<BinOpTokens> for BinOpKind {
    fn from(value: BinOpTokens) -> Self {
        match value {
            BinOpTokens::Plus(_) => Self::Add,
            BinOpTokens::Asterisk(_) => Self::Multiply,
            BinOpTokens::PlusEquals(_, _) => Self::Multiply,
            _ => panic!(),
        }
    }
}

trait Precedence {
    fn precedence(&self) -> usize;
}

/// Produce the next [`BinOpKind`].
pub struct BinOpKindRule;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOpTokens {
    Plus(TokenId),
    Asterisk(TokenId),
    PlusEquals(TokenId, TokenId),
    OpenParen(TokenId),
    CloseParen(TokenId),
}

impl BinOpTokens {
    pub fn span(&self, buffer: &TokenBuffer) -> Span {
        match self {
            Self::Plus(t) | Self::Asterisk(t) | Self::OpenParen(t) | Self::CloseParen(t) => {
                buffer.span(*t)
            }
            Self::PlusEquals(t1, t2) => Span::from_spans(buffer.span(*t1), buffer.span(*t2)),
        }
    }
}

impl Precedence for BinOpTokens {
    fn precedence(&self) -> usize {
        match self {
            Self::Plus(_) | Self::PlusEquals(_, _) | Self::OpenParen(_) | Self::CloseParen(_) => 1,
            Self::Asterisk(_) => 2,
        }
    }
}

impl<'a> ParserRule<'a> for BinOpKindRule {
    type Output = BinOpTokens;

    fn parse(
        _buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        _stack: &mut Vec<TokenId>,
    ) -> RResult<'a, Self::Output> {
        let str = *stream;
        match stream.peek_kind() {
            Some(TokenKind::Plus) => {
                let plus = stream.next().unwrap();
                if stream.peek_kind() == Some(TokenKind::Equals) {
                    let equals = stream.next().unwrap();
                    return Ok(BinOpTokens::PlusEquals(plus, equals));
                }
            }
            _ => {}
        }

        *stream = str;
        match stream.peek_kind() {
            Some(TokenKind::Plus) => Ok(BinOpTokens::Plus(stream.next().unwrap())),
            Some(TokenKind::Asterisk) => Ok(BinOpTokens::Asterisk(stream.next().unwrap())),
            Some(TokenKind::OpenParen) => Ok(BinOpTokens::OpenParen(stream.next().unwrap())),
            Some(TokenKind::CloseParen) => Ok(BinOpTokens::CloseParen(stream.next().unwrap())),
            kind => Err(RuleErr::from_diag(stream.error(format!(
                "expected binary operator, got `{}`",
                kind.map(|k| k.as_str()).unwrap_or("???")
            )))),
        }
    }
}

/// Collection of viable terms within an expression.
///
/// Directly translate to [`Expr`] with [`Term::into_expr`].
#[derive(Debug)]
pub enum Term {
    Lit(TokenId),
    Ident(TokenId),
    Call { span: Span, func: TokenId },
}

impl Term {
    pub fn into_expr(self) -> Expr {
        match self {
            Self::Lit(id) => Expr::Lit(id),
            Self::Ident(id) => Expr::Ident(id),
            Self::Call { span, func } => Expr::Call { span, func },
        }
    }
}

/// Produce the next [`Term`].
pub struct TermRule;

impl<'a> ParserRule<'a> for TermRule {
    type Output = Term;

    fn parse(
        buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        stack: &mut Vec<TokenId>,
    ) -> RResult<'a, Self::Output> {
        match stream.peek_kind() {
            Some(TokenKind::Ident) => {
                let ident = Next::<Ident>::parse(buffer, stream, stack)?;
                match stream.peek_kind() {
                    Some(TokenKind::OpenParen) => {
                        let (_, close) =
                            <(Next<OpenParen>, Next<CloseParen>)>::parse(buffer, stream, stack)
                                .unwrap();

                        let span =
                            Span::from_range_u32(buffer.span(ident).start..buffer.span(close).end);

                        Ok(Term::Call { span, func: ident })
                    }
                    _ => Ok(Term::Ident(ident)),
                }
            }
            Some(TokenKind::Int) => {
                let lit = Next::<Int>::parse(buffer, stream, stack).unwrap();
                Ok(Term::Lit(lit))
            }
            kind => Err(RuleErr::from_diag(stream.full_error(
                "malformed expression",
                buffer.span(stream.peek().unwrap()),
                format!(
                    "expected one of `int literal`, `function call`, or `identifier`, got `{}`",
                    kind.map(|k| k.as_str()).unwrap_or("???")
                ),
            ))),
        }
    }
}

/// Produce the next [`Expr`].
#[derive(Debug, Default)]
pub struct ExprRule;

impl<'a> ParserRule<'a> for ExprRule {
    type Output = Expr;

    fn parse(
        buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        stack: &mut Vec<TokenId>,
    ) -> RResult<'a, Self::Output> {
        if stream.peek_kind() == Some(TokenKind::Ret) {
            let span = buffer.span(stream.next().unwrap());
            let expr = Opt::<ExprRule>::parse(buffer, stream, stack)?;
            return Ok(Expr::Ret(span, expr.map(Box::new)));
        }

        #[derive(Debug)]
        enum Item {
            Op(BinOpTokens),
            Term(Term),
        }

        // https://en.wikipedia.org/wiki/Shunting_yard_algorithm
        let mut expr_stack = Vec::new();
        let mut operators = Vec::new();
        let mut open_parens = 0;
        while stream.peek_kind().is_some_and(|k| k != TokenKind::Semi)
            && stream
                .peek_kind()
                .is_some_and(|k| k != TokenKind::CloseCurly)
        {
            if let Some(op) = Opt::<BinOpKindRule>::parse(buffer, stream, stack)? {
                match op {
                    BinOpTokens::Plus(_) | BinOpTokens::Asterisk(_) => {
                        while operators.last().is_some_and(|t: &BinOpTokens| {
                            t.precedence() > op.precedence()
                                && !matches!(t, BinOpTokens::OpenParen(_))
                        }) {
                            expr_stack.push(Item::Op(operators.pop().unwrap()));
                        }
                        operators.push(op)
                    }
                    BinOpTokens::OpenParen(_) => {
                        open_parens += 1;
                        operators.push(op)
                    }
                    BinOpTokens::CloseParen(t) => {
                        open_parens -= 1;
                        if open_parens < 0 {
                            return Err(RuleErr::from_diag(stream.full_error(
                                "malformed expression",
                                buffer.span(t),
                                "unopened delimiter",
                            )));
                        }

                        loop {
                            let Some(op) = operators.pop() else {
                                unreachable!()
                            };

                            match op {
                                BinOpTokens::OpenParen(_) => {
                                    break;
                                }
                                _ => expr_stack.push(Item::Op(op)),
                            }
                        }
                    }
                    _ => panic!(),
                }
            } else {
                let term = TermRule::parse(buffer, stream, stack)?;
                expr_stack.push(Item::Term(term));
            }
        }

        // no operators means there should only be one term
        if operators.is_empty() && expr_stack.len() == 1 {
            assert!(expr_stack.len() == 1, "todo error");
            return Ok(match expr_stack.pop().unwrap() {
                Item::Term(term) => term.into_expr(),
                _ => panic!(),
            });
        }

        expr_stack.extend(operators.drain(..).rev().map(Item::Op));
        expr_stack.reverse();

        for item in expr_stack.iter() {
            if let Item::Op(op) = item {
                match op {
                    BinOpTokens::OpenParen(t) => {
                        return Err(RuleErr::from_diag(stream.full_error(
                            "malformed expression",
                            buffer.span(*t),
                            "unclosed delimiter",
                        )))
                    }
                    _ => {}
                }
            }
        }

        let mut accum_op = None;
        let mut terms = Vec::with_capacity(expr_stack.len());
        while let Some(item) = expr_stack.pop() {
            match item {
                Item::Op(op) => match accum_op {
                    Some(acop) => {
                        accum_op = Some(Expr::Bin(
                            BinOp {
                                span: op.span(buffer),
                                kind: BinOpKind::from(op),
                            },
                            Box::new(terms.pop().unwrap()),
                            Box::new(acop),
                        ));
                    }
                    None => {
                        accum_op = Some(Expr::Bin(
                            BinOp {
                                span: op.span(buffer),
                                kind: BinOpKind::from(op),
                            },
                            Box::new(terms.pop().unwrap()),
                            Box::new(terms.pop().unwrap()),
                        ));
                    }
                },
                Item::Term(next) => {
                    terms.push(next.into_expr());
                }
            }
        }

        Ok(accum_op.unwrap())
    }
}
