use super::block::Block;
use super::enom::EnumDef;
use super::strukt::StructDef;
use super::{Next, ParserRule, RResult};
use crate::ir::{AssignKind, BinOpKind};
use crate::lex::{buffer::*, kind::*};
use crate::parse::combinator::opt::Opt;
use crate::parse::combinator::wile::{NextToken, While};
use crate::parse::rules::enom::EnumDefRule;
use crate::parse::rules::strukt::StructDefRule;
use crate::parse::{matc::*, stream::TokenStream};

/// A composition of tokens that resolve into a value.
#[derive(Debug, Clone)]
pub enum Expr {
    Ident(TokenId),
    Lit(TokenId),
    Str(TokenId),
    Bool(TokenId),
    Bin(BinOpKind, Box<Expr>, Box<Expr>),
    Ret(Span, Option<Box<Expr>>),
    Assign(Assign),
    StructDef(StructDef),
    EnumDef(EnumDef),
    TakeRef(TokenId, Box<Expr>),
    Access {
        span: Span,
        lhs: Box<Expr>,
        field: TokenId,
    },
    Call {
        span: Span,
        func: TokenId,
        args: Vec<Expr>,
    },
    If(TokenId, Box<Expr>, Block, Option<Block>),
    Loop(TokenId, Block),
}

impl From<BinOpTokens> for BinOpKind {
    fn from(value: BinOpTokens) -> Self {
        match value {
            BinOpTokens::Plus(_) => BinOpKind::Add,
            BinOpTokens::Hyphen(_) => BinOpKind::Sub,
            BinOpTokens::Asterisk(_) => BinOpKind::Mul,
            BinOpTokens::DoubleEquals(_) => BinOpKind::Eq,
            _ => panic!(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Assign {
    pub assign_span: Span,
    pub kind: AssignKind,
    pub lhs: Box<Expr>,
    pub rhs: Box<Expr>,
}

/// Produce the next [`BinOpKind`].
pub struct BinOpKindRule;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOpTokens {
    Plus(TokenId),
    Hyphen(TokenId),
    Asterisk(TokenId),
    OpenParen(TokenId),
    CloseParen(TokenId),
    DoubleEquals(TokenId),
}

impl BinOpTokens {
    pub fn span(&self, buffer: &TokenBuffer) -> Span {
        match self {
            Self::Hyphen(t)
            | Self::Plus(t)
            | Self::Asterisk(t)
            | Self::OpenParen(t)
            | Self::DoubleEquals(t)
            | Self::CloseParen(t) => buffer.span(*t),
        }
    }
}

trait Precedence {
    fn precedence(&self) -> usize;
}

impl Precedence for BinOpTokens {
    fn precedence(&self) -> usize {
        match self {
            Self::DoubleEquals(_)
            | Self::Hyphen(_)
            | Self::Plus(_)
            | Self::OpenParen(_)
            | Self::CloseParen(_) => 1,
            Self::Asterisk(_) => 2,
        }
    }
}

impl<'a> ParserRule<'a> for BinOpKindRule {
    type Output = BinOpTokens;

    fn parse(
        buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        _stack: &mut Vec<TokenId>,
    ) -> RResult<'a, Self::Output> {
        let str = *stream;

        match stream.peek_kind() {
            Some(TokenKind::Plus) => {
                let plus = stream.expect();
                if stream.peek_kind() == Some(TokenKind::Equals) {
                    let equals = stream.expect();
                    return Err(stream.full_error(
                        "invalid assign",
                        Span::from_spans(buffer.span(plus), buffer.span(equals)),
                        "cannot assign expression",
                    ));
                }
            }
            _ => {}
        }

        *stream = str;
        match stream.peek_kind() {
            Some(TokenKind::Equals) => {
                let equals = stream.expect();
                if stream.peek_kind() == Some(TokenKind::Equals) {
                    stream.expect();
                    Ok(BinOpTokens::DoubleEquals(equals))
                } else {
                    Err(stream.full_error(
                        "invalid assign",
                        buffer.span(equals),
                        "cannot assign expression",
                    ))
                }
            }
            Some(TokenKind::Plus) => Ok(BinOpTokens::Plus(stream.expect())),
            Some(TokenKind::Hyphen) => Ok(BinOpTokens::Hyphen(stream.expect())),
            Some(TokenKind::Asterisk) => Ok(BinOpTokens::Asterisk(stream.expect())),
            Some(TokenKind::OpenParen) => Ok(BinOpTokens::OpenParen(stream.expect())),
            Some(TokenKind::CloseParen) => Ok(BinOpTokens::CloseParen(stream.expect())),
            kind => Err(stream.error(format!(
                "expected binary operator, got `{}`",
                kind.map(|k| k.as_str()).unwrap_or("???")
            ))),
        }
    }
}

/// Collection of viable terms within an expression.
///
/// Directly translate to [`Expr`] with [`Term::into_expr`].
#[derive(Debug, Clone)]
pub enum Term {
    Lit(TokenId),
    Str(TokenId),
    Ident(TokenId),
    Bool(TokenId),
    Access {
        span: Span,
        lhs: Box<Expr>,
        field: TokenId,
    },
    Call {
        span: Span,
        func: TokenId,
        args: Vec<Expr>,
    },
    Ref(TokenId, Box<Term>),
    Deref(TokenId, Box<Term>),
}

impl Term {
    pub fn into_expr(self) -> Expr {
        match self {
            Self::Lit(id) => Expr::Lit(id),
            Self::Str(id) => Expr::Str(id),
            Self::Ident(id) => Expr::Ident(id),
            Self::Bool(bool) => Expr::Bool(bool),
            Self::Call { span, func, args } => Expr::Call { span, func, args },
            Self::Ref(token, term) => Expr::TakeRef(token, Box::new(term.clone().into_expr())),
            Self::Access { span, lhs, field } => Expr::Access { span, lhs, field },
            _ => todo!(), //Self::Deref(token, term) => Expr::Deref(token, Box::new(term.clone().into_expr())),
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
        let Some(start_span) = stream.peek().map(|t| stream.span(t)) else {
            return Err(stream.error("expected expression term"));
        };

        let term = match stream.peek_kind() {
            Some(TokenKind::Ident) => {
                let ident = Next::<Ident>::parse(buffer, stream, stack)?;
                match stream.peek_kind() {
                    Some(TokenKind::OpenParen) => {
                        let str = *stream;
                        match Next::<OpenParen>::parse(buffer, stream, stack) {
                            Err(err) => {
                                *stream = str;
                                return Err(err);
                            }
                            Ok(_) => {}
                        }
                        if !stream.match_peek::<CloseParen>() {
                            let offset = stream.find_matched_delim_offset::<Paren>();
                            let mut slice = stream.slice(offset);
                            let args = match While::<
                                NextToken<Not<CloseParen>>,
                                (ExprRule, Opt<Next<Comma>>),
                            >::parse(
                                buffer, &mut slice, stack
                            ) {
                                Err(err) => {
                                    *stream = str;
                                    return Err(err);
                                }
                                Ok(args) => args,
                            };

                            stream.eat_n(offset);
                            let close = match Next::<CloseParen>::parse(buffer, stream, stack) {
                                Err(err) => {
                                    *stream = str;
                                    return Err(err);
                                }
                                Ok(close) => close,
                            };

                            let span = Span::from_spans(buffer.span(ident), buffer.span(close));

                            Ok(Term::Call {
                                span,
                                func: ident,
                                args: args.into_iter().map(|(arg, _)| arg).collect(),
                            })
                        } else {
                            let close = match Next::<CloseParen>::parse(buffer, stream, stack) {
                                Err(err) => {
                                    *stream = str;
                                    return Err(err);
                                }
                                Ok(close) => close,
                            };

                            let span = Span::from_spans(buffer.span(ident), buffer.span(close));

                            Ok(Term::Call {
                                span,
                                func: ident,
                                args: Vec::new(),
                            })
                        }
                    }
                    _ => Ok(Term::Ident(ident)),
                }
            }
            Some(TokenKind::Float) | Some(TokenKind::Int) => Ok(Term::Lit(stream.expect())),
            Some(TokenKind::True) | Some(TokenKind::False) => Ok(Term::Bool(stream.expect())),
            Some(TokenKind::Str) => Ok(Term::Str(stream.expect())),
            Some(TokenKind::Ampersand) => Ok(Term::Ref(
                stream.expect(),
                Box::new(TermRule::parse(buffer, stream, stack)?),
            )),
            kind => Err(stream.full_error(
                "malformed expression",
                buffer.span(stream.peek().unwrap()),
                format!(
                    "expected one of `int literal`, `function call`, or `identifier`, got `{}`",
                    kind.map(|k| k.as_str()).unwrap_or("???")
                ),
            )),
        };

        let mut term_result = term?;
        while stream.match_peek::<Dot>() {
            let dot = stream.expect();

            if !stream.match_peek::<Ident>() {
                return Err(stream.full_error(
                    "malformed field access",
                    buffer.span(dot),
                    "expected identifier after `.`",
                ));
            }

            let field = stream.expect();
            let expr = term_result.into_expr();
            term_result = Term::Access {
                span: Span::from_spans(start_span, buffer.span(field)),
                lhs: Box::new(expr),
                field,
            };
        }

        Ok(term_result)
    }
}

/// Produce the next [`Assign`].
#[derive(Debug, Default)]
pub struct AssignRule;

impl<'a> ParserRule<'a> for AssignRule {
    type Output = Expr;

    fn parse(
        buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        stack: &mut Vec<TokenId>,
    ) -> RResult<'a, Self::Output> {
        let mut slice = stream.slice(stream.find_offset::<Any<(Equals, Plus)>>());
        let expr = ExprRule::parse(buffer, &mut slice, stack)?;
        stream.eat_until::<Any<(Equals, Plus)>>();

        match stream.peek_kind() {
            Some(TokenKind::Equals) => Ok(Expr::Assign(Assign {
                assign_span: buffer.span(stream.expect()),
                kind: AssignKind::Equals,
                lhs: Box::new(expr),
                rhs: Box::new(ExprRule::parse(buffer, stream, stack)?),
            })),
            Some(TokenKind::Plus) => {
                let plus = stream.expect();
                let next = stream.next();
                match next.map(|next| buffer.kind(next)) {
                    Some(TokenKind::Equals) => Ok(Expr::Assign(Assign {
                        assign_span: Span::from_spans(
                            buffer.span(plus),
                            buffer.span(next.unwrap()),
                        ),
                        kind: AssignKind::Add,
                        lhs: Box::new(expr),
                        rhs: Box::new(ExprRule::parse(buffer, stream, stack)?),
                    })),
                    _ => Err(stream.error("expected `+`")),
                }
            }
            _ => Err(stream.error("expected assignment")),
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
        if stream.match_peek::<Ret>() {
            let span = buffer.span(stream.next().unwrap());
            let expr = Opt::<ExprRule>::parse(buffer, stream, stack)?;
            return Ok(Expr::Ret(span, expr.map(Box::new)));
        }

        let str = *stream;
        if stream.match_peek::<Ident>() {
            stream.eat();
            if stream.match_peek::<OpenCurly>() {
                *stream = str;
                let def = StructDefRule::parse(buffer, stream, stack)?;
                return Ok(Expr::StructDef(def));
            } else if stream.match_peek::<Colon>() {
                *stream = str;
                let def = EnumDefRule::parse(buffer, stream, stack)?;
                return Ok(Expr::EnumDef(def));
            }
        }
        *stream = str;

        #[derive(Debug)]
        enum Item {
            Op(BinOpTokens),
            Term(Term),
        }

        // https://en.wikipedia.org/wiki/Shunting_yard_algorithm
        let mut expr_stack = Vec::new();
        let mut operators = Vec::new();
        let mut open_parens = 0;
        while stream.match_peek::<Not<Any<(Semi, CloseCurly, Comma, OpenCurly)>>>() {
            if let Some(op) = Opt::<BinOpKindRule>::parse(buffer, stream, stack)? {
                match op {
                    BinOpTokens::DoubleEquals(_)
                    | BinOpTokens::Hyphen(_)
                    | BinOpTokens::Plus(_)
                    | BinOpTokens::Asterisk(_) => {
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
                            return Err(stream.full_error(
                                "malformed expression",
                                buffer.span(t),
                                "unopened delimiter",
                            ));
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
                        return Err(stream.full_error(
                            "malformed expression",
                            buffer.span(*t),
                            "unclosed delimiter",
                        ))
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
                            BinOpKind::from(op),
                            Box::new(terms.pop().unwrap()),
                            Box::new(acop),
                        ));
                    }
                    None => {
                        accum_op = Some(Expr::Bin(
                            BinOpKind::from(op),
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

        match accum_op {
            Some(op) => Ok(op),
            None => Err(stream.error("empty expression")),
        }
    }
}
