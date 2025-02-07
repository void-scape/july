use super::strukt::StructDef;
use super::{Next, ParserRule, RResult};
use crate::ir::{AssignKind, BinOpKind};
use crate::lex::{buffer::*, kind::*};
use crate::parse::combinator::opt::Opt;
use crate::parse::rules::strukt::StructDefRule;
use crate::parse::{matc::*, stream::TokenStream};

/// A composition of tokens that resolve into a value.
#[derive(Debug, Clone)]
pub enum Expr {
    Ident(TokenId),
    Lit(TokenId),
    Bin(BinOp, Box<Expr>, Box<Expr>),
    Ret(Span, Option<Box<Expr>>),
    Assign(Assign),
    StructDef(StructDef),
    Call { span: Span, func: TokenId },
}

#[derive(Debug, Clone, Copy)]
pub struct BinOp {
    pub span: Span,
    pub kind: BinOpKind,
}

#[derive(Debug, Clone)]
pub struct Assign {
    pub assign_span: Span,
    pub kind: AssignKind,
    pub lhs: Box<Expr>,
    pub rhs: Box<Expr>,
}

impl From<BinOpTokens> for BinOpKind {
    fn from(value: BinOpTokens) -> Self {
        match value {
            BinOpTokens::Plus(_) => Self::Add,
            BinOpTokens::Hyphen(_) => Self::Sub,
            BinOpTokens::Asterisk(_) => Self::Mul,
            BinOpTokens::Field(_) => Self::Field,
            _ => panic!(),
        }
    }
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
    Field(TokenId),
}

impl BinOpTokens {
    pub fn span(&self, buffer: &TokenBuffer) -> Span {
        match self {
            Self::Hyphen(t)
            | Self::Plus(t)
            | Self::Asterisk(t)
            | Self::OpenParen(t)
            | Self::Field(t)
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
            Self::Hyphen(_) | Self::Plus(_) | Self::OpenParen(_) | Self::CloseParen(_) => 1,
            Self::Asterisk(_) => 2,
            Self::Field(_) => 3,
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
                Err(stream.full_error(
                    "invalid assign",
                    buffer.span(equals),
                    "cannot assign expression",
                ))
            }
            Some(TokenKind::Plus) => Ok(BinOpTokens::Plus(stream.expect())),
            Some(TokenKind::Hyphen) => Ok(BinOpTokens::Hyphen(stream.expect())),
            Some(TokenKind::Asterisk) => Ok(BinOpTokens::Asterisk(stream.expect())),
            Some(TokenKind::OpenParen) => Ok(BinOpTokens::OpenParen(stream.expect())),
            Some(TokenKind::CloseParen) => Ok(BinOpTokens::CloseParen(stream.expect())),
            Some(TokenKind::Dot) => Ok(BinOpTokens::Field(stream.expect())),
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
                        let span = Span::from_spans(buffer.span(ident), buffer.span(close));

                        Ok(Term::Call { span, func: ident })
                    }
                    _ => Ok(Term::Ident(ident)),
                }
            }
            Some(TokenKind::Int) => {
                let lit = Next::<Int>::parse(buffer, stream, stack).unwrap();
                Ok(Term::Lit(lit))
            }
            kind => Err(stream.full_error(
                "malformed expression",
                buffer.span(stream.peek().unwrap()),
                format!(
                    "expected one of `int literal`, `function call`, or `identifier`, got `{}`",
                    kind.map(|k| k.as_str()).unwrap_or("???")
                ),
            )),
        }
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
        while stream.match_peek::<Not<Any<(Semi, CloseCurly, Comma)>>>() {
            if let Some(op) = Opt::<BinOpKindRule>::parse(buffer, stream, stack)? {
                match op {
                    BinOpTokens::Field(_)
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

        //let mut assigns = 0;
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
