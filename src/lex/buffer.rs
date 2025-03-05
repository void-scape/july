use crate::lex::kind::*;
use crate::unit::source::Source;
use std::ops::Range;

pub trait Buffer<'a> {
    fn token_buffer(&'a self) -> &'a TokenBuffer<'a>;
}

/// Query about a specific token with a [`TokenId`].
pub trait TokenQuery<'a> {
    fn kind(&'a self, token: TokenId) -> TokenKind;
    fn span(&'a self, token: TokenId) -> Span;
    fn int_lit(&'a self, token: TokenId) -> i64;
    fn ident(&'a self, token: TokenId) -> &'a str;
    fn as_str(&'a self, token: TokenId) -> &'a str;
    fn is_terminator(&'a self, token: TokenId) -> bool;
}

impl<'a, T> TokenQuery<'a> for T
where
    T: Buffer<'a>,
{
    #[track_caller]
    fn kind(&'a self, token: TokenId) -> TokenKind {
        self.token_buffer().kind(token)
    }

    #[track_caller]
    fn span(&'a self, token: TokenId) -> Span {
        self.token_buffer().span(token)
    }

    #[track_caller]
    fn ident(&'a self, token: TokenId) -> &'a str {
        self.token_buffer().ident(token)
    }

    #[track_caller]
    fn int_lit(&'a self, token: TokenId) -> i64 {
        self.token_buffer().int_lit(token)
    }

    #[track_caller]
    fn as_str(&'a self, token: TokenId) -> &'a str {
        self.token_buffer().as_str(token)
    }

    fn is_terminator(&'a self, token: TokenId) -> bool {
        self.token_buffer().is_terminator(token)
    }
}

/// Key into a buffer containing tokens generated by the lexer.
///
/// Used in [`TokenQuery`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TokenId(usize);

impl TokenId {
    pub fn new(id: usize) -> Self {
        Self(id)
    }
}

/// Storage of the generated tokens for a given source file.
#[derive(Debug)]
pub struct TokenBuffer<'a> {
    tokens: Vec<Token>,
    source: &'a Source,
}

impl<'a> TokenBuffer<'a> {
    pub fn new(tokens: Vec<Token>, source: &'a Source) -> Self {
        Self { tokens, source }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn source(&self) -> &Source {
        self.source
    }

    pub fn token(&self, token: TokenId) -> Option<&Token> {
        self.tokens.get(token.0)
    }

    pub fn next(&self, token: TokenId) -> Option<TokenId> {
        (self.len() > token.0 + 1).then_some(TokenId(token.0 + 1))
    }

    pub fn last(&self) -> Option<TokenId> {
        if !self.is_empty() {
            Some(TokenId(self.len() - 1))
        } else {
            None
        }
    }
}

impl<'a> TokenQuery<'a> for TokenBuffer<'a> {
    #[track_caller]
    fn kind(&self, token: TokenId) -> TokenKind {
        self.token(token)
            .expect("called `TokenQuery::kind` with an invalid token id")
            .kind
    }

    #[track_caller]
    fn span(&self, token: TokenId) -> Span {
        self.token(token)
            .expect("called `TokenQuery::span` with an invalid token id")
            .span
    }

    #[track_caller]
    fn ident(&'a self, token: TokenId) -> &'a str {
        if !matches!(self.kind(token), TokenKind::Ident) {
            panic!(
                "called `TokenQuery::ident` on a {:?} token",
                self.kind(token)
            );
        }

        self.as_str(token)
    }

    #[track_caller]
    fn as_str(&'a self, token: TokenId) -> &'a str {
        &self.source.raw()[self.span(token).range()]
    }

    #[track_caller]
    fn int_lit(&self, token: TokenId) -> i64 {
        if !matches!(self.kind(token), TokenKind::Int) {
            panic!(
                "called `TokenQuery::int_lit` on a {:?} token",
                self.kind(token)
            );
        }

        // TODO: check bounds of integer size somewhere
        let str = self.as_str(token);
        if str.contains("0x") {
            i64::from_str_radix(&str[2..], 16).unwrap()
        } else {
            str.parse().unwrap()
        }
    }

    fn is_terminator(&self, token: TokenId) -> bool {
        self.kind(token).is_terminator()
    }
}

/// Metadata about a token held within a [`TokenBuffer`].
#[derive(Debug)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
    //pub data: u64,
}

impl Token {
    pub fn new(kind: TokenKind, span: Span) -> Self {
        Self { kind, span }
    }
}

/// Token's position within a source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}

impl Span {
    pub fn empty() -> Self {
        Self { start: 0, end: 0 }
    }

    pub fn from_range(range: Range<usize>) -> Self {
        Self {
            start: range.start as u32,
            end: range.end as u32,
        }
    }

    pub fn from_range_u32(range: Range<u32>) -> Self {
        Self {
            start: range.start,
            end: range.end,
        }
    }

    pub fn from_spans(first: Self, second: Self) -> Self {
        if first.start > second.start {
            Self::from_spans(second, first)
        } else {
            Self::from_range_u32(first.start..second.end)
        }
    }

    pub fn range(&self) -> Range<usize> {
        self.start as usize..self.end as usize
    }
}
