use super::matc::MatchTokenKind;
use crate::lex::buffer::{TokenBuffer, TokenId, TokenQuery};
use crate::lex::kind::TokenKind;

impl TokenBuffer<'_> {
    pub fn stream(&self) -> TokenStream {
        TokenStream::new(self)
    }
}

// TODO: makes many assumptions
#[derive(Debug, Clone, Copy)]
pub struct TokenStream<'a> {
    buffer: &'a TokenBuffer<'a>,
    start: usize,
    end: usize,
    index: usize,
}

impl<'a> TokenStream<'a> {
    fn new(buffer: &'a TokenBuffer<'a>) -> Self {
        Self {
            start: 0,
            end: buffer.len(),
            index: 0,
            buffer,
        }
    }

    pub fn remaining(&self) -> usize {
        self.end.saturating_sub(self.index)
    }

    pub fn is_empty(&self) -> bool {
        self.remaining() == 0
    }

    pub fn peek(&self) -> Option<TokenId> {
        let idx = self.start + self.index;
        (idx < self.end).then_some(TokenId::new(idx))
    }

    pub fn peekn(&self, n: usize) -> Option<TokenId> {
        let idx = self.start + self.index + n;
        (idx < self.end).then_some(TokenId::new(idx))
    }

    pub fn peek_kind(&self) -> Option<TokenKind> {
        self.peek().map(|t| self.buffer.kind(t))
    }

    pub fn next(&mut self) -> Option<TokenId> {
        self.peek().and_then(|t| {
            self.index += 1;
            Some(t)
        })
    }

    pub fn eat(&mut self) {
        self.index += 1;
    }

    pub fn eat_until<T: MatchTokenKind>(&mut self, _token: T) {
        while self
            .peek()
            .is_some_and(|t| !T::matches(Some(self.buffer.kind(t))))
        {
            self.eat();
        }
    }

    pub fn eat_until_consume<T: MatchTokenKind>(&mut self, _token: T) {
        while self
            .peek()
            .is_some_and(|t| !T::matches(Some(self.buffer.kind(t))))
        {
            self.eat();
        }

        if T::matches(self.peek().map(|t| self.buffer.kind(t))) {
            self.eat();
        }
    }

    pub fn prev(&self) -> TokenId {
        if self.index == 0 {
            panic!("no prev on first element");
        }

        let idx = self.index - 1;
        (idx < self.end)
            .then_some(TokenId::new(idx))
            .unwrap_or_else(|| TokenId::new(self.end - 1))
    }

    pub fn match_peek<T: MatchTokenKind>(&self, _token: T) -> bool {
        self.peek()
            .is_some_and(|t| T::matches(Some(self.buffer.kind(t))))
    }
}
