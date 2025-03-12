use super::matc::{DelimPair, MatchTokenKind, OpenCurly};
use super::rules::{Next, PErr, ParserRule};
use crate::diagnostic::{Diag, Msg};
use crate::lex::buffer::{Buffer, Span, TokenBuffer, TokenId, TokenQuery};
use crate::lex::kind::TokenKind;

impl TokenBuffer<'_> {
    pub fn stream(&self) -> TokenStream {
        TokenStream::new(self)
    }
}

// TODO: makes many assumptions
#[derive(Debug, Clone, Copy)]
pub struct TokenStream<'a, 's> {
    buffer: &'a TokenBuffer<'s>,
    start: usize,
    end: usize,
    index: usize,
}

impl<'a, 's> Buffer<'s> for TokenStream<'a, 's> {
    fn token_buffer(&self) -> &TokenBuffer<'s> {
        self.buffer
    }
}

impl<'a, 's> TokenStream<'a, 's> {
    fn new(buffer: &'a TokenBuffer<'s>) -> Self {
        Self {
            start: 0,
            end: buffer.len(),
            index: 0,
            buffer,
        }
    }

    pub fn slice(&self, len: usize) -> Self {
        Self {
            end: (self.index + len).min(self.end),
            start: self.start,
            index: self.index,
            buffer: self.buffer,
        }
    }

    pub fn print(&self, max: usize) {
        let mut slf = *self;
        for _ in 0..max {
            if let Some(t) = slf.next() {
                println!("{}", self.as_str(t));
            }
        }
    }

    pub fn find_matched_delim_offset<T: DelimPair>(&self) -> usize {
        let index = self.index;
        let mut other = self.clone();

        let mut open = 1;
        let mut closed = 0;

        while let Some(t) = other.peek() {
            if T::matches_open(Some(other.buffer.kind(t))) {
                open += 1;
            } else if T::matches_close(Some(other.buffer.kind(t))) {
                closed += 1;

                if open == closed {
                    break;
                }
            }

            other.eat();
        }

        other.index.saturating_sub(index)
    }

    pub fn consume_matched_delimiters_inclusive<T: DelimPair>(&mut self) {
        assert!(self
            .next()
            .is_some_and(|t| T::matches_open(Some(self.kind(t)))));
        let offset = self.find_matched_delim_offset::<T>();
        self.eat_n(offset);
        assert!(self
            .next()
            .is_some_and(|t| T::matches_close(Some(self.kind(t)))));
    }

    pub fn find_offset<T: MatchTokenKind>(&self) -> usize {
        let index = self.index;
        let mut other = self.clone();
        other.eat_until::<T>();
        other.index.saturating_sub(index)
    }

    #[track_caller]
    pub fn full_error(&self, title: impl Into<String>, span: Span) -> Diag<'s> {
        Diag::sourced(title, self.buffer.source(), Msg::error_span(span))
            .loc(std::panic::Location::caller())
    }

    #[track_caller]
    pub fn error(&self, msg: impl Into<String>) -> Diag<'s> {
        let prev = self.prev();
        match self.buffer.next(prev) {
            Some(next) => {
                let span = self.buffer.span(next);
                self.full_error(msg, span)
            }
            None => {
                let span = self.buffer.span(prev);
                self.full_error(msg, span)
            }
        }
    }

    #[track_caller]
    pub fn recover(&self, msg: impl Into<String>) -> PErr<'s> {
        PErr::Recover(self.error(msg))
    }

    #[track_caller]
    pub fn fail(&self, msg: impl Into<String>) -> PErr<'s> {
        PErr::Fail(self.error(msg))
    }

    pub fn remaining(&self) -> usize {
        self.end.saturating_sub(self.index)
    }

    pub fn is_empty(&self) -> bool {
        self.end <= self.index
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

    // TODO: take slice from buffer
    pub fn drain(&mut self) -> Vec<TokenId> {
        let mut tokens = Vec::with_capacity(self.remaining());
        while let Some(next) = self.next() {
            tokens.push(next);
        }
        tokens
    }

    #[track_caller]
    pub fn expect(&mut self) -> TokenId {
        self.next().unwrap()
    }

    pub fn eat(&mut self) {
        self.index += 1;
    }

    pub fn eat_n(&mut self, n: usize) {
        self.index += n;
    }

    pub fn eat_to_index(&mut self, index: usize) {
        self.index += index.saturating_sub(self.index);
    }

    pub fn eat_until<T: MatchTokenKind>(&mut self) {
        while self
            .peek()
            .is_some_and(|t| !T::matches(Some(self.buffer.kind(t))))
        {
            self.eat();
        }
    }

    pub fn eat_until_consume<T: MatchTokenKind>(&mut self) {
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

    pub fn match_peek<T: MatchTokenKind>(&self) -> bool {
        self.peek()
            .is_some_and(|t| T::matches(Some(self.buffer.kind(t))))
    }

    pub fn match_peekn<T: MatchTokenKind>(&self, n: usize) -> bool {
        self.peekn(n)
            .is_some_and(|t| T::matches(Some(self.buffer.kind(t))))
    }
}
