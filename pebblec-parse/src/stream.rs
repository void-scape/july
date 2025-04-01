use super::matc::{DelimPair, MatchTokenKind};
use super::rules::PErr;
use crate::diagnostic::Diag;
use crate::lex::buffer::{Buffer, Span, TokenBuffer, TokenId, TokenQuery};
use crate::lex::kind::TokenKind;
use annotate_snippets::Level;
use std::borrow::Cow;

impl TokenBuffer {
    pub fn stream(&self) -> TokenStream {
        TokenStream::new(self)
    }
}

// TODO: makes many assumptions
#[derive(Debug, Clone, Copy)]
pub struct TokenStream<'a> {
    buffer: &'a TokenBuffer,
    start: usize,
    end: usize,
    index: usize,
}

impl<'a, 's> Iterator for TokenStream<'a> {
    type Item = TokenId;

    fn next(&mut self) -> Option<Self::Item> {
        self.next()
    }
}

impl<'a> Buffer<'a> for TokenStream<'a> {
    fn token_buffer(&self) -> &TokenBuffer {
        self.buffer
    }
}

impl<'a, 's> TokenStream<'a> {
    fn new(buffer: &'a TokenBuffer) -> Self {
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

    #[allow(unused)]
    pub fn print(&self, max: usize) {
        let mut slf = *self;
        for _ in 0..max {
            if let Some(t) = slf.next() {
                println!("{}", self.as_str(t));
            }
        }
    }

    #[allow(unused)]
    pub fn print_kind(&self, max: usize) {
        let mut slf = *self;
        for _ in 0..max {
            if let Some(t) = slf.next() {
                println!("{:?}", self.kind(t));
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
        assert!(
            self.next()
                .is_none_or(|t| T::matches_open(Some(self.kind(t))))
        );
        let offset = self.find_matched_delim_offset::<T>();
        self.eat_n(offset);
        assert!(
            self.next()
                .is_none_or(|t| T::matches_close(Some(self.kind(t))))
        );
    }

    pub fn find_offset<T: MatchTokenKind>(&self) -> usize {
        let index = self.index;
        let mut other = self.clone();
        other.eat_until::<T>();
        other.index.saturating_sub(index)
    }

    #[track_caller]
    pub fn report_error(&self, title: impl Into<Cow<'static, str>>, span: Span) -> Diag {
        Diag::new(Level::Error, self.buffer.source(), span, title, Vec::new())
    }

    #[track_caller]
    pub fn error(&self, title: impl Into<Cow<'static, str>>) -> Diag {
        let prev = self.prev();
        match self.token_buffer().token(prev.next()) {
            Some(next) => self.report_error(title, next.span),
            None => {
                let span = self.buffer.span(prev);
                self.report_error(title, span)
            }
        }
    }

    #[track_caller]
    pub fn recover(&self, title: impl Into<Cow<'static, str>>) -> PErr {
        PErr::Recover(self.error(title))
    }

    #[track_caller]
    pub fn fail(&self, title: impl Into<Cow<'static, str>>) -> PErr {
        PErr::Fail(self.error(title))
    }

    pub fn remaining(&self) -> usize {
        self.end.saturating_sub(self.index)
    }

    pub fn is_empty(&self) -> bool {
        self.end <= self.index
    }

    pub fn peek(&self) -> Option<TokenId> {
        let idx = self.start + self.index;
        (idx < self.end).then_some(TokenId::new(idx as u32, self.buffer.source_id() as u32))
    }

    pub fn peekn(&self, n: usize) -> Option<TokenId> {
        let idx = self.start + self.index + n;
        (idx < self.end).then_some(TokenId::new(idx as u32, self.buffer.source_id() as u32))
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

    pub fn back(&mut self) {
        self.index = self.index.saturating_sub(1);
    }

    pub fn shrink(&mut self) {
        self.end = self.end.saturating_sub(1);
    }

    pub fn eat_n(&mut self, n: usize) {
        self.index += n;
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
            return TokenId::new(0, self.buffer.source_id() as u32);
        }

        let idx = self.index - 1;
        (idx < self.end)
            .then_some(TokenId::new(idx as u32, self.buffer.source_id() as u32))
            .unwrap_or_else(|| TokenId::new(self.end as u32 - 1, self.buffer.source_id() as u32))
    }

    pub fn match_peek<T: MatchTokenKind>(&self) -> bool {
        self.peek()
            .is_some_and(|t| T::matches(Some(self.buffer.kind(t))))
    }
}
