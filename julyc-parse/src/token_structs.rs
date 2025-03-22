use crate::lex::buffer::TokenId;
use std::ops::Deref;

#[derive(Debug, Clone)]
pub struct Delimited<T> {
    pub open: TokenId,
    pub inner: T,
    pub close: TokenId,
}

impl<T> Deref for Delimited<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
