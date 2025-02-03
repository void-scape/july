use crate::lex::buffer::Span;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ident {
    pub span: Span,
    pub id: IdentId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IdentId(usize);

#[derive(Debug, Default)]
pub struct IdentStore<'a> {
    map: HashMap<&'a str, IdentId>,
    buf: Vec<&'a str>,
}

impl<'a> IdentStore<'a> {
    pub fn store(&mut self, ident: &'a str) -> IdentId {
        if let Some(id) = self.map.get(ident) {
            *id
        } else {
            let id = IdentId(self.buf.len());
            self.map.insert(ident, id);
            self.buf.push(ident);
            id
        }
    }

    #[track_caller]
    pub fn ident(&self, id: IdentId) -> &'a str {
        *self.buf.get(id.0).expect("invalid ident id")
    }
}
