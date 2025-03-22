use pebblec_parse::lex::buffer::Span;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ident {
    pub span: Span,
    pub id: IdentId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IdentId(usize);

#[derive(Debug, Default, Clone, PartialEq, Eq)]
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

    pub fn get_ident(&self, id: IdentId) -> Option<&'a str> {
        self.buf.get(id.0).copied()
    }

    pub fn get_id(&self, str: &str) -> Option<IdentId> {
        self.map.get(&str).copied()
    }
}
