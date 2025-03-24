use pebblec_parse::lex::buffer::Span;
use std::collections::HashMap;
use std::marker::PhantomData;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ident {
    pub span: Span,
    pub id: IdentId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IdentId(usize);

// TODO: make this take references
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct IdentStore<'a> {
    map: HashMap<String, IdentId>,
    buf: Vec<String>,
    _phantom: PhantomData<&'a str>,
}

impl<'a> IdentStore<'a> {
    pub fn store(&mut self, ident: &str) -> IdentId {
        if let Some(id) = self.map.get(ident) {
            *id
        } else {
            let id = IdentId(self.buf.len());
            self.buf.push(ident.to_string());
            self.map.insert(ident.to_string(), id);
            id
        }
    }

    pub fn get_ident(&self, id: IdentId) -> Option<&str> {
        self.buf.get(id.0).map(|s| s.as_str())
    }

    pub fn get_id(&self, str: &str) -> Option<IdentId> {
        self.map.get(&str.to_owned()).copied()
    }
}
