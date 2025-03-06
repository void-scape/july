use crate::lex::buffer::Span;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Lit<'a> {
    pub span: Span,
    pub kind: &'a LitKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LitKind {
    Int(i64),
}

impl LitKind {
    pub fn is_int(&self) -> bool {
        matches!(self, Self::Int(_))
    }
}
