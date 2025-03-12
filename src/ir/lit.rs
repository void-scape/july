use std::hash::Hash;
use crate::lex::buffer::Span;

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub struct Lit<'a> {
    pub span: Span,
    pub kind: &'a LitKind,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LitKind {
    Int(u64),
    Float(f64),
}

impl Hash for LitKind {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Self::Int(int) => int.hash(state),
            Self::Float(float) => ((float * i64::MAX as f64) as i64).hash(state),
        }
    }
}

impl LitKind {
    pub fn is_int(&self) -> bool {
        matches!(self, Self::Int(_))
    }
}
