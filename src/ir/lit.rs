use super::ty::Ty;
use crate::lex::buffer::Span;

#[derive(Debug, Clone, Copy)]
pub struct Lit {
    pub span: Span,
    pub kind: LitId,
}

#[derive(Debug, Clone, Copy)]
pub enum LitKind<'a> {
    Int(i64),
    Str(&'a str),
}

impl LitKind<'_> {
    pub fn is_int(&self) -> bool {
        matches!(self, Self::Int(_))
    }

    pub fn satisfies(&self, ty: Ty) -> bool {
        match self {
            Self::Int(_) => ty.is_int(),
            _ => todo!(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LitId(usize);

#[derive(Debug, Default)]
pub struct LitStore<'a> {
    lits: Vec<LitKind<'a>>,
}

impl<'a> LitStore<'a> {
    pub fn store(&mut self, lit: LitKind<'a>) -> LitId {
        let idx = self.lits.len();
        self.lits.push(lit);
        LitId(idx)
    }

    pub fn get_lit(&self, id: LitId) -> Option<LitKind<'a>> {
        self.lits.get(id.0).copied()
    }
}
