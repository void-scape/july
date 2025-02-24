use super::ident::{Ident, IdentId};
use super::ty::store::TyId;
use super::FuncHash;
use crate::lex::buffer::Span;
use std::hash::{DefaultHasher, Hash, Hasher};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Sig<'a> {
    pub span: Span,
    pub ident: IdentId,
    pub ty: TyId,
    pub params: &'a [Param],
}

impl Sig<'_> {
    pub fn hash(&self) -> FuncHash {
        let mut hash = DefaultHasher::new();
        <Sig as Hash>::hash(self, &mut hash);
        FuncHash(hash.finish())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Param {
    pub span: Span,
    pub ty_binding: Span,
    pub ident: Ident,
    pub ty: TyId,
}
