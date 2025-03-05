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
    pub linkage: Linkage<'a>,
}

impl Sig<'_> {
    pub fn hash(&self) -> FuncHash {
        let mut hash = DefaultHasher::new();
        <Sig as Hash>::hash(self, &mut hash);
        FuncHash(hash.finish())
    }

    pub fn is_external(&self) -> bool {
        self.linkage.is_external()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Linkage<'a> {
    Local,
    External { link: &'a str },
}

impl Linkage<'_> {
    pub fn is_external(&self) -> bool {
        matches!(self, Self::External { .. })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Param {
    pub span: Span,
    pub ty_binding: Span,
    pub ident: Ident,
    pub ty: TyId,
}
