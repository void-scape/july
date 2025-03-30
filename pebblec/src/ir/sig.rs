use super::FuncHash;
use super::ident::{Ident, IdentId};
use super::ty::Ty;
use pebblec_parse::lex::buffer::Span;
use std::hash::{DefaultHasher, Hash, Hasher};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Sig<'a> {
    pub span: Span,
    pub ident: IdentId,
    pub ty: Ty,
    pub params: &'a [Param],
    pub linkage: Linkage<'a>,
}

impl Sig<'_> {
    pub fn hash(&self) -> FuncHash {
        let mut hash = deterministic_hash::DeterministicHasher::new(DefaultHasher::new());
        <Sig as Hash>::hash(self, &mut hash);
        FuncHash(hash.finish())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Linkage<'a> {
    Local,
    External { link: &'a str },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Param {
    Slf(Ident),
    SlfRef(Ident),
    Named { span: Span, ident: Ident, ty: Ty },
}

impl Param {
    pub fn span(&self) -> Span {
        match self {
            Self::Slf(ident) => ident.span,
            Self::SlfRef(ident) => ident.span,
            Self::Named { span, .. } => *span,
        }
    }
}
