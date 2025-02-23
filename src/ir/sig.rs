use super::ident::{Ident, IdentId};
use super::ty::store::TyId;
use super::FuncHash;
use crate::lex::buffer::Span;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Sig {
    pub span: Span,
    pub ident: IdentId,
    pub ty: TyId,
    pub params: Vec<Param>,
}

impl Sig {
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

#[derive(Debug, Default)]
pub struct SigStore {
    sigs: HashMap<IdentId, Sig>,
}

impl SigStore {
    pub fn store(&mut self, sig: Sig) {
        self.sigs.insert(sig.ident, sig);
    }

    pub fn get_sig(&self, ident: IdentId) -> Option<&Sig> {
        self.sigs.get(&ident)
    }
}
