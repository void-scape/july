use super::ident::IdentId;
use super::ty::store::TyId;
use crate::lex::buffer::Span;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Sig {
    pub span: Span,
    pub ident: IdentId,
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
