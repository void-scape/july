use std::collections::HashMap;

use super::ident::IdentId;
use super::ty::Ty;
use crate::lex::buffer::Span;

#[derive(Debug, Clone, Copy)]
pub struct Sig {
    pub span: Span,
    pub ident: IdentId,
    pub ty: Ty,
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
