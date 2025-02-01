use crate::ir::prelude::Ty;
use constraint::*;
use std::collections::HashMap;

pub mod constraint;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TyVar(usize);

#[derive(Debug, Default)]
pub struct TyCtx<'a> {
    consts: HashMap<TyVar, Vec<Constraint>>,
    map: HashMap<&'a str, TyVar>,
}

impl<'a> TyCtx<'a> {
    pub fn register(&mut self, ident: &'a str) -> TyVar {
        let var = TyVar(self.map.len());
        self.map.insert(ident, var);
        var
    }

    pub fn constrain(&mut self, var: TyVar, constraint: Constraint) {
        self.consts.entry(var).or_default().push(constraint);
    }

    pub fn reconstruct(self) -> Result<Vec<(TyVar, Ty)>, ()> {
        self.consts
            .into_iter()
            .map(|(var, c)| Constraint::unify(c).map(|ty| (var, ty)))
            .collect::<Result<Vec<_>, ()>>()
    }
}
