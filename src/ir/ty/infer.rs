use super::store::{TyId, TyStore};
use super::{Ty, TyVar, TypeKey, VarHash};
use crate::diagnostic::{self, Diag};
use crate::ir::ctx::Ctx;
use crate::ir::ident::{Ident, IdentId};
use crate::ir::strukt::StructId;
use crate::ir::{Func, FuncHash};
use crate::lex::buffer::Span;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy)]
pub struct FnSpans {
    pub name: Span,
    pub sig: Span,
    pub block: Span,
}

#[derive(Debug, Default)]
pub struct InferCtx {
    key: TypeKey,
    vars: HashMap<VarHash, (TyVar, Span)>,
    constraints: HashMap<TyVar, (VarHash, Vec<Cnst>)>,
    var_index: usize,
    spans: Option<FnSpans>,
    hash: Option<FuncHash>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Cnst {
    pub kind: CnstKind,
    pub src: Span,
    pub var: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CnstKind {
    Ty(TyId),
    Integral,
    Equate(TyVar),
}

impl InferCtx {
    pub fn for_func(&mut self, func: &Func) {
        self.spans = Some(FnSpans {
            name: func.name_span,
            sig: func.sig.span,
            block: func.block.span,
        });
        self.hash = Some(func.hash());
    }

    pub fn finish<'a>(&mut self, ctx: &Ctx<'a>) -> Result<(), Vec<Diag<'a>>> {
        self.unify(ctx)
    }

    pub fn into_key(self) -> TypeKey {
        self.key
    }

    #[track_caller]
    pub fn new_var(&mut self, ident: Ident) -> TyVar {
        let idx = self.var_index;
        self.var_index += 1;
        self.vars
            .insert(self.var(ident.id), (TyVar(idx), ident.span));
        self.constraints
            .insert(TyVar(idx), (self.var(ident.id), Vec::new()));
        TyVar(idx)
    }

    pub fn get_var(&self, ident: IdentId) -> Option<TyVar> {
        self.vars.get(&self.var(ident)).map(|(var, _)| *var)
    }

    #[track_caller]
    pub fn var_ident<'a>(&self, ctx: &Ctx<'a>, var: TyVar) -> &'a str {
        ctx.expect_ident(self.constraints.get(&var).expect("invalid ty var").0.ident)
    }

    #[track_caller]
    pub fn var_span(&self, var: TyVar) -> Span {
        let hash = self.constraints.get(&var).expect("invalid ty var").0;
        self.vars.get(&hash).expect("invalid hash").1
    }

    pub fn guess_var_ty(&self, ctx: &Ctx, var: TyVar) -> Option<TyId> {
        self.constraints
            .get(&var)
            .map(|cnsts| self.unify_constraints(ctx, var, &cnsts.1).ok())?
    }

    pub fn integral(&mut self, var: TyVar, src: Span) {
        let span = self.var_span(var);
        let (_, cnsts) = self.constraints.get_mut(&var).expect("invalid ty var");
        cnsts.push(Cnst {
            kind: CnstKind::Integral,
            var: span,
            src,
        })
    }

    #[track_caller]
    pub fn eq(&mut self, var: TyVar, ty: TyId, src: Span) {
        let span = self.var_span(var);
        let (_, cnsts) = self.constraints.get_mut(&var).expect("invalid ty var");
        cnsts.push(Cnst {
            kind: CnstKind::Ty(ty),
            var: span,
            src,
        })
    }

    pub fn var_eq(&mut self, ctx: &Ctx, var: TyVar, other: TyVar) {
        if let Some(ty) = self.guess_var_ty(ctx, other) {
            self.eq(var, ty, self.var_span(other));
        } else if let Some(ty) = self.guess_var_ty(ctx, var) {
            self.eq(other, ty, self.var_span(var));
        } else {
            let span = self.var_span(var);
            let other_span = self.var_span(other);
            let (_, cnsts) = self.constraints.get_mut(&var).expect("invalid ty var");
            cnsts.push(Cnst {
                kind: CnstKind::Equate(other),
                var: span,
                src: other_span,
            })
        }
    }

    #[track_caller]
    fn fn_hash(&self) -> FuncHash {
        self.hash
            .expect("called `InferCtx::fn_hash` without first calling `InferCtx::for_func`")
    }

    #[track_caller]
    pub fn fn_spans(&self) -> FnSpans {
        self.spans
            .expect("called `InferCtx::fn_spans` without first calling `InferCtx::for_func`")
    }

    #[track_caller]
    pub fn fn_name(&self) -> Span {
        self.fn_spans().name
    }

    #[track_caller]
    pub fn fn_sig(&self) -> Span {
        self.fn_spans().sig
    }

    #[track_caller]
    pub fn fn_block(&self) -> Span {
        self.fn_spans().block
    }

    fn unify<'a>(&mut self, ctx: &Ctx<'a>) -> Result<(), Vec<Diag<'a>>> {
        let mut errors = Vec::new();

        let map = self
            .vars
            .iter()
            .map(|(hash, (var, span))| (var, (hash, span)))
            .collect::<HashMap<&TyVar, (&VarHash, &Span)>>();
        for (var, constraints) in self.constraints.iter() {
            let (ident, _) = map.get(&var).unwrap();
            match self.unify_constraints(ctx, *var, &constraints.1) {
                Ok(ty) => {
                    self.key.insert(**ident, ty);
                }
                Err(diag) => {
                    errors.push(diag);
                }
            }
        }

        self.vars.clear();
        self.constraints.clear();

        if !errors.is_empty() {
            Err(errors)
        } else {
            Ok(())
        }
    }

    #[track_caller]
    fn unify_constraints<'a>(
        &self,
        ctx: &Ctx<'a>,
        var: TyVar,
        constraints: &[Cnst],
    ) -> Result<TyId, Diag<'a>> {
        let mut integral = false;
        let mut abs = None;

        let mut constraints = constraints.to_vec();
        self.resolve_equates(ctx, &mut constraints, &mut vec![var]);

        for c in constraints.iter() {
            match &c.kind {
                CnstKind::Ty(ty) => {
                    if abs.is_some_and(|abs| abs != *ty) {
                        panic!("invalid abosulte types");
                        //return Err(TyErr::Abs(c.span));
                    }

                    abs = Some(*ty);
                }
                CnstKind::Integral => {
                    integral = true;
                }
                CnstKind::Equate(_) => unreachable!(),
            }
        }

        if let Some(abs) = abs {
            if integral && !ctx.tys.ty(abs).is_int() {
                panic!("expected integral");
            }

            Ok(abs)
        } else {
            Err(ctx.error(
                "inference error",
                self.var_span(var),
                format!("could not infer type of `{}`", self.var_ident(ctx, var)),
            ))
        }
    }

    fn resolve_equates(&self, ctx: &Ctx, constraints: &mut Vec<Cnst>, resolved: &mut Vec<TyVar>) {
        let mut new_constraints = Vec::<Cnst>::new();

        *constraints = constraints
            .iter()
            .cloned()
            .flat_map(|c| match c.kind {
                CnstKind::Equate(other) => {
                    if !resolved.contains(&other) {
                        resolved.push(other);
                        new_constraints.extend(
                            self.constraints
                                .get(&other)
                                .expect("invalid ty var")
                                .1
                                .iter()
                                .cloned()
                                .filter(|c| {
                                    if let CnstKind::Equate(other) = c.kind {
                                        !resolved.contains(&other)
                                    } else {
                                        true
                                    }
                                })
                                .clone(),
                        )
                    }

                    None
                }
                _ => Some(c),
            })
            .collect::<Vec<_>>();

        if !new_constraints.is_empty() {
            constraints.extend(new_constraints);
            self.resolve_equates(ctx, constraints, resolved);
        }
    }

    #[track_caller]
    fn var(&self, ident: IdentId) -> VarHash {
        let func = self.fn_hash();
        VarHash { ident, func }
    }
}
