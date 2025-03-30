use super::{Ty, TyVar, TypeKey};
use crate::ir::ctx::Ctx;
use crate::ir::ident::{Ident, IdentId};
use indexmap::IndexMap;
use pebblec_parse::diagnostic::{Diag, Msg};
use pebblec_parse::lex::buffer::Span;
use std::collections::HashMap;
use std::panic::Location;

#[derive(Debug, PartialEq, Eq)]
pub struct SymbolTable<T> {
    table: HashMap<IdentId, Vec<(Ident, T)>>,
}

impl<T> Default for SymbolTable<T> {
    fn default() -> Self {
        Self {
            table: HashMap::default(),
        }
    }
}

impl<T> SymbolTable<T> {
    #[track_caller]
    pub fn register(&mut self, ident: Ident, data: T) {
        self.table.entry(ident.id).or_default().push((ident, data));
    }

    pub fn symbol(&self, ident: IdentId) -> Option<&(Ident, T)> {
        self.table.get(&ident).and_then(|idents| idents.last())
    }

    pub fn iter(&self) -> impl Iterator<Item = (&T, &Ident)> {
        self.table
            .iter()
            .flat_map(|(_, idents)| idents.iter().map(|(ident, var)| (var, ident)))
    }
}

#[derive(Debug, Default)]
pub struct InferCtx {
    key: TypeKey,
    tables: Vec<SymbolTable<TyVar>>,
    constraints: IndexMap<TyVar, (Ident, Vec<Cnst>)>,
    var_index: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Cnst {
    pub kind: CnstKind,
    pub src: Span,
    pub var: Span,
    pub loc: &'static Location<'static>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CnstKind {
    Ty(Ty),
    Integral(Integral),
    Equate(TyVar),
}

const INVALID: &'static str = "invalid ty var";

impl InferCtx {
    pub fn into_key(self) -> TypeKey {
        self.key
    }

    pub fn in_scope<'a, R>(
        &mut self,
        ctx: &mut Ctx<'a>,
        f: impl FnOnce(&mut Ctx<'a>, &mut Self) -> Result<R, Diag>,
    ) -> Result<R, Diag> {
        self.tables.push(SymbolTable::default());
        let result = f(ctx, self);
        let unify_result = self.unify_top_scope(ctx);
        match result {
            Ok(inner) => unify_result.map(|_| inner),
            Err(diag) => match unify_result {
                Ok(_) => Err(diag),
                Err(unify_err) => Err(diag.join(unify_err)),
            },
        }
    }

    #[track_caller]
    pub fn new_var(&mut self, ident: Ident) -> TyVar {
        let ty_var = self.init_var(ident);
        if self.tables.is_empty() {
            self.tables.push(SymbolTable::default());
        }

        match self.tables.last_mut() {
            Some(scope) => scope.register(ident, ty_var),
            None => unreachable!(),
        }

        ty_var
    }

    #[track_caller]
    pub fn new_var_deferred<'a, R>(
        &'a mut self,
        ident: Ident,
        f: impl FnOnce(&mut Self, TyVar) -> R,
    ) -> R {
        let ty_var = self.init_var(ident);
        let result = f(self, ty_var);

        if self.tables.is_empty() {
            self.tables.push(SymbolTable::default());
        }

        match self.tables.last_mut() {
            Some(scope) => scope.register(ident, ty_var),
            None => unreachable!(),
        }

        result
    }

    fn init_var(&mut self, ident: Ident) -> TyVar {
        let idx = self.var_index;
        self.var_index += 1;
        self.constraints.insert(TyVar(idx), (ident, Vec::new()));
        TyVar(idx)
    }

    pub fn var_meta(&self, ident: IdentId) -> Option<&(Ident, TyVar)> {
        self.tables.iter().rev().find_map(|t| t.symbol(ident))
    }

    pub fn var(&self, ident: IdentId) -> Option<TyVar> {
        self.var_meta(ident).map(|(_, var)| *var)
    }

    #[track_caller]
    pub fn var_ident<'a>(&self, ctx: &'a Ctx<'a>, var: TyVar) -> &'a str {
        self.constraints
            .get(&var)
            .map(|(ident, _)| ctx.expect_ident(ident.id))
            .expect(INVALID)
    }

    #[track_caller]
    pub fn var_span<'a>(&self, var: TyVar) -> Span {
        self.constraints
            .get(&var)
            .map(|(ident, _)| ident.span)
            .expect(INVALID)
    }

    // TODO: if there are any conflicting absolute types, say from a parameter binding or a struct
    // definition, then this will return none and cause spurious type errors
    pub fn guess_var_ty(&self, ctx: &Ctx, var: TyVar) -> Option<Ty> {
        let cnsts = self.constraints.get(&var).expect(INVALID);
        self.unify_constraints(ctx, var, &cnsts.1).ok()
    }

    pub fn is_var_absolute(&self, var: TyVar) -> bool {
        let cnsts = self.constraints.get(&var).expect(INVALID);
        cnsts.1.iter().any(|c| matches!(c.kind, CnstKind::Ty(_)))
    }

    pub fn is_var_integral_int(&self, var: TyVar) -> bool {
        let cnsts = self.constraints.get(&var).expect(INVALID);
        cnsts
            .1
            .iter()
            .any(|c| matches!(c.kind, CnstKind::Integral(Integral::Int)))
    }

    pub fn is_var_integral_float(&self, var: TyVar) -> bool {
        let cnsts = self.constraints.get(&var).expect(INVALID);
        cnsts
            .1
            .iter()
            .any(|c| matches!(c.kind, CnstKind::Integral(Integral::Float)))
    }

    #[track_caller]
    pub fn integral(&mut self, integral: Integral, var: TyVar, src: Span) {
        let span = self.var_span(var);
        let (_, cnsts) = self.constraints.get_mut(&var).expect(INVALID);
        cnsts.push(Cnst {
            loc: std::panic::Location::caller(),
            kind: CnstKind::Integral(integral),
            var: span,
            src,
        })
    }

    #[track_caller]
    pub fn eq(&mut self, var: TyVar, ty: Ty, src: Span) {
        let span = self.var_span(var);
        let (_, cnsts) = self.constraints.get_mut(&var).expect(INVALID);
        cnsts.push(Cnst {
            loc: std::panic::Location::caller(),
            kind: CnstKind::Ty(ty),
            var: span,
            src,
        })
    }

    #[track_caller]
    pub fn var_eq(&mut self, ctx: &Ctx, var: TyVar, other: TyVar) {
        if let Some(ty) = self.guess_var_ty(ctx, other) {
            if self.is_var_absolute(other) {
                self.eq(var, ty, self.var_span(other));
                return;
            }
        }

        if let Some(ty) = self.guess_var_ty(ctx, var) {
            if self.is_var_absolute(var) {
                self.eq(other, ty, self.var_span(var));
                return;
            }
        }

        let span = self.var_span(var);
        let other_span = self.var_span(other);
        let (_, cnsts) = self.constraints.get_mut(&var).expect(INVALID);
        cnsts.push(Cnst {
            loc: std::panic::Location::caller(),
            kind: CnstKind::Equate(other),
            var: span,
            src: other_span,
        })
    }

    pub fn unify_top_scope<'a>(&mut self, ctx: &Ctx<'a>) -> Result<(), Diag> {
        let mut errors = Vec::new();

        let top_scope = self.tables.pop();
        let map = top_scope
            .iter()
            .flat_map(|t| t.iter())
            .collect::<HashMap<&TyVar, &Ident>>();
        for (var, constraints) in self.constraints.iter() {
            let Some(ident) = map.get(&var) else {
                continue;
            };

            match self.unify_constraints(ctx, *var, &constraints.1) {
                Ok(ty) => {
                    self.key.insert(**ident, ty);
                }
                Err(diag) => {
                    errors.push(diag);
                }
            }
        }

        if !errors.is_empty() {
            Err(Diag::bundle(errors))
        } else {
            Ok(())
        }
    }

    // TODO: need better error reporting. this will not cut it
    fn unify_constraints<'a>(
        &self,
        ctx: &Ctx<'a>,
        var: TyVar,
        constraints: &[Cnst],
    ) -> Result<Ty, Diag> {
        let mut integral = None;
        let mut abs = None;

        let mut constraints = constraints.to_vec();
        self.resolve_equates(ctx, &mut constraints, &mut vec![var]);

        for c in constraints.iter() {
            match &c.kind {
                CnstKind::Ty(ty) => {
                    if abs.is_some_and(|(abs, _): (Ty, _)| !ty.equiv(*abs.0)) {
                        let ty_str = abs.unwrap().0.to_string(ctx);
                        let other = ty.to_string(ctx);

                        // TODO: better error reporting, describe where the value is used
                        //
                        // in some cases, this doesn't even show where it is used.
                        return Err(ctx
                            .report_error(
                                c.var,
                                format!(
                                    "value of type `{}` cannot be coerced into a `{}`",
                                    ty_str, other
                                ),
                            )
                            //.msg(Msg::note(
                            //    abs.unwrap().1,
                            //    format!("but this is of type `{}`", ty_str),
                            //))
                            .msg(Msg::info(&ctx.source_map, c.src, c.loc.to_string())));
                    }

                    if abs.is_none() {
                        abs = Some((*ty, c.src));
                    }
                }
                CnstKind::Integral(int) => {
                    integral = Some((c.src, *int));
                }
                CnstKind::Equate(_) => unreachable!(),
            }
        }

        if let Some((abs, _)) = abs {
            if let Some((span, integral)) = integral {
                if (matches!(integral, Integral::Int) && !abs.is_int())
                    && (matches!(integral, Integral::Float) && !abs.is_float())
                    && (matches!(integral, Integral::Int) && abs.is_ref())
                {
                    return Err(ctx.mismatch(self.var_span(var), "an integer", abs).msg(
                        Msg::note(
                            &ctx.source_map,
                            span,
                            format!("because `{}` is used here", self.var_ident(ctx, var)),
                        ),
                    ));
                }
            }

            Ok(abs)
        } else {
            if let Some((_, integral)) = integral {
                Ok(match integral {
                    Integral::Int => Ty::ISIZE,
                    Integral::Float => Ty::FSIZE,
                })
            } else {
                Err(ctx.report_error(
                    self.var_span(var),
                    format!("could not infer type of `{}`", self.var_ident(ctx, var)),
                ))
            }
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
                                .expect(INVALID)
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Integral {
    Int,
    Float,
}
