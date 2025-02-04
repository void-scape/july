use super::ctx::Ctx;
use super::ident::IdentId;
use crate::lex::buffer::Span;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ty {
    Unit,
    Int(IntKind),
}

impl Ty {
    pub fn is_unit(&self) -> bool {
        matches!(self, Ty::Unit)
    }

    pub fn is_int(&self) -> bool {
        matches!(self, Ty::Int(_))
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Unit => "()",
            Self::Int(kind) => match kind {
                IntKind::I8 => "i8",
                IntKind::I16 => "i16",
                IntKind::I32 => "i32",
                IntKind::I64 => "i64",
                IntKind::U8 => "u8",
                IntKind::U16 => "u16",
                IntKind::U32 => "u32",
                IntKind::U64 => "u64",
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntKind {
    I8,
    I16,
    I32,
    I64,

    U8,
    U16,
    U32,
    U64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TyVar(usize);

#[derive(Debug, Default)]
pub struct TyCtx {
    consts: Vec<Vec<Constraint>>,
    idents: HashMap<IdentId, TyVar>,
}

impl TyCtx {
    pub fn var(&mut self, ident: IdentId) -> TyVar {
        let idx = self.consts.len();
        self.idents.insert(ident, TyVar(idx));
        self.consts.push(Vec::new());
        TyVar(idx)
    }

    pub fn try_get_var(&self, ident: IdentId) -> Option<TyVar> {
        self.idents.get(&ident).copied()
    }

    #[track_caller]
    pub fn get_var(&self, ident: IdentId) -> TyVar {
        self.try_get_var(ident).expect("invalid ident")
    }

    #[track_caller]
    pub fn constrain(&mut self, ty_var: TyVar, constraint: Constraint) {
        self.consts
            .get_mut(ty_var.0)
            .expect("invalid ty var")
            .push(constraint);
    }

    pub fn resolve(&self, ctx: &Ctx) -> Result<TypeKey, Vec<TyErr>> {
        let map = self
            .idents
            .iter()
            .map(|(ident, var)| (*var, *ident))
            .collect::<HashMap<_, _>>();

        let mut key = HashMap::with_capacity(self.consts.len());
        let mut errs = Vec::new();
        for (i, c) in self.consts.iter().enumerate() {
            let var = TyVar(i);
            match Constraint::unify(self, ctx, var, c) {
                Ok(ty) => {
                    key.insert(*map.get(&var).unwrap(), ty);
                }
                Err(err) => errs.push(err),
            }
        }

        if !errs.is_empty() {
            Err(errs)
        } else {
            Ok(TypeKey { key })
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TyId(usize);

#[derive(Debug)]
pub struct TyRegistry<'a> {
    symbol_map: HashMap<&'a str, TyId>,
    ty_map: HashMap<TyId, Ty>,
    tys: Vec<&'a str>,
}

impl Default for TyRegistry<'_> {
    fn default() -> Self {
        let mut slf = Self {
            symbol_map: HashMap::default(),
            ty_map: HashMap::default(),
            tys: Vec::new(),
        };

        slf.register_ty("i8", Ty::Int(IntKind::I8));
        slf.register_ty("i16", Ty::Int(IntKind::I16));
        slf.register_ty("i32", Ty::Int(IntKind::I32));
        slf.register_ty("i64", Ty::Int(IntKind::I64));

        slf.register_ty("u8", Ty::Int(IntKind::U8));
        slf.register_ty("u16", Ty::Int(IntKind::U16));
        slf.register_ty("u32", Ty::Int(IntKind::U32));
        slf.register_ty("u64", Ty::Int(IntKind::U64));

        slf
    }
}

impl<'a> TyRegistry<'a> {
    pub fn ty_str(&self, ty: &'a str) -> Option<Ty> {
        self.symbol_map
            .get(ty)
            .map(|id| self.ty_map.get(id).copied())?
    }

    fn register_ty(&mut self, ty_str: &'a str, ty: Ty) {
        let id = TyId(self.tys.len());
        self.ty_map.insert(id, ty);
        self.tys.push(ty_str);
        self.symbol_map.insert(ty_str, id);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Constraint {
    pub span: Span,
    pub kind: ConstraintKind,
}

#[derive(Debug, Clone, Copy)]
pub enum ConstraintKind {
    Arch(Arch),
    Equate(TyVar),
    Abs(Ty),
}

impl ConstraintKind {
    pub fn hint_satisfies(&self, ty: Ty) -> Option<bool> {
        match self {
            Self::Arch(arch) => Some(arch.satisfies(ty)),
            Self::Abs(abs) => Some(*abs == ty),
            Self::Equate(_) => None,
        }
    }

    pub fn is_int(&self) -> Option<bool> {
        match self {
            Self::Arch(arch) => Some(matches!(arch, Arch::Int)),
            Self::Abs(abs) => Some(abs.is_int()),
            Self::Equate(_) => None,
        }
    }
}

#[derive(Debug)]
pub enum TyErr {
    NotEnoughInfo(Span, TyVar),
    Arch(Span, Arch, Ty),
    Abs(Span),
}

impl Constraint {
    pub fn unify(
        ty_ctx: &TyCtx,
        ctx: &Ctx,
        var: TyVar,
        constraints: &[Constraint],
    ) -> Result<Ty, TyErr> {
        let mut archs = Vec::with_capacity(constraints.len());
        let mut abs = None;
        //let mut equates = Vec::new();

        let mut constraints = constraints.to_vec();
        Self::resolve_equates(ty_ctx, ctx, &mut constraints, &mut vec![var]);

        for c in constraints.iter() {
            match c.kind {
                ConstraintKind::Abs(ty) => {
                    if abs.is_some_and(|abs| abs != ty) {
                        return Err(TyErr::Abs(c.span));
                    }

                    abs = Some(ty);
                }
                ConstraintKind::Arch(a) => archs.push((c.span, a)),
                _ => unreachable!(),
            }
        }

        let ty = abs.ok_or_else(|| {
            panic!("not enoug info")
            //ctx.exprs
            //    .iter()
            //    .find(|expr| expr.ty == var)
            //    .map(|expr| TyErr::NotEnoughInfo(expr.span, var))
            //    .unwrap_or_else(|| panic!("no one uses this ty var?"))
        })?;

        for (span, arch) in archs.iter() {
            if !arch.satisfies(ty) {
                return Err(TyErr::Arch(*span, *arch, ty));
            }
        }

        Ok(ty)
    }

    fn resolve_equates(
        ty_ctx: &TyCtx,
        ctx: &Ctx,
        constraints: &mut Vec<Constraint>,
        resolved: &mut Vec<TyVar>,
    ) {
        let mut new_constraints = Vec::<Constraint>::new();

        *constraints = constraints
            .iter()
            .copied()
            .flat_map(|c| match c.kind {
                ConstraintKind::Equate(other) => {
                    if !resolved.contains(&other) {
                        resolved.push(other);
                        new_constraints.extend(ty_ctx.consts[other.0].iter().filter(|c| {
                            if let ConstraintKind::Equate(other) = c.kind {
                                !resolved.contains(&other)
                            } else {
                                true
                            }
                        }))
                    }

                    None
                }
                _ => Some(c),
            })
            .collect::<Vec<_>>();

        if !new_constraints.is_empty() {
            constraints.extend(new_constraints);
            Self::resolve_equates(ty_ctx, ctx, constraints, resolved);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Arch {
    Int,
}

impl Arch {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Int => "int",
        }
    }

    pub fn satisfies(&self, ty: Ty) -> bool {
        match self {
            Arch::Int => ty.is_int(),
        }
    }
}

pub struct TypeKey {
    key: HashMap<IdentId, Ty>,
}

impl TypeKey {
    #[track_caller]
    pub fn ty(&self, ident: IdentId) -> Ty {
        *self.key.get(&ident).expect("invalid type var")
    }
}
