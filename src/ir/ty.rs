use super::ctx::Ctx;
use super::ident::IdentId;
use super::strukt::Layout;
use crate::lex::buffer::{Span, TokenQuery};
use std::collections::HashMap;

/// Primitive type.
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

    pub fn size(&self) -> usize {
        match self {
            Self::Unit => 0,
            Self::Int(kind) => kind.size(),
        }
    }

    pub fn layout(&self) -> Layout {
        match self {
            Self::Unit => Layout::new(0, 1),
            Self::Int(kind) => kind.layout(),
        }
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

impl IntKind {
    pub fn size(&self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 => 4,
            Self::I64 | Self::U64 => 8,
        }
    }

    pub fn layout(&self) -> Layout {
        match self {
            Self::I8 | Self::U8 => Layout::splat(1),
            Self::I16 | Self::U16 => Layout::splat(2),
            Self::I32 | Self::U32 => Layout::splat(4),
            Self::I64 | Self::U64 => Layout::splat(8),
        }
    }
}

/// Unifies primitive and user types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FullTy {
    Ty(Ty),
    Struct(IdentId),
}

impl FullTy {
    pub fn is_unit(&self) -> bool {
        self.is_ty_and(|ty| ty.is_unit())
    }

    pub fn is_int(&self) -> bool {
        self.is_ty_and(|ty| ty.is_int())
    }

    pub fn is_ty_and(&self, f: impl FnOnce(&Ty) -> bool) -> bool {
        match self {
            Self::Ty(ty) => f(ty),
            Self::Struct(_) => false,
        }
    }

    pub fn is_struct_and(&self, f: impl FnOnce(&IdentId) -> bool) -> bool {
        match self {
            Self::Struct(strukt) => f(strukt),
            Self::Ty(_) => false,
        }
    }

    #[track_caller]
    pub fn expect_ty(&self) -> Ty {
        match self {
            Self::Ty(ty) => *ty,
            Self::Struct(_) => panic!("expected ty"),
        }
    }

    #[track_caller]
    pub fn expect_struct(&self) -> IdentId {
        match self {
            Self::Struct(s) => *s,
            Self::Ty(_) => panic!("expected struct"),
        }
    }

    pub fn as_str<'a>(&'a self, ctx: &'a Ctx<'a>) -> &'a str {
        match self {
            Self::Ty(ty) => ty.as_str(),
            Self::Struct(s) => ctx.expect_ident(*s),
        }
    }
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
    Struct(IdentId),
}

impl ConstraintKind {
    pub fn full(ty: FullTy) -> Self {
        match ty {
            FullTy::Ty(ty) => Self::Abs(ty),
            FullTy::Struct(s) => Self::Struct(s),
        }
    }

    pub fn hint_satisfies(&self, ty: FullTy) -> Option<bool> {
        match self {
            Self::Arch(arch) => Some(ty.is_ty_and(|ty| arch.satisfies(*ty))),
            Self::Abs(abs) => Some(ty.is_ty_and(|ty| abs == ty)),
            Self::Struct(strukt) => Some(ty.is_struct_and(|s| s == strukt)),
            Self::Equate(_) => None,
        }
    }

    pub fn is_int(&self) -> Option<bool> {
        match self {
            Self::Arch(arch) => Some(matches!(arch, Arch::Int)),
            Self::Abs(abs) => Some(abs.is_int()),
            Self::Struct(_) => Some(false),
            Self::Equate(_) => None,
        }
    }
}

#[derive(Debug)]
pub enum TyErr {
    NotEnoughInfo(Span, TyVar),
    Arch(Span, Arch, Ty),
    Abs(Span),
    Struct(IdentId),
}

impl Constraint {
    pub fn unify(
        ty_ctx: &TyCtx,
        ctx: &Ctx,
        var: TyVar,
        constraints: &[Constraint],
    ) -> Result<FullTy, TyErr> {
        let mut archs = Vec::with_capacity(constraints.len());
        let mut abs = None;
        let mut strukt = None;
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
                ConstraintKind::Struct(s) => {
                    if strukt.is_some_and(|ident| s != ident) {
                        return Err(TyErr::Struct(s));
                    }

                    strukt = Some(s);
                }
                _ => unreachable!(),
            }
        }

        if let Some(abs) = abs {
            if strukt.is_some() {
                return Err(TyErr::Struct(strukt.unwrap()));
            }

            for (span, arch) in archs.iter() {
                if !arch.satisfies(abs) {
                    return Err(TyErr::Arch(*span, *arch, abs));
                }
            }

            Ok(FullTy::Ty(abs))
        } else if let Some(strukt) = strukt {
            for (_span, _arch) in archs.iter() {
                panic!("type err");
                //return Err(TyErr::Arch(*span, *arch, abs));
            }

            Ok(FullTy::Struct(strukt))
        } else {
            panic!("not enough info");
            //ctx.exprs
            //    .iter()
            //    .find(|expr| expr.ty == var)
            //    .map(|expr| TyErr::NotEnoughInfo(expr.span, var))
            //    .unwrap_or_else(|| panic!("no one uses this ty var?"))
        }
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
    key: HashMap<IdentId, FullTy>,
}

impl TypeKey {
    #[track_caller]
    pub fn ty(&self, ident: IdentId) -> FullTy {
        *self.key.get(&ident).expect("invalid type var")
    }
}
