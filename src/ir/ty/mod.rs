use self::store::TyId;
use super::ctx::Ctx;
use super::ident::{Ident, IdentId};
use super::mem::Layout;
use super::strukt::StructId;
use std::collections::HashMap;

pub mod infer;
pub mod store;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ty<'a> {
    Int(IntTy),
    Float(FloatTy),
    /// TODO: store struct in Ty itself?
    Struct(StructId),
    Ref(&'a Ty<'a>),
    Array(usize, &'a Ty<'a>),
    Bool,
    Str,
    Unit,
}

impl<'a> Ty<'a> {
    pub const PTR_SIZE: usize = 8;
    pub const FAT_PTR_SIZE: usize = 16;

    pub fn size(&self, ctx: &Ctx) -> usize {
        match self {
            Self::Unit => 0,
            Self::Bool => 1,
            Self::Ref(inner) => match inner {
                Ty::Str => Self::FAT_PTR_SIZE,
                _ => Self::PTR_SIZE,
            },
            Self::Int(int) => int.size(),
            Self::Float(float) => float.size(),
            Self::Str => panic!("size of str is unknown"),
            Self::Struct(id) => ctx.tys.struct_layout(*id).size,
            Self::Array(len, inner) => inner.size(ctx) * len,
        }
    }

    pub fn is_int(&self) -> bool {
        matches!(self, Self::Int(_))
    }

    pub fn is_float(&self) -> bool {
        matches!(self, Self::Float(_))
    }

    pub fn is_ref(&self) -> bool {
        matches!(self, Self::Ref(ty) if **ty != Ty::Str)
    }

    pub fn is_arr(&self) -> bool {
        matches!(self, Self::Array(_, _))
    }

    pub fn is_castable(&self) -> bool {
        match self {
            Self::Struct(_)
            | Self::Ref(&Self::Str)
            | Self::Str
            | Self::Array(_, _)
            | Self::Unit => false,
            Self::Int(_) | Self::Float(_) | Self::Bool | Self::Ref(_) => true,
        }
    }

    pub fn layout(&self, ctx: &Ctx) -> Layout {
        match self {
            Self::Unit => Layout::new(0, 1),
            Self::Bool => Layout::splat(1),
            Self::Int(int) => int.layout(),
            Self::Float(float) => float.layout(),
            Self::Ref(&Self::Str) => Layout::FAT_PTR,
            Self::Str | Self::Ref(_) => Layout::PTR,
            Self::Array(len, inner) => inner.layout(ctx).to_array(*len),
            Self::Struct(id) => ctx.tys.struct_layout(*id),
        }
    }

    pub fn is_sized(&self) -> bool {
        !matches!(self, Self::Str)
    }

    #[track_caller]
    pub fn expect_int(&self) -> IntTy {
        match self {
            Self::Int(ty) => *ty,
            Self::Bool => IntTy::BOOL,
            Self::Ref(_) => IntTy::PTR,
            _ => panic!("expected int, got {:?}", self),
        }
    }

    #[track_caller]
    pub fn expect_struct(&self) -> StructId {
        match self {
            Self::Struct(s) => *s,
            _ => panic!("expected struct"),
        }
    }

    pub fn to_string(&self, ctx: &Ctx) -> String {
        match self {
            Self::Unit => "()".to_string(),
            Self::Bool => "bool".to_string(),
            Self::Ref(inner) => format!("&{}", inner.to_string(ctx)),
            Self::Str => "str".to_string(),
            Self::Int(int) => int.as_str().to_string(),
            Self::Float(float) => float.as_str().to_string(),
            Self::Struct(s) => ctx.expect_ident(ctx.tys.strukt(*s).name.id).to_string(),
            Self::Array(len, inner) => format!("[{}] {}", len, inner.to_string(ctx)),
        }
    }

    pub fn peel_refs(&self) -> (&Ty<'a>, usize) {
        match self {
            Self::Ref(inner) => {
                let (ty, level) = inner.peel_refs();
                (ty, level + 1)
            }
            inner => (inner, 0),
        }
    }

    pub fn peel_one_ref(&self) -> &Ty {
        match self {
            Self::Ref(inner) => inner,
            inner => inner,
        }
    }

    pub fn ref_inner_ty(&self, ctx: &Ctx<'a>) -> Option<Result<TyId, ()>> {
        match self {
            Self::Ref(inner) => Some(ctx.tys.get_ty_id(inner).ok_or(())),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IntTy {
    sign: Sign,
    width: Width,
}

impl IntTy {
    pub const BOOL: Self = IntTy::new_8(Sign::U);
    pub const PTR: Self = IntTy::new_64(Sign::U);
    pub const ISIZE: Self = IntTy::new_64(Sign::I);
    pub const USIZE: Self = IntTy::new_64(Sign::U);

    pub const fn new(sign: Sign, width: Width) -> Self {
        Self { sign, width }
    }

    pub const fn new_8(sign: Sign) -> Self {
        Self::new(sign, Width::W8)
    }

    pub const fn new_16(sign: Sign) -> Self {
        Self::new(sign, Width::W16)
    }

    pub const fn new_32(sign: Sign) -> Self {
        Self::new(sign, Width::W32)
    }

    pub const fn new_64(sign: Sign) -> Self {
        Self::new(sign, Width::W64)
    }

    pub const fn size(&self) -> usize {
        self.width.bytes()
    }

    pub const fn layout(&self) -> Layout {
        Layout::splat(self.size())
    }

    pub const fn width(&self) -> Width {
        self.width
    }

    pub const fn sign(&self) -> Sign {
        self.sign
    }

    pub const fn as_str(&self) -> &'static str {
        match self.sign {
            Sign::I => match self.width {
                Width::W8 => "i8",
                Width::W16 => "i16",
                Width::W32 => "i32",
                Width::W64 => "i64",
            },
            Sign::U => match self.width {
                Width::W8 => "u8",
                Width::W16 => "u16",
                Width::W32 => "u32",
                Width::W64 => "u64",
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Sign {
    I,
    U,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Width {
    W8,
    W16,
    W32,
    W64,
}

impl Width {
    pub const BOOL: Self = Self::W8;
    pub const PTR: Self = Self::SIZE;
    pub const SIZE: Self = Self::W64;

    pub const fn bytes(&self) -> usize {
        match self {
            Self::W8 => 1,
            Self::W16 => 2,
            Self::W32 => 4,
            Self::W64 => 8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FloatTy {
    F32,
    F64,
}

impl FloatTy {
    pub const fn size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    pub const fn layout(&self) -> Layout {
        Layout::splat(self.size())
    }

    pub const fn width(&self) -> Width {
        match self {
            Self::F32 => Width::W32,
            Self::F64 => Width::W64,
        }
    }

    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TyVar(usize);

#[derive(Debug, Default)]
pub struct TypeKey {
    key: HashMap<IdentId, Vec<(Ident, TyId)>>,
}

impl TypeKey {
    pub fn insert(&mut self, ident: Ident, ty: TyId) {
        let entry = self.key.entry(ident.id).or_default();
        let elem = (ident, ty);
        if entry.contains(&elem) {
            assert!(entry
                .iter()
                .filter(|(i, _)| *i == ident)
                .all(|(_, t)| *t == ty))
        } else {
            entry.push(elem);
            entry.sort_unstable_by_key(|(ident, _)| ident.span.start);
        }
    }

    pub fn ident_set(&self, ident: IdentId) -> &[(Ident, TyId)] {
        self.key
            .get(&ident)
            .map(Vec::as_slice)
            .unwrap_or_else(|| &[])
    }

    pub fn ident_set_mut(&mut self, ident: IdentId) -> Option<&mut Vec<(Ident, TyId)>> {
        self.key.get_mut(&ident)
    }
}
