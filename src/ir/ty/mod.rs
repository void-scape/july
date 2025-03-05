use self::store::TyId;
use super::ctx::Ctx;
use super::ident::IdentId;
use super::mem::Layout;
use super::strukt::StructId;
use super::FuncHash;
use std::collections::HashMap;

pub mod infer;
pub mod store;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ty<'a> {
    Int(IntTy),
    Struct(StructId),
    Ref(&'a Ty<'a>),
    Bool,
    Str,
    Unit,
}

impl Ty<'_> {
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
            Self::Str => panic!("size of str is unknown"),
            Self::Struct(id) => ctx.tys.struct_layout(*id).size,
        }
    }

    pub fn is_int(&self) -> bool {
        matches!(self, Self::Int(_))
    }

    pub fn is_ref(&self) -> bool {
        matches!(self, Self::Ref(_))
    }

    #[track_caller]
    pub fn expect_int(&self) -> IntTy {
        match self {
            Self::Int(ty) => *ty,
            Self::Bool => IntTy::BOOL,
            Self::Ref(_) => IntTy::PTR,
            ty => panic!("expected int, got {:?}", ty),
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
            Self::Struct(s) => ctx.expect_ident(ctx.tys.strukt(*s).name.id).to_string(),
        }
    }

    pub fn peel_refs(&self) -> (&Ty, usize) {
        match self {
            Self::Ref(inner) => {
                let (ty, level) = inner.peel_refs();
                (ty, level + 1)
            }
            inner => (inner, 0),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IntTy {
    pub sign: Sign,
    pub width: IWidth,
}

impl IntTy {
    pub const BOOL: Self = IntTy::new_8(Sign::U);
    pub const PTR: Self = IntTy::new_64(Sign::I);

    pub const fn new(sign: Sign, width: IWidth) -> Self {
        Self { sign, width }
    }

    pub const fn new_8(sign: Sign) -> Self {
        Self::new(sign, IWidth::W8)
    }

    pub const fn new_16(sign: Sign) -> Self {
        Self::new(sign, IWidth::W16)
    }

    pub const fn new_32(sign: Sign) -> Self {
        Self::new(sign, IWidth::W32)
    }

    pub const fn new_64(sign: Sign) -> Self {
        Self::new(sign, IWidth::W64)
    }

    pub const fn size(&self) -> usize {
        self.width.bytes()
    }

    pub const fn layout(&self) -> Layout {
        Layout::splat(self.size())
    }

    pub const fn as_str(&self) -> &'static str {
        match self.sign {
            Sign::I => match self.width {
                IWidth::W8 => "i8",
                IWidth::W16 => "i16",
                IWidth::W32 => "i32",
                IWidth::W64 => "i64",
            },
            Sign::U => match self.width {
                IWidth::W8 => "u8",
                IWidth::W16 => "u16",
                IWidth::W32 => "u32",
                IWidth::W64 => "u64",
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
pub enum IWidth {
    W8,
    W16,
    W32,
    W64,
}

impl IWidth {
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
pub struct TyVar(usize);

#[derive(Debug, Default)]
pub struct TypeKey {
    key: HashMap<VarHash, TyId>,
}

// TODO: rename to VarPath
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VarHash {
    ident: IdentId,
    func: FuncHash,
}

impl TypeKey {
    pub fn insert(&mut self, var: VarHash, ty: TyId) {
        self.key.insert(var, ty);
    }

    #[track_caller]
    pub fn ty(&self, ident: IdentId, func: FuncHash) -> TyId {
        *self
            .key
            .get(&VarHash { ident, func })
            .expect("variable is not keyed")
    }
}
