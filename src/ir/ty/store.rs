use super::{FloatTy, IntTy, Sign, VarHash};
use crate::diagnostic::Diag;
use crate::ir::ctx::Ctx;
use crate::ir::ident::IdentId;
use crate::ir::mem::Layout;
use crate::ir::strukt::{FieldMap, Struct, StructId};
use crate::ir::ty::Ty;
use crate::ir::Const;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TyPath<'a> {
    Global(&'a [IdentId]),
    Var(VarHash),
}

pub trait IntoTyPath<'a> {
    fn into_ty_path(&self) -> TyPath<'a>;
}

impl<'a> IntoTyPath<'a> for TyPath<'a> {
    fn into_ty_path(&self) -> TyPath<'a> {
        *self
    }
}

impl<'a> IntoTyPath<'a> for &'a [IdentId] {
    fn into_ty_path(&self) -> TyPath<'a> {
        TyPath::Global(self)
    }
}

impl<'a, const N: usize> IntoTyPath<'a> for &'a [IdentId; N] {
    fn into_ty_path(&self) -> TyPath<'a> {
        TyPath::Global(&self[..])
    }
}

impl IntoTyPath<'static> for VarHash {
    fn into_ty_path(&self) -> TyPath<'static> {
        TyPath::Var(*self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TyId(usize);

impl TyId {
    pub const UNIT: Self = TyId(0);
    pub const BOOL: Self = TyId(1);
    pub const STR_LIT: Self = TyId(2);
}

#[derive(Debug, Default, Clone)]
pub struct TyStore<'a> {
    ty_map: HashMap<Ty<'a>, TyId>,
    tys: Vec<Ty<'a>>,

    const_map: HashMap<IdentId, &'a Const<'a>>,

    struct_map: HashMap<IdentId, StructId>,
    struct_ty_map: HashMap<StructId, TyId>,
    structs: Vec<Struct>,
    fields: HashMap<StructId, FieldMap>,

    layouts: HashMap<TyId, Layout>,
}

impl<'a> TyStore<'a> {
    pub fn new() -> Self {
        let mut slf = Self::default();

        slf.store_ty(Ty::Unit);
        slf.store_ty(Ty::Bool);
        slf.store_ty(Ty::Ref(&Ty::Str));
        slf.store_ty(Ty::Float(FloatTy::F32));
        slf.store_ty(Ty::Float(FloatTy::F64));
        slf.store_ty(Ty::Int(IntTy::new_8(Sign::U)));
        slf.store_ty(Ty::Int(IntTy::new_16(Sign::U)));
        slf.store_ty(Ty::Int(IntTy::new_32(Sign::U)));
        slf.store_ty(Ty::Int(IntTy::new_64(Sign::U)));
        slf.store_ty(Ty::Int(IntTy::new_8(Sign::I)));
        slf.store_ty(Ty::Int(IntTy::new_16(Sign::I)));
        slf.store_ty(Ty::Int(IntTy::new_32(Sign::I)));
        slf.store_ty(Ty::Int(IntTy::new_64(Sign::I)));

        slf
    }

    pub fn store_ty(&mut self, ty: Ty<'a>) -> TyId {
        let idx = self.tys.len();
        assert!(self.ty_map.get(&ty).is_none_or(|old| self.tys[old.0] == ty));
        self.ty_map.insert(ty, TyId(idx));
        self.tys.push(ty);
        TyId(idx)
    }

    pub fn store_struct(&mut self, strukt: Struct) -> StructId {
        let idx = self.structs.len();
        let ty_id = self.store_ty(Ty::Struct(StructId(idx)));
        self.struct_map.insert(strukt.name.id, StructId(idx));
        self.struct_ty_map.insert(StructId(idx), ty_id);
        self.structs.push(strukt);
        StructId(idx)
    }

    pub fn structs(&self) -> &[Struct] {
        &self.structs
    }

    pub fn store_const(&mut self, konst: &'a Const<'a>) {
        self.const_map.insert(konst.name.id, konst);
    }

    pub fn bool(&self) -> TyId {
        TyId::BOOL
    }

    pub fn str_lit(&self) -> TyId {
        TyId::STR_LIT
    }

    pub fn unit(&self) -> TyId {
        TyId::UNIT
    }

    pub fn is_unit(&self, ty: TyId) -> bool {
        ty == self.unit()
    }

    /// Used during the construction of types, where [`Ty`]s are not easily accessible.
    pub fn is_builtin(&self, ident: &str) -> bool {
        match ident {
            "u8" | "u16" | "u32" | "u64" | "i8" | "i16" | "i32" | "i64" | "f32" | "f64"
            | "bool" | "str" => true,
            _ => false,
        }
    }

    #[track_caller]
    pub fn ty(&self, ty_id: TyId) -> Ty<'a> {
        self.tys.get(ty_id.0).copied().expect("invalid type id")
    }

    pub fn ty_id(&mut self, ty: &Ty<'a>) -> TyId {
        if let Some(id) = self.ty_map.get(ty) {
            *id
        } else {
            self.store_ty(*ty)
        }
    }

    pub fn get_ty_id(&self, ty: &Ty<'a>) -> Option<TyId> {
        self.ty_map.get(ty).copied()
    }

    #[track_caller]
    pub fn builtin(&self, ty: Ty<'a>) -> TyId {
        self.get_ty_id(&ty).expect("invalid builtin type")
    }

    pub fn get_const(&self, id: IdentId) -> Option<&Const> {
        self.const_map.get(&id).copied()
    }

    pub fn consts(&self) -> impl Iterator<Item = &&'a Const<'a>> {
        self.const_map.values()
    }

    #[track_caller]
    pub fn strukt(&self, id: StructId) -> &Struct {
        self.structs.get(id.0).expect("invalid struct id")
    }

    #[track_caller]
    pub fn struct_ty_id(&self, struct_id: StructId) -> TyId {
        self.struct_ty_map
            .get(&struct_id)
            .copied()
            .expect("invalid struct id")
    }

    pub fn struct_id(&self, ident: IdentId) -> Option<StructId> {
        self.struct_map.get(&ident).copied()
    }

    #[track_caller]
    pub fn expect_struct_id(&self, ident: IdentId) -> StructId {
        self.struct_id(ident).expect("invalid struct ident")
    }

    #[track_caller]
    pub fn fields(&self, struct_id: StructId) -> &FieldMap {
        self.fields.get(&struct_id).expect("invalid struct id")
    }

    #[track_caller]
    pub fn layout(&self, ty_id: TyId) -> Layout {
        *self.layouts.get(&ty_id).expect("invalid type id")
    }

    #[track_caller]
    pub fn struct_layout(&self, struct_id: StructId) -> Layout {
        let ty_id = self
            .struct_ty_map
            .get(&struct_id)
            .expect("invalid struct id");
        self.layout(*ty_id)
    }

    pub fn build_layouts(&mut self, ctx: &Ctx<'a>) -> Result<(), Diag<'a>> {
        for (i, strukt) in self.structs.iter().enumerate() {
            let ty_id = self.struct_ty_map.get(&StructId(i)).unwrap();
            if !self.layouts.contains_key(&ty_id) {
                let id = self.expect_struct_id(strukt.name.id);
                Self::layout_struct(
                    ctx,
                    &self.structs,
                    &self.struct_ty_map,
                    &mut self.layouts,
                    &mut self.fields,
                    id,
                    None,
                )?;
            }
        }

        Ok(())
    }

    fn layout_struct(
        ctx: &Ctx<'a>,
        buf: &[Struct],
        struct_ty_map: &HashMap<StructId, TyId>,
        layouts: &mut HashMap<TyId, Layout>,
        offsets: &mut HashMap<StructId, FieldMap>,
        strukt: StructId,
        prev: Option<StructId>,
    ) -> Result<(), Diag<'a>> {
        let mut errors = Vec::new();
        let struct_id = strukt;
        let strukt = &buf[strukt.0];

        if prev.is_some_and(|id| {
            strukt.fields.iter().any(|f| match ctx.tys.ty(f.ty) {
                Ty::Struct(s) => s == id,
                _ => false,
            })
        }) {
            unreachable!("checked during struct sizing");
        }

        let mut unresolved_layouts = Vec::new();
        for field in strukt.fields.iter() {
            match ctx.tys.ty(field.ty) {
                Ty::Struct(s) => {
                    unresolved_layouts.push(s);
                }
                _ => {}
            }
        }

        for id in unresolved_layouts.iter() {
            Self::layout_struct(
                ctx,
                buf,
                struct_ty_map,
                layouts,
                offsets,
                *id,
                Some(struct_id),
            )?
        }

        let strukt = &buf[struct_id.0];
        let mut struct_layouts = Vec::with_capacity(strukt.fields.len());
        for field in strukt.fields.iter() {
            let field_ty = ctx.tys.ty(field.ty);
            if !field_ty.is_sized() {
                errors.push(ctx.report_error(field.span, "struct fields must be sized"));
            }

            match field_ty {
                Ty::Struct(id) => {
                    let ty_id = struct_ty_map.get(&id).unwrap();
                    let layout = layouts.get(ty_id).unwrap();
                    struct_layouts.push(*layout);
                }
                _ => {
                    struct_layouts.push(field_ty.layout(ctx));
                }
            }
        }

        let mut alignment = 1;
        for layout in struct_layouts.iter() {
            if layout.alignment > alignment {
                alignment = layout.alignment;
            }
        }

        let mut struct_offsets = HashMap::new();
        let mut byte = 0;
        for (layout, field) in struct_layouts.iter().zip(strukt.fields.iter()) {
            while byte % layout.alignment != 0 {
                byte += 1;
            }

            struct_offsets.insert(field.name.id, (field.ty, byte as i32));
            byte += layout.size;
        }

        while byte % alignment != 0 {
            byte += 1;
        }

        let ty_id = struct_ty_map.get(&struct_id).unwrap();
        layouts.insert(*ty_id, Layout::new(byte, alignment));
        offsets.insert(
            struct_id,
            FieldMap {
                fields: struct_offsets,
            },
        );

        if !errors.is_empty() {
            Err(Diag::bundle(errors))
        } else {
            Ok(())
        }
    }
}
