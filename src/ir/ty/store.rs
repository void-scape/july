use super::{IntTy, Sign, VarHash};
use crate::diagnostic::Diag;
use crate::ir::ctx::Ctx;
use crate::ir::ident::{IdentId, IdentStore};
use crate::ir::mem::Layout;
use crate::ir::strukt::{FieldMap, Struct, StructId};
use crate::ir::ty::Ty;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TyPath {
    Global(IdentId),
    Var(VarHash),
}

pub trait IntoTyPath {
    fn into_ty_path(&self) -> TyPath;
}

impl IntoTyPath for IdentId {
    fn into_ty_path(&self) -> TyPath {
        TyPath::Global(*self)
    }
}

impl IntoTyPath for VarHash {
    fn into_ty_path(&self) -> TyPath {
        TyPath::Var(*self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TyId(usize);

#[derive(Debug, Default, Clone)]
pub struct TyStore {
    ty_map: HashMap<TyPath, TyId>,
    tys: Vec<Ty>,

    struct_map: HashMap<IdentId, StructId>,
    struct_ty_map: HashMap<StructId, TyId>,
    structs: Vec<Struct>,
    fields: HashMap<StructId, FieldMap>,

    layouts: HashMap<TyId, Layout>,
}

impl TyStore {
    pub fn register_builtins(&mut self, idents: &mut IdentStore) {
        self.register_builtin(idents, "bool", Ty::Bool);
        self.register_builtin(idents, "u8", Ty::Int(IntTy::new_8(Sign::U)));
        self.register_builtin(idents, "u16", Ty::Int(IntTy::new_16(Sign::U)));
        self.register_builtin(idents, "u32", Ty::Int(IntTy::new_32(Sign::U)));
        self.register_builtin(idents, "u64", Ty::Int(IntTy::new_64(Sign::U)));
        self.register_builtin(idents, "i8", Ty::Int(IntTy::new_8(Sign::I)));
        self.register_builtin(idents, "i16", Ty::Int(IntTy::new_16(Sign::I)));
        self.register_builtin(idents, "i32", Ty::Int(IntTy::new_32(Sign::I)));
        self.register_builtin(idents, "i64", Ty::Int(IntTy::new_64(Sign::I)));
    }

    fn register_builtin(&mut self, idents: &mut IdentStore, str: &'static str, ty: Ty) {
        idents.store(str);
        let id = idents.get_id(str).unwrap();
        self.store_ty(id, ty);
    }

    pub fn store_ty<P: IntoTyPath>(&mut self, path: P, ty: Ty) -> TyId {
        let path = path.into_ty_path();
        let idx = self.tys.len();
        assert!(self
            .ty_map
            .get(&path)
            .is_none_or(|old| self.tys[old.0] == ty));
        self.ty_map.insert(path, TyId(idx));
        self.tys.push(ty);
        TyId(idx)
    }

    pub fn store_struct(&mut self, strukt: Struct) -> StructId {
        let idx = self.structs.len();
        let ty_id = self.store_ty(strukt.name.id, Ty::Struct(StructId(idx)));
        self.struct_map.insert(strukt.name.id, StructId(idx));
        self.struct_ty_map.insert(StructId(idx), ty_id);
        self.structs.push(strukt);
        StructId(idx)
    }

    pub fn bool(&self) -> TyId {
        TyId(0)
    }

    pub fn unit(&self) -> TyId {
        TyId(usize::MAX)
    }

    pub fn is_unit(&self, ty: TyId) -> bool {
        ty == self.unit()
    }

    pub fn builtin(&self, ident: &str) -> bool {
        match ident {
            "u8" | "u16" | "u32" | "u64" | "i8" | "i16" | "i32" | "i64" | "bool" => true,
            _ => false,
        }
    }

    #[track_caller]
    pub fn ty(&self, ty_id: TyId) -> Ty {
        self.tys.get(ty_id.0).copied().expect("invalid type id")
    }

    pub fn ty_id<P: IntoTyPath>(&self, path: P) -> Option<TyId> {
        let path = path.into_ty_path();
        self.ty_map.get(&path).copied()
    }

    #[track_caller]
    pub fn strukt(&self, id: StructId) -> &Struct {
        self.structs.get(id.0).expect("invalid struct id")
    }

    #[track_caller]
    pub fn expect_struct_ident(&self, ident: IdentId) -> &Struct {
        let id = self.expect_struct_id(ident);
        self.strukt(id)
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

    pub fn build_layouts<'a>(&mut self, ctx: &Ctx<'a>) -> Result<(), Diag<'a>> {
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

    fn layout_struct<'a>(
        ctx: &Ctx<'a>,
        buf: &[Struct],
        struct_ty_map: &HashMap<StructId, TyId>,
        layouts: &mut HashMap<TyId, Layout>,
        offsets: &mut HashMap<StructId, FieldMap>,
        strukt: StructId,
        prev: Option<StructId>,
    ) -> Result<(), Diag<'a>> {
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
            match ctx.tys.ty(field.ty) {
                Ty::Struct(s) => {
                    let ty_id = struct_ty_map.get(&s).unwrap();
                    let layout = layouts.get(ty_id).unwrap();
                    struct_layouts.push(*layout);
                }
                Ty::Int(int) => {
                    struct_layouts.push(int.layout());
                }
                Ty::Bool => {
                    struct_layouts.push(Layout::splat(1));
                }
                Ty::Unit => panic!("field cannot be unit"),
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

        Ok(())
    }
}
