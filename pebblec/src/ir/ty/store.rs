use super::TyKind;
use crate::ir::ident::IdentId;
use crate::ir::mem::Layout;
use crate::ir::strukt::{FieldMap, Struct, StructId};
use crate::ir::ty::Ty;
use pebblec_arena::BlobArena;
use std::collections::HashMap;

pub const BUILTIN_TYPES: &[&str] = &[
    "u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64", "f32", "f64", "bool", "str",
];

#[derive(Debug, Default)]
pub struct TyStore {
    storage: BlobArena,
    interned: HashMap<TyKind, Ty>,

    struct_map: HashMap<IdentId, StructId>,
    struct_ty_map: HashMap<StructId, Ty>,
    structs: Vec<Struct>,
    fields: HashMap<StructId, FieldMap>,

    layouts: HashMap<Ty, Layout>,
}

impl PartialEq for TyStore {
    fn eq(&self, other: &Self) -> bool {
        self.interned == other.interned
            && self.struct_map == other.struct_map
            && self.struct_ty_map == other.struct_ty_map
            && self.structs == other.structs
            && self.fields == other.fields
            && self.layouts == other.layouts
    }
}

impl TyStore {
    pub fn intern_kind(&mut self, kind: TyKind) -> Ty {
        match self.interned.get(&kind) {
            Some(ty) => *ty,
            None => {
                let ty = Ty(self.storage.alloc(kind));
                self.interned.insert(kind, ty);
                ty
            }
        }
    }

    // TODO: structs should be stored within `TyKind`
    pub fn store_struct(&mut self, strukt: Struct) -> StructId {
        let idx = self.structs.len();
        let ty = self.intern_kind(TyKind::Struct(StructId(idx)));
        self.struct_map.insert(strukt.name.id, StructId(idx));
        self.struct_ty_map.insert(StructId(idx), ty);
        self.structs.push(strukt);
        StructId(idx)
    }

    /// Used during the construction of types, where [`Ty`]s are not easily accessible.
    pub fn is_builtin(&self, ident: &str) -> bool {
        BUILTIN_TYPES.contains(&ident)
    }

    #[track_caller]
    pub fn strukt(&self, id: StructId) -> &Struct {
        self.structs.get(id.0).expect("invalid struct id")
    }

    #[track_caller]
    pub fn struct_ty_id(&self, struct_id: StructId) -> Ty {
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
    pub fn layout(&self, ty: Ty) -> Layout {
        *self.layouts.get(&ty).expect("invalid type id")
    }

    #[track_caller]
    pub fn struct_layout(&self, struct_id: StructId) -> Layout {
        let ty_id = self
            .struct_ty_map
            .get(&struct_id)
            .expect("invalid struct id");
        self.layout(*ty_id)
    }

    pub fn build_layouts(&mut self) {
        for (i, strukt) in self.structs.iter().enumerate() {
            let ty_id = self.struct_ty_map.get(&StructId(i)).unwrap();
            if !self.layouts.contains_key(&ty_id) {
                let id = self.expect_struct_id(strukt.name.id);
                Self::layout_struct(
                    &self.structs,
                    &self.struct_ty_map,
                    &mut self.layouts,
                    &mut self.fields,
                    id,
                    None,
                );
            }
        }
    }

    fn layout_struct(
        buf: &[Struct],
        struct_ty_map: &HashMap<StructId, Ty>,
        layouts: &mut HashMap<Ty, Layout>,
        offsets: &mut HashMap<StructId, FieldMap>,
        strukt: StructId,
        prev: Option<StructId>,
    ) {
        let struct_id = strukt;
        let strukt = &buf[strukt.0];

        if prev.is_some_and(|id| {
            strukt.fields.iter().any(|f| match f.ty.0 {
                TyKind::Struct(s) => *s == id,
                _ => false,
            })
        }) {
            unreachable!("checked during struct sizing");
        }

        let mut unresolved_layouts = Vec::new();
        for field in strukt.fields.iter() {
            match field.ty.0 {
                TyKind::Struct(s) => {
                    unresolved_layouts.push(s);
                }
                _ => {}
            }
        }

        for id in unresolved_layouts.iter() {
            Self::layout_struct(buf, struct_ty_map, layouts, offsets, **id, Some(struct_id));
        }

        let strukt = &buf[struct_id.0];
        let mut struct_layouts = Vec::with_capacity(strukt.fields.len());
        for field in strukt.fields.iter() {
            assert!(field.ty.is_sized());
            struct_layouts.push(field.ty.layout_with(struct_ty_map, layouts));
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
    }
}

impl TyKind {
    #[track_caller]
    fn layout_with(
        &self,
        struct_ty_map: &HashMap<StructId, Ty>,
        layouts: &HashMap<Ty, Layout>,
    ) -> Layout {
        match self {
            Self::Unit => Layout::new(0, 1),
            Self::Bool => Layout::splat(1),
            Self::Int(int) => int.layout(),
            Self::Float(float) => float.layout(),
            Self::Ref(Self::Str) => Layout::FAT_PTR,
            Self::Ref(Self::Slice(_)) => Layout::FAT_PTR,
            Self::Str | Self::Ref(_) => Layout::PTR,
            Self::Array(len, inner) => inner.layout_with(struct_ty_map, layouts).to_array(*len),
            Self::Slice(_) => todo!("unsized"),
            Self::Struct(id) => {
                let ty_id = struct_ty_map.get(&id).unwrap();
                *layouts.get(ty_id).unwrap()
            }
        }
    }
}
