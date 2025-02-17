use super::ctx::Ctx;
use super::ident::{Ident, IdentId};
use super::mem::Layout;
use super::ty::FullTy;
use super::LetExpr;
use crate::diagnostic::Diag;
use crate::lex::buffer::Span;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Struct {
    pub span: Span,
    pub name: Ident,
    pub fields: Vec<Field>,
}

impl Struct {
    pub fn get_field_ty(&self, field: IdentId) -> Option<FullTy> {
        self.fields
            .iter()
            .find(|f| f.name.id == field)
            .map(|f| f.ty)
    }

    #[track_caller]
    pub fn field_ty(&self, field: IdentId) -> FullTy {
        self.get_field_ty(field).expect("invalid field")
    }

    #[track_caller]
    pub fn field_offset(&self, ctx: &Ctx, field: IdentId) -> i32 {
        let map = ctx.structs.fields(ctx.expect_struct_id(self.name.id));
        map.fields.get(&field).expect("invalid field").1
    }
}

#[derive(Debug, Clone)]
pub struct Field {
    pub span: Span,
    pub name: Ident,
    pub ty: FullTy,
}

#[derive(Debug, Clone)]
pub struct StructDef {
    pub span: Span,
    pub id: StructId,
    pub fields: Vec<FieldDef>,
}

#[derive(Debug, Clone)]
pub struct FieldDef {
    pub span: Span,
    pub name: Ident,
    pub expr: LetExpr,
}

#[derive(Debug, Clone)]
pub struct FieldMap {
    pub fields: HashMap<IdentId, (FullTy, ByteOffset)>,
}

pub type ByteOffset = i32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StructId(pub(super) usize);

#[derive(Debug, Default)]
pub struct StructStore {
    pub(super) map: HashMap<IdentId, StructId>,
    pub(super) layouts: HashMap<StructId, Layout>,
    pub(super) fields: HashMap<StructId, FieldMap>,
    pub(super) buf: Vec<Struct>,
}

impl StructStore {
    pub fn store(&mut self, strukt: Struct) -> StructId {
        let idx = self.buf.len();
        self.map.insert(strukt.name.id, StructId(idx));
        self.buf.push(strukt);
        StructId(idx)
    }

    pub fn set_storage(&mut self, map: HashMap<IdentId, StructId>, buf: Vec<Struct>) {
        self.map = map;
        self.buf = buf;
    }

    #[track_caller]
    pub fn strukt(&self, id: StructId) -> &Struct {
        self.buf.get(id.0).expect("invalid struct id")
    }

    #[track_caller]
    pub fn expect_struct_ident(&self, ident: IdentId) -> &Struct {
        let id = self.expect_struct_id(ident);
        self.strukt(id)
    }

    pub fn struct_id(&self, ident: IdentId) -> Option<StructId> {
        self.map.get(&ident).copied()
    }

    #[track_caller]
    pub fn expect_struct_id(&self, ident: IdentId) -> StructId {
        self.struct_id(ident).expect("invalid struct ident")
    }

    #[track_caller]
    pub fn fields(&self, id: StructId) -> &FieldMap {
        self.fields.get(&id).expect("invalid struct id")
    }

    #[track_caller]
    pub fn layout(&self, id: StructId) -> Layout {
        *self.layouts.get(&id).expect("invalid struct id")
    }

    pub(super) fn build_layouts<'a>(
        &self,
        ctx: &Ctx<'a>,
    ) -> Result<(HashMap<StructId, Layout>, HashMap<StructId, FieldMap>), Diag<'a>> {
        let mut map = HashMap::default();
        let mut fields = HashMap::default();

        for (i, strukt) in self.buf.iter().enumerate() {
            if !map.contains_key(&StructId(i)) {
                self.layout_struct(
                    ctx,
                    &mut map,
                    &mut fields,
                    self.expect_struct_id(strukt.name.id),
                    None,
                )?;
            }
        }

        //println!("{:#?}", map);

        Ok((map, fields))
    }

    fn layout_struct<'a>(
        &self,
        ctx: &Ctx<'a>,
        layouts: &mut HashMap<StructId, Layout>,
        offsets: &mut HashMap<StructId, FieldMap>,
        strukt: StructId,
        prev: Option<StructId>,
    ) -> Result<(), Diag<'a>> {
        let struct_id = strukt;
        let strukt = &self.buf[strukt.0];

        if prev.is_some_and(|id| {
            strukt.fields.iter().any(|f| match f.ty {
                FullTy::Struct(s) => s == id,
                _ => false,
            })
        }) {
            return Err(ctx.error(
                "cyclic struct definition",
                strukt.name.span,
                "a field references a type that references this type",
            ));
        }

        let mut unresolved_layouts = Vec::new();
        for field in strukt.fields.iter() {
            match field.ty {
                FullTy::Struct(s) => {
                    unresolved_layouts.push(s);
                }
                _ => {}
            }
        }

        for id in unresolved_layouts.iter() {
            self.layout_struct(ctx, layouts, offsets, *id, Some(struct_id))?
        }

        let strukt = &self.buf[struct_id.0];
        let mut struct_layouts = Vec::with_capacity(strukt.fields.len());
        for field in strukt.fields.iter() {
            match field.ty {
                FullTy::Struct(s) => {
                    let layout = layouts.get(&s).unwrap();
                    struct_layouts.push(*layout);
                }
                FullTy::Ty(ty) => {
                    struct_layouts.push(ty.layout());
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

        layouts.insert(struct_id, Layout::new(byte, alignment));
        offsets.insert(
            struct_id,
            FieldMap {
                fields: struct_offsets,
            },
        );

        Ok(())
    }
}
