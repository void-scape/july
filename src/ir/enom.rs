use super::ctx::Ctx;
use super::ident::{Ident, IdentId};
use super::mem::Layout;
use crate::diagnostic::Diag;
use crate::lex::buffer::Span;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Enum {
    pub span: Span,
    pub name: Ident,
    pub variants: Vec<Variant>,
}

impl Enum {
    //pub fn get_field_ty(&self, field: IdentId) -> Option<Ty> {
    //    self.fields
    //        .iter()
    //        .find(|f| f.name.id == field)
    //        .map(|f| f.ty)
    //}
    //
    //#[track_caller]
    //pub fn field_ty(&self, field: IdentId) -> Ty {
    //    self.get_field_ty(field).expect("invalid field")
    //}

    //#[track_caller]
    //pub fn variant_val(&self, ctx: &Ctx, field: IdentId) -> usize {
    //    let map = ctx
    //        .enums
    //        .variants
    //        .get(&ctx.expect_enum_id(self.name.id))
    //        .unwrap();
    //    *map.get(&field).expect("invalid field")
    //}
}

#[derive(Debug, Clone, Copy)]
pub struct Variant {
    pub span: Span,
    pub name: Ident,
}

#[derive(Debug, Clone, Copy)]
pub struct EnumDef {
    pub span: Span,
    pub name: Ident,
    pub variant: Variant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EnumId(usize);

#[derive(Debug, Default)]
pub struct EnumStore {
    pub(super) map: HashMap<IdentId, EnumId>,
    pub(super) layouts: HashMap<EnumId, Layout>,
    pub(super) variants: HashMap<EnumId, HashMap<IdentId, usize>>,
    pub(super) buf: Vec<Enum>,
}

impl EnumStore {
    pub fn store(&mut self, enom: Enum) -> EnumId {
        let idx = self.buf.len();
        self.map.insert(enom.name.id, EnumId(idx));
        self.buf.push(enom);
        EnumId(idx)
    }

    #[track_caller]
    pub fn enom(&self, id: EnumId) -> &Enum {
        self.buf.get(id.0).expect("invalid struct id")
    }

    #[track_caller]
    pub fn expect_enum_ident(&self, ident: IdentId) -> &Enum {
        let id = self.expect_enum_id(ident);
        self.enom(id)
    }

    pub fn enum_id(&self, ident: IdentId) -> Option<EnumId> {
        self.map.get(&ident).copied()
    }

    #[track_caller]
    pub fn expect_enum_id(&self, ident: IdentId) -> EnumId {
        self.enum_id(ident).expect("invalid struct ident")
    }

    #[track_caller]
    pub fn layout(&self, id: EnumId) -> Layout {
        *self.layouts.get(&id).expect("invalid struct id")
    }

    pub(super) fn build_layouts<'a>(
        &self,
        ctx: &Ctx<'a>,
    ) -> Result<
        (
            HashMap<EnumId, Layout>,
            HashMap<EnumId, HashMap<IdentId, usize>>,
        ),
        Diag<'a>,
    > {
        let mut map = HashMap::default();
        let mut variants = HashMap::default();

        for (i, enom) in self.buf.iter().enumerate() {
            if !map.contains_key(&EnumId(i)) {
                self.layout_enum(
                    ctx,
                    &mut map,
                    &mut variants,
                    self.expect_enum_id(enom.name.id),
                    None,
                )?;
            }
        }

        Ok((map, variants))
    }

    fn layout_enum<'a>(
        &self,
        ctx: &Ctx<'a>,
        layouts: &mut HashMap<EnumId, Layout>,
        variants: &mut HashMap<EnumId, HashMap<IdentId, usize>>,
        enom: EnumId,
        prev: Option<IdentId>,
    ) -> Result<(), Diag<'a>> {
        let enum_id = enom;
        //let enom = &self.buf[enom.0];

        //if prev.is_some_and(|id| {
        //    strukt.fields.iter().any(|f| match f.ty {
        //        Ty::Struct(s) => s == id,
        //        _ => false,
        //    })
        //}) {
        //    return Err(ctx.error(
        //        "cyclic struct definition",
        //        enom.name.span,
        //        "a field references a type that references this type",
        //    ));
        //}

        //let mut unresolved_layouts = Vec::new();
        //for variant in enom.fields.iter() {
        //    match field.ty {
        //        Ty::Struct(s) => {
        //            let id = self.expect_enum_id(s);
        //            unresolved_layouts.push(id);
        //        }
        //        _ => {}
        //    }
        //}

        //let name = enom.name.id;
        //for id in unresolved_layouts.iter() {
        //    self.layout_struct(ctx, layouts, offsets, *id, Some(name))?
        //}

        //let strukt = &self.buf[enum_id.0];
        //let mut struct_layouts = Vec::with_capacity(strukt.fields.len());
        //for field in strukt.fields.iter() {
        //    match field.ty {
        //        Ty::Struct(s) => {
        //            let id = self.expect_enum_id(s);
        //            let layout = layouts.get(&id).unwrap();
        //            struct_layouts.push(*layout);
        //        }
        //        Ty::Ty(ty) => {
        //            struct_layouts.push(ty.layout());
        //        }
        //    }
        //}
        //
        //let mut alignment = 1;
        //for layout in struct_layouts.iter() {
        //    if layout.alignment > alignment {
        //        alignment = layout.alignment;
        //    }
        //}

        layouts.insert(enum_id, Layout::new(1, 1));
        let enom = ctx.enums.enom(enum_id);
        let entry = variants.entry(enum_id).or_default();
        for (i, variant) in enom.variants.iter().enumerate() {
            entry.insert(variant.name.id, i);
        }

        Ok(())
    }
}
