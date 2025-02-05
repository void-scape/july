use super::ctx::Ctx;
use super::ident::{Ident, IdentId};
use super::ty::FullTy;
use super::LetExpr;
use crate::diagnostic::Diag;
use crate::lex::buffer::Span;
use std::collections::HashMap;
use std::fmt::Alignment;

#[derive(Debug, Clone)]
pub struct Struct {
    pub span: Span,
    pub name: Ident,
    pub fields: Vec<Field>,
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
    pub name: Ident,
    pub fields: Vec<FieldDef>,
}

#[derive(Debug, Clone)]
pub struct FieldDef {
    pub span: Span,
    pub name: Ident,
    pub expr: LetExpr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StructId(usize);

#[derive(Debug, Default)]
pub struct StructStore {
    pub(super) map: HashMap<IdentId, StructId>,
    pub(super) layouts: HashMap<StructId, Layout>,
    pub(super) buf: Vec<Struct>,
}

impl StructStore {
    pub fn store(&mut self, strukt: Struct) -> StructId {
        let idx = self.buf.len();
        self.map.insert(strukt.name.id, StructId(idx));
        self.buf.push(strukt);
        StructId(idx)
    }

    pub fn strukt(&self, id: StructId) -> Option<&Struct> {
        self.buf.get(id.0)
    }

    pub fn id(&self, ident: IdentId) -> Option<StructId> {
        self.map.get(&ident).copied()
    }

    pub fn layout<'a>(&self, ctx: &Ctx<'a>) -> Result<HashMap<StructId, Layout>, Diag<'a>> {
        let mut map = HashMap::default();

        for (i, strukt) in self.buf.iter().enumerate() {
            if !map.contains_key(&StructId(i)) {
                self.layout_struct(ctx, &mut map, self.expect_struct_id(strukt.name.id), None)?;
            }
        }

        Ok(map)
    }

    pub fn struct_id(&self, ident: IdentId) -> Option<StructId> {
        self.map.get(&ident).copied()
    }

    #[track_caller]
    fn expect_struct_id(&self, ident: IdentId) -> StructId {
        self.struct_id(ident).expect("invalid struct ident")
    }

    fn layout_struct<'a>(
        &self,
        ctx: &Ctx<'a>,
        layouts: &mut HashMap<StructId, Layout>,
        strukt: StructId,
        prev: Option<IdentId>,
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
                    let id = self.expect_struct_id(s);
                    unresolved_layouts.push(id);
                }
                _ => {}
            }
        }

        let name = strukt.name.id;
        for id in unresolved_layouts.iter() {
            self.layout_struct(ctx, layouts, *id, Some(name))?
        }

        let strukt = &self.buf[struct_id.0];
        let mut struct_layouts = Vec::with_capacity(strukt.fields.len());
        for field in strukt.fields.iter() {
            match field.ty {
                FullTy::Struct(s) => {
                    let id = self.expect_struct_id(s);
                    let layout = layouts.get(&id).unwrap();
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

        let mut byte = 0;
        for layout in struct_layouts.iter() {
            //if layout.
        }

        println!("pushing: {struct_id:?}");
        layouts.insert(struct_id, Layout::splat(1));

        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    pub size: usize,
    pub alignment: usize,
}

impl Layout {
    pub fn new(size: usize, alignment: usize) -> Self {
        Self { size, alignment }
    }

    pub fn splat(n: usize) -> Self {
        Self::new(n, n)
    }
}
