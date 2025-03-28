use super::Expr;
use super::ident::{Ident, IdentId};
use super::ty::Ty;
use super::ty::store::TyStore;
use pebblec_parse::lex::buffer::Span;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Struct {
    #[allow(unused)]
    pub span: Span,
    pub name: Ident,
    pub fields: Vec<Field>,
}

impl Struct {
    pub fn get_field_ty(&self, field: IdentId) -> Option<Ty> {
        self.fields
            .iter()
            .find(|f| f.name.id == field)
            .map(|f| f.ty)
    }

    #[track_caller]
    pub fn field_ty(&self, field: IdentId) -> Ty {
        self.get_field_ty(field).expect("invalid field")
    }

    #[track_caller]
    pub fn field_offset(&self, tys: &TyStore, field: IdentId) -> i32 {
        let map = tys.fields(tys.expect_struct_id(self.name.id));
        map.fields.get(&field).expect("invalid field").1
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Field {
    pub span: Span,
    pub name: Ident,
    pub ty: Ty,
}

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub struct StructDef<'a> {
    pub span: Span,
    pub id: StructId,
    pub fields: &'a [FieldDef<'a>],
}

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub struct FieldDef<'a> {
    pub span: Span,
    pub name: Ident,
    pub expr: Expr<'a>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldMap {
    pub fields: HashMap<IdentId, (Ty, ByteOffset)>,
}

impl FieldMap {
    pub fn field_ty(&self, field: IdentId) -> Option<Ty> {
        self.fields.get(&field).map(|(ty, _)| *ty)
    }
}

pub type ByteOffset = i32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StructId(pub(super) usize);
