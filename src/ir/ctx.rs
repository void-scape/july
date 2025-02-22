use super::enom::Enum;
use super::enom::EnumId;
use super::enom::EnumStore;
use super::ident::*;
use super::lit::Lit;
use super::lit::LitId;
use super::lit::LitKind;
use super::lit::LitStore;
use super::sig::Sig;
use super::sig::SigStore;
use super::strukt::Struct;
use super::strukt::StructId;
use super::ty::store::TyId;
use super::ty::store::TyStore;
use super::ty::*;
use super::Func;
use crate::diagnostic::Diag;
use crate::diagnostic::Msg;
use crate::diagnostic::Sourced;
use crate::lex::buffer::Buffer;
use crate::lex::buffer::Span;
use crate::lex::buffer::TokenQuery;
use crate::lex::buffer::{TokenBuffer, TokenId};
use annotate_snippets::Level;
use std::collections::HashSet;

#[derive(Debug)]
pub struct Ctx<'a> {
    pub tokens: &'a TokenBuffer<'a>,
    pub idents: IdentStore<'a>,
    pub tys: TyStore,
    pub enums: EnumStore,
    pub funcs: Vec<Func>,
    lits: LitStore<'a>,
    //ty: TyRegistry<'a>,
    sigs: SigStore,
}

impl<'a> Buffer<'a> for Ctx<'a> {
    fn token_buffer(&'a self) -> &'a TokenBuffer<'a> {
        self.tokens
    }
}

impl<'a> Ctx<'a> {
    pub fn new(tokens: &'a TokenBuffer<'a>) -> Self {
        Self {
            idents: IdentStore::default(),
            //ty: TyRegistry::default(),
            lits: LitStore::default(),
            sigs: SigStore::default(),
            tys: TyStore::default(),
            enums: EnumStore::default(),
            funcs: Vec::new(),
            tokens,
        }
    }

    #[track_caller]
    pub fn error(&self, title: &'static str, span: Span, msg: impl Into<String>) -> Diag<'a> {
        Diag::sourced(title, self.tokens.source(), Msg::error(span, msg))
            .with_loc(std::panic::Location::caller())
    }

    pub fn mismatch<E: DebugTyId, G: DebugTyId>(
        &self,
        span: Span,
        expected: E,
        got: G,
    ) -> Diag<'a> {
        self.error(
            "mismatched types",
            span,
            format!(
                "expected `{}`, got `{}`",
                expected.debug_ty_id(self),
                got.debug_ty_id(self)
            ),
        )
    }

    pub fn errors(&self, title: &'static str, msgs: impl IntoIterator<Item = Msg>) -> Diag<'a> {
        let mut diag = Diag::new(title, Sourced::new(self.tokens.source(), Level::Error));
        for msg in msgs {
            diag = diag.msg(msg);
        }
        diag.with_loc(std::panic::Location::caller())
    }

    pub fn ty_str(&self, ty: TyId) -> &str {
        self.tys.ty(ty).as_str(self)
    }

    pub fn store_funcs(&mut self, funcs: Vec<Func>) {
        self.funcs.extend(funcs.into_iter());
    }

    //pub fn store_structs(&mut self, structs: Vec<Struct>) {
    //    for strukt in structs.into_iter() {
    //        println!("store: {strukt:#?}");
    //        self.structs.store(strukt);
    //    }
    //}

    pub fn struct_id(&self, id: IdentId) -> Option<StructId> {
        self.tys.struct_id(id)
    }

    pub fn struct_name(&self, id: StructId) -> &str {
        let strukt = self.tys.strukt(id);
        self.expect_ident(strukt.name.id)
    }

    pub fn struct_def(&self, id: StructId) -> &Struct {
        self.tys.strukt(id)
    }

    #[track_caller]
    pub fn expect_struct_id(&self, id: IdentId) -> StructId {
        self.tys.expect_struct_id(id)
    }

    //pub fn store_enums(&mut self, enums: Vec<Enum>) {
    //    for enom in enums.into_iter() {
    //        self.enums.store(enom);
    //    }
    //}

    //#[track_caller]
    //pub fn expect_enum_id(&self, id: IdentId) -> EnumId {
    //    self.enums.expect_enum_id(id)
    //}

    pub fn build_type_layouts(&mut self) {
        let mut tys = self.tys.clone();
        tys.build_layouts(self).unwrap();
        self.tys = tys;

        //todo!();
        //let (layouts, variants) = self.enums.build_layouts(self).unwrap();
        //self.enums.layouts = layouts;
        //self.enums.variants = variants;
    }

    pub fn store_sigs(&mut self, sigs: Vec<Sig>) -> Result<(), Diag<'a>> {
        let mut set = HashSet::<&Sig>::default();
        for sig in sigs.iter() {
            if !set.insert(sig) {
                return Err(self.errors(
                    "duplicate function names",
                    [Msg::error(sig.span, "Function is already defined")],
                ));
            }
        }

        for sig in sigs.into_iter() {
            self.sigs.store(sig);
        }

        Ok(())
    }

    pub fn get_sig(&self, ident: IdentId) -> Option<&Sig> {
        self.sigs.get_sig(ident)
    }

    #[track_caller]
    pub fn store_ident(&mut self, ident: TokenId) -> Ident {
        Ident {
            span: self.span(ident),
            id: self.idents.store(self.tokens.ident(ident)),
        }
    }

    pub fn get_ident(&self, id: IdentId) -> Option<&'a str> {
        self.idents.get_ident(id)
    }

    #[track_caller]
    pub fn expect_ident(&self, id: IdentId) -> &'a str {
        self.get_ident(id).expect("invalid ident id")
    }

    pub fn get_lit(&self, id: LitId) -> Option<LitKind<'a>> {
        self.lits.get_lit(id)
    }

    #[track_caller]
    pub fn expect_lit(&self, id: LitId) -> LitKind<'a> {
        self.get_lit(id).expect("invalid lit id")
    }

    #[track_caller]
    pub fn store_int(&mut self, int: TokenId) -> Lit {
        Lit {
            span: self.span(int),
            kind: self.lits.store(LitKind::Int(self.int_lit(int))),
        }
    }

    //pub fn ty(&self, ty: TokenId) -> Option<Ty> {
    //    self.ty.ty_str(self.tokens.ident(ty))
    //}
}

pub trait DebugTyId {
    fn debug_ty_id<'a>(&self, ctx: &'a Ctx<'a>) -> &'a str;
}

impl<T> DebugTyId for &T
where
    T: DebugTyId,
{
    fn debug_ty_id<'a>(&self, ctx: &'a Ctx<'a>) -> &'a str {
        <T as DebugTyId>::debug_ty_id(self, ctx)
    }
}

impl DebugTyId for &'static str {
    fn debug_ty_id<'a>(&self, _: &Ctx<'a>) -> &'a str {
        self
    }
}

impl DebugTyId for TyId {
    fn debug_ty_id<'a>(&self, ctx: &'a Ctx<'a>) -> &'a str {
        ctx.ty_str(*self)
    }
}

impl DebugTyId for Ty {
    fn debug_ty_id<'a>(&self, ctx: &'a Ctx<'a>) -> &'a str {
        self.as_str(ctx)
    }
}
