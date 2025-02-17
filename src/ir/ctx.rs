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
use super::strukt::StructStore;
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

#[derive(Debug)]
pub struct Ctx<'a> {
    pub tokens: &'a TokenBuffer<'a>,
    pub idents: IdentStore<'a>,
    pub structs: StructStore,
    pub enums: EnumStore,
    pub funcs: Vec<Func>,
    lits: LitStore<'a>,
    ty: TyRegistry<'a>,
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
            ty: TyRegistry::default(),
            lits: LitStore::default(),
            sigs: SigStore::default(),
            structs: StructStore::default(),
            enums: EnumStore::default(),
            funcs: Vec::new(),
            tokens,
        }
    }

    pub fn error(&self, title: &'static str, span: Span, msg: impl Into<String>) -> Diag<'a> {
        Diag::sourced(title, self.tokens.source(), Msg::error(span, msg))
    }

    pub fn errors(&self, title: &'static str, msgs: impl IntoIterator<Item = Msg>) -> Diag<'a> {
        let mut diag = Diag::new(title, Sourced::new(self.tokens.source(), Level::Error));
        for msg in msgs {
            diag = diag.msg(msg);
        }
        diag
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
        self.structs.struct_id(id)
    }

    pub fn struct_name(&self, id: StructId) -> &str {
        let strukt = self.structs.strukt(id);
        self.expect_ident(strukt.name.id)
    }

    pub fn struct_def(&self, id: StructId) -> &Struct {
        self.structs.strukt(id)
    }

    #[track_caller]
    pub fn expect_struct_id(&self, id: IdentId) -> StructId {
        self.structs.expect_struct_id(id)
    }

    pub fn store_enums(&mut self, enums: Vec<Enum>) {
        for enom in enums.into_iter() {
            self.enums.store(enom);
        }
    }

    #[track_caller]
    pub fn expect_enum_id(&self, id: IdentId) -> EnumId {
        self.enums.expect_enum_id(id)
    }

    pub fn build_type_layouts(&mut self) {
        let (layouts, fields) = self.structs.build_layouts(self).unwrap();
        self.structs.layouts = layouts;
        self.structs.fields = fields;
        let (layouts, variants) = self.enums.build_layouts(self).unwrap();
        self.enums.layouts = layouts;
        self.enums.variants = variants;
    }

    pub fn store_sigs(&mut self, sigs: Vec<Sig>) {
        for sig in sigs.into_iter() {
            self.sigs.store(sig);
        }
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

    pub fn ty(&self, ty: TokenId) -> Option<Ty> {
        self.ty.ty_str(self.tokens.ident(ty))
    }
}
