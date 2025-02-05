use annotate_snippets::Level;
//use super::block::*;
//use super::expr::Expr;
//use super::expr::ExprId;
//use super::expr::ExprKind;
//use super::expr::ExprStore;
//use super::func::*;
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
//use super::stmt::Stmt;
use super::ty::*;
use super::Func;
use crate::diagnostic::Diag;
use crate::diagnostic::Msg;
use crate::diagnostic::Sourced;
use crate::lex::buffer::Buffer;
use crate::lex::buffer::Span;
use crate::lex::buffer::TokenQuery;
use crate::lex::buffer::{TokenBuffer, TokenId};

#[derive(Debug)]
pub struct Ctx<'a> {
    pub tokens: &'a TokenBuffer<'a>,
    //pub exprs: ExprStore,
    idents: IdentStore<'a>,
    //pub blocks: BlockStore,
    lits: LitStore<'a>,
    //pub funcs: FuncStore,
    ty: TyRegistry<'a>,
    sigs: SigStore,
    structs: StructStore,
    pub funcs: Vec<Func>,
    //pub ty_ctx: TyCtx,
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
            //blocks: BlockStore::default(),
            //funcs: FuncStore::default(),
            ty: TyRegistry::default(),
            lits: LitStore::default(),
            sigs: SigStore::default(),
            structs: StructStore::default(),
            funcs: Vec::new(),
            //exprs: ExprStore::default(),
            //ty_ctx: TyCtx::default(),
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

    pub fn store_structs(&mut self, structs: Vec<Struct>) {
        for strukt in structs.into_iter() {
            self.structs.store(strukt);
        }
    }

    pub fn struct_id(&self, id: IdentId) -> Option<StructId> {
        self.structs.struct_id(id)
    }

    pub fn layout_structs(&mut self) {
        let layouts = self.structs.layout(self).unwrap();
        self.structs.layouts = layouts;
    }

    pub fn store_sig(&mut self, sig: Sig) {
        self.sigs.store(sig);
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

    //pub fn key(&self) -> Result<TypeKey, Diag<'a>> {
    //    match self.ty_ctx.resolve(self) {
    //        Ok(map) => Ok(map),
    //        Err(errs) => Err(self.errors(
    //            "type resolution",
    //            errs.iter().map(|e| match e {
    //                TyErr::NotEnoughInfo(span, ty_var) => {
    //                    let mut msg = None;
    //                    for block in self.blocks.iter() {
    //                        for stmt in block.stmts.iter() {
    //                            match stmt {
    //                                Stmt::Let(ident, _, expr) => {
    //                                    if self.expr(*expr).ty == *ty_var {
    //                                        msg = Some(Msg::error(ident.span, "cannot infer type"));
    //                                        break;
    //                                    }
    //                                }
    //                                _ => {}
    //                            }
    //                        }
    //                        if msg.is_some() {
    //                            break;
    //                        }
    //                    }
    //
    //                    msg.unwrap_or_else(|| Msg::error(*span, "cannot infer type"))
    //                }
    //                TyErr::Arch(span, arch, ty) => {
    //                    Msg::error(*span, format!("`{ty:?}` does not satify `{arch:?}`"))
    //                }
    //                TyErr::Abs(span) => Msg::error(*span, format!("conflicting types")),
    //            }),
    //        )),
    //    }
    //}
}
