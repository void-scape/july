use annotate_snippets::Level;

use super::block::*;
use super::expr::Expr;
use super::expr::ExprId;
use super::expr::ExprKind;
use super::expr::ExprStore;
use super::func::*;
use super::ident::*;
use super::lit::LitStore;
use super::stmt::Stmt;
use super::ty::*;
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
    pub exprs: ExprStore,
    pub idents: IdentStore<'a>,
    pub blocks: BlockStore,
    pub lits: LitStore<'a>,
    pub funcs: FuncStore,
    pub ty: TyRegistry<'a>,
    pub ty_ctx: TyCtx,
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
            blocks: BlockStore::default(),
            funcs: FuncStore::default(),
            ty: TyRegistry::default(),
            lits: LitStore::default(),
            exprs: ExprStore::default(),
            ty_ctx: TyCtx::default(),
            tokens,
        }
    }

    pub fn error(&self, title: &'static str, span: Span, msg: impl Into<String>) -> Diag<'a> {
        Diag::sourced(title, self.tokens.source(), Msg::error(span, msg))
    }

    pub fn errors(&self, title: &'static str, msgs: impl Iterator<Item = Msg>) -> Diag<'a> {
        let mut diag = Diag::new(title, Sourced::new(self.tokens.source(), Level::Error));
        for msg in msgs {
            diag = diag.msg(msg);
        }
        diag
    }

    pub fn store_func(&mut self, func: Func) -> FuncId {
        self.funcs.store(func)
    }

    pub fn store_block(&mut self, block: Block) -> BlockId {
        self.blocks.store(block)
    }

    pub fn store_expr(&mut self, expr: Expr) -> ExprId {
        self.exprs.store(expr)
    }

    #[track_caller]
    pub fn expr(&self, expr: ExprId) -> Expr {
        *self.exprs.expr(expr)
    }

    #[track_caller]
    pub fn store_ident(&mut self, ident: TokenId) -> Ident {
        Ident {
            span: self.span(ident),
            id: self.idents.store(self.tokens.ident(ident)),
        }
    }

    pub fn ty(&self, ty: TokenId) -> Option<Ty> {
        self.ty.ty_str(self.tokens.ident(ty))
    }

    pub fn key(&self) -> Result<TypeKey, Diag<'a>> {
        match self.ty_ctx.resolve(self) {
            Ok(map) => Ok(map),
            Err(errs) => Err(self.errors(
                "type resolution",
                errs.iter().map(|e| match e {
                    TyErr::NotEnoughInfo(span, ty_var) => {
                        let mut msg = None;
                        for block in self.blocks.iter() {
                            for stmt in block.stmts.iter() {
                                match stmt {
                                    Stmt::Let(ident, _, expr) => {
                                        if self.expr(*expr).ty == *ty_var {
                                            msg = Some(Msg::error(ident.span, "cannot infer type"));
                                            break;
                                        }
                                    }
                                    _ => {}
                                }
                            }
                            if msg.is_some() {
                                break;
                            }
                        }

                        msg.unwrap_or_else(|| Msg::error(*span, "cannot infer type"))
                    }
                    TyErr::Arch(span, arch, ty) => {
                        Msg::error(*span, format!("`{ty:?}` does not satify `{arch:?}`"))
                    }
                    TyErr::Abs(span) => Msg::error(*span, format!("conflicting types")),
                }),
            )),
        }
    }
}
