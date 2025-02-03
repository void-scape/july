use super::block::{Block, BlockId};
use super::ctx::Ctx;
use super::ident::{Ident, IdentId};
use super::ty::Ty;
use super::SYM_DEF;
use crate::diagnostic::{Diag, Msg};
use crate::lex::buffer::{Span, TokenQuery};
use crate::parse::rules::prelude as rules;
use std::collections::HashMap;

#[derive(Debug)]
pub struct Func {
    pub span: Span,
    pub sig: Sig,
    pub block: BlockId,
}

#[derive(Debug, Clone, Copy)]
pub struct Sig {
    pub span: Span,
    pub ident: IdentId,
    pub ty: Ty,
}

impl Func {
    pub fn sig<'a>(ctx: &mut Ctx<'a>, func: &rules::Func) -> Result<(), Diag<'a>> {
        let ty = if let Some(ty) = func.ty {
            ctx.ty(ty).ok_or_else(|| {
                ctx.error(
                    SYM_DEF,
                    ctx.span(ty),
                    format!("`{}` is not a type, expected a type", ctx.ident(ty)),
                )
            })?
        } else {
            Ty::Unit
        };

        let span = ctx.span(func.name);
        let ident = ctx.store_ident(func.name).id;
        ctx.funcs.store_sig(Sig { span, ident, ty });

        Ok(())
    }

    pub fn lower<'a>(ctx: &mut Ctx<'a>, func: &rules::Func) -> Result<(), Diag<'a>> {
        let ident = ctx.store_ident(func.name);
        let sig = ctx.funcs.sig(ident).unwrap().clone();
        let func = Self {
            span: func.span,
            block: Block::lower(ctx, &func.block, Some(sig.ty)).map_err(|diag| {
                diag.msg(Msg::note(func.block.span, "while parsing this function"))
            })?,
            sig,
        };
        ctx.store_func(func);

        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FuncId(usize);

#[derive(Debug, Default)]
pub struct FuncStore {
    sigs: HashMap<IdentId, Sig>,
    map: HashMap<IdentId, FuncId>,
    funcs: Vec<Func>,
}

impl FuncStore {
    pub fn store(&mut self, func: Func) -> FuncId {
        let idx = self.funcs.len();
        self.map.insert(func.sig.ident, FuncId(idx));
        self.funcs.push(func);
        FuncId(idx)
    }

    pub fn store_sig(&mut self, sig: Sig) {
        self.sigs.insert(sig.ident, sig);
    }

    #[track_caller]
    pub fn func(&self, id: FuncId) -> &Func {
        self.funcs.get(id.0).expect("invalid func id")
    }

    pub fn sig(&self, ident: Ident) -> Option<&Sig> {
        self.sigs.get(&ident.id)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Func> {
        self.funcs.iter()
    }
}
