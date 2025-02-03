use super::ctx::Ctx;
use super::expr::{Expr, ExprId};
use super::ident::Ident;
use super::ty::{Ty, TyVar};
use super::SYM_DEF;
use crate::diagnostic::Diag;
use crate::lex::buffer::TokenQuery;
use crate::parse::rules::prelude as rules;

#[derive(Debug)]
pub enum Stmt {
    Let(Ident, Option<Ty>, ExprId),
    Semi(ExprId),
    Open(ExprId),
}

impl Stmt {
    pub fn lower<'a>(
        ctx: &mut Ctx<'a>,
        stmt: &rules::Stmt,
        ty_var: TyVar,
    ) -> Result<Stmt, Diag<'a>> {
        Ok(match stmt {
            rules::Stmt::Let { name, ty, assign } => {
                let ty = if let Some(ty) = ty {
                    Some(ctx.ty(*ty).ok_or_else(|| {
                        ctx.error(
                            SYM_DEF,
                            ctx.span(*ty),
                            format!("`{}` is not a type, expected a type", ctx.ident(*ty)),
                        )
                    })?)
                } else {
                    None
                };

                let ident = ctx.store_ident(*name);
                ctx.ty_ctx.register(ty_var, ident.id);
                Stmt::Let(ident, ty, Expr::lower(ctx, assign, ty_var)?)
            }
            rules::Stmt::Semi(expr) => Stmt::Semi(Expr::lower(ctx, expr, ty_var)?),
            rules::Stmt::Open(expr) => Stmt::Open(Expr::lower(ctx, expr, ty_var)?),
        })
    }
}
