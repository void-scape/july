use super::ctx::Ctx;
use super::expr::ExprId;
use super::ident::{Ident, IdentId};
use super::stmt::Stmt;
use super::ty::{Constraint, ConstraintKind, Ty, TypeKey};
use super::SYM_DEF;
use crate::diagnostic::{Diag, Msg};
use crate::lex::buffer::Span;
use crate::parse::rules::prelude as rules;
use std::collections::HashMap;

#[derive(Debug)]
pub struct Block {
    pub span: Span,
    pub stmts: Vec<Stmt>,
    pub end: Option<ExprId>,
}

impl Block {
    pub fn lower<'a>(
        ctx: &mut Ctx<'a>,
        block: &rules::Block,
        expected_return: Option<Ty>,
    ) -> Result<BlockId, Diag<'a>> {
        let mut stmts = block
            .stmts
            .iter()
            .map(|stmt| {
                let var = ctx.ty_ctx.var();
                Stmt::lower(ctx, stmt, var)
            })
            .collect::<Result<Vec<Stmt>, _>>()?;

        validate_declarations(ctx, &stmts)?;
        validate_shadows(ctx, &stmts)?;

        let end = if let Some(Stmt::Open(expr)) = stmts.last() {
            let expr = *expr;
            stmts.pop();
            Some(expr)
        } else {
            None
        };

        no_open_stmt(ctx, &stmts)?;
        constrain_types(ctx, &stmts, end, expected_return)?;
        let block = Self {
            span: block.span,
            stmts,
            end,
            //types,
        };

        Ok(ctx.store_block(block))
    }
}

pub fn validate_end_exprs<'a>(ctx: &Ctx<'a>, key: &TypeKey) -> Result<(), Diag<'a>> {
    for func in ctx.funcs.iter() {
        let block = ctx.blocks.block(func.block);
        if let Some(end) = block.end {
            let end = ctx.expr(end);
            let ty = key.ty(end.ty);
            if ty != func.sig.ty {
                return Err(ctx.errors(
                    "mismatched types",
                    [
                        Msg::error(
                            end.span,
                            format!("expected `{}`, got `{}`", func.sig.ty.as_str(), ty.as_str()),
                        ),
                        Msg::help(func.sig.span, "because of this"),
                    ]
                    .into_iter(),
                ));
            }
        } else {
            if func.sig.ty != Ty::Unit {
                return Err(ctx.error(
                    "mismatched types",
                    block.span,
                    format!("expected `{}`, got `()`", func.sig.ty.as_str()),
                ));
            }
        }
    }

    Ok(())
}

fn constrain_types<'a>(
    ctx: &mut Ctx<'a>,
    stmts: &[Stmt],
    end: Option<ExprId>,
    expected_return: Option<Ty>,
) -> Result<(), Diag<'a>> {
    let mut ident_hash = HashMap::<IdentId, Ident>::default();

    for stmt in stmts.iter() {
        match stmt {
            Stmt::Let(ident, ty, expr) => {
                ident_hash.insert(ident.id, *ident);
                let expr = ctx.expr(*expr);

                if let Some(ty) = ty {
                    ctx.ty_ctx.constrain(
                        expr.ty,
                        Constraint {
                            span: expr.span,
                            kind: ConstraintKind::Abs(*ty),
                        },
                    );
                }

                expr.constrain(ctx);
            }
            Stmt::Semi(expr) | Stmt::Open(expr) => {
                ctx.expr(*expr).constrain(ctx);
            }
        }
    }

    if let Some(end) = end {
        if let Some(exp) = expected_return {
            let end = ctx.expr(end);
            end.constrain(ctx);
            ctx.ty_ctx.constrain(
                end.ty,
                Constraint {
                    span: end.span,
                    kind: ConstraintKind::Abs(exp),
                },
            );

            if exp.is_unit() {
                if !end.hint_satisfies(ctx, Ty::Unit) {
                    return Err(ctx.error("invalid return type", end.span, "expected unit"));
                }
            }
        }
    }

    Ok(())
}

fn no_open_stmt<'a>(_ctx: &Ctx<'a>, stmts: &[Stmt]) -> Result<(), Diag<'a>> {
    for stmt in stmts.iter() {
        if matches!(stmt, Stmt::Open(_)) {
            panic!("how is this possible?");
        }
    }

    Ok(())
}

fn validate_declarations<'a>(ctx: &Ctx<'a>, stmts: &[Stmt]) -> Result<(), Diag<'a>> {
    let mut declared_idents = Vec::new();
    for stmt in stmts.iter() {
        match stmt {
            Stmt::Let(ident, _, expr) => {
                if let Some(ident) = ctx
                    .expr(*expr)
                    .accessed_idents(ctx)
                    .iter()
                    .find(|i| !declared_idents.contains(&i.id))
                {
                    return Err(ctx.error(SYM_DEF, ident.span, "undeclared variable"));
                }
                declared_idents.push(ident.id);
            }
            Stmt::Semi(expr) | Stmt::Open(expr) => {
                if let Some(ident) = ctx
                    .expr(*expr)
                    .accessed_idents(ctx)
                    .iter()
                    .find(|i| !declared_idents.contains(&i.id))
                {
                    return Err(ctx.error(SYM_DEF, ident.span, "undeclared variable"));
                }
            }
        }
    }

    Ok(())
}

fn validate_shadows<'a>(ctx: &Ctx<'a>, stmts: &[Stmt]) -> Result<(), Diag<'a>> {
    let mut idents = HashMap::<IdentId, Span>::default();
    for stmt in stmts.iter() {
        match stmt {
            Stmt::Let(ident, _, _) => {
                if let Some(prev_span) = idents.get(&ident.id) {
                    return Err(ctx
                        .error("invalid declaration", ident.span, "cannot shadow variables")
                        .msg(Msg::note(*prev_span, "previously defined here")));
                } else {
                    idents.insert(ident.id, ident.span);
                }
            }
            _ => {}
        }
    }

    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub struct BlockId(usize);

#[derive(Debug, Default)]
pub struct BlockStore {
    blocks: Vec<Block>,
}

impl BlockStore {
    pub fn store(&mut self, block: Block) -> BlockId {
        let idx = self.blocks.len();
        self.blocks.push(block);
        BlockId(idx)
    }

    #[track_caller]
    pub fn block(&self, id: BlockId) -> &Block {
        self.blocks.get(id.0).expect("invalid block id")
    }

    pub fn iter(&self) -> impl Iterator<Item = &Block> {
        self.blocks.iter()
    }
}
