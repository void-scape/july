use super::ctx::Ctx;
use super::func::Sig;
use super::ident::Ident;
use super::lit::{Lit, LitId};
use super::ty::{Arch, Constraint, ConstraintKind, Ty, TyVar};
use super::SYM_DEF;
use crate::diagnostic::{Diag, Msg};
use crate::lex::buffer::{Span, TokenQuery};
use crate::parse::rules::prelude as rules;

#[derive(Debug, Clone, Copy)]
pub struct Expr {
    pub span: Span,
    pub ty: TyVar,
    pub kind: ExprKind,
}

#[derive(Debug, Clone, Copy)]
pub enum ExprKind {
    Ident(Ident),
    Lit(LitId),
    Bin(BinOp, ExprId, ExprId),
    Call(Sig),
    Ret(Option<ExprId>),
}

#[derive(Debug, Clone, Copy)]
pub struct BinOp {
    pub span: Span,
    pub kind: BinOpKind,
}

impl Expr {
    pub fn lower<'a>(
        ctx: &mut Ctx<'a>,
        expr: &rules::Expr,
        ty_var: TyVar,
    ) -> Result<ExprId, Diag<'a>> {
        let expr = match expr {
            rules::Expr::Ident(id) => {
                let ident = ctx.store_ident(*id);
                for other in ctx.ty_ctx.get_vars(ident.id).to_vec().iter() {
                    ctx.ty_ctx.constrain(
                        *other,
                        Constraint {
                            span: ctx.span(*id),
                            kind: ConstraintKind::Equate(ty_var),
                        },
                    );

                    ctx.ty_ctx.constrain(
                        ty_var,
                        Constraint {
                            span: ctx.span(*id),
                            kind: ConstraintKind::Equate(*other),
                        },
                    );
                }

                Expr {
                    span: ctx.span(*id),
                    ty: ty_var,
                    kind: ExprKind::Ident(ident),
                }
            }
            rules::Expr::Lit(id) => Expr {
                span: ctx.span(*id),
                ty: ty_var,
                kind: ExprKind::Lit(ctx.lits.store(Lit::Int(ctx.int_lit(*id)))),
            },
            rules::Expr::Bin(op, lhs, rhs) => {
                let lhs = Expr::lower(ctx, lhs, ty_var)?;
                let rhs = Expr::lower(ctx, rhs, ty_var)?;

                Expr {
                    span: { Span::from_spans(ctx.exprs.expr(lhs).span, ctx.exprs.expr(rhs).span) },
                    ty: ty_var,
                    kind: ExprKind::Bin(*op, lhs, rhs),
                }
            }
            rules::Expr::Ret(span, expr) => {
                let inner = if let Some(expr) = expr {
                    Some(Expr::lower(ctx, &expr, ty_var)?)
                } else {
                    None
                };

                Expr {
                    span: *span,
                    ty: ty_var,
                    kind: ExprKind::Ret(inner),
                }
            }
            rules::Expr::Call { span, func } => {
                let ident = ctx.store_ident(*func);
                Expr {
                    span: *span,
                    ty: ty_var,
                    kind: ExprKind::Call(
                        ctx.funcs
                            .sig(ident)
                            .ok_or_else(|| {
                                ctx.error(
                                    SYM_DEF,
                                    *span,
                                    format!(
                                        "function `{}` is not in the current scope",
                                        ctx.ident(*func)
                                    ),
                                )
                            })?
                            .clone(),
                    ),
                }
            }
            _ => panic!(),
        };

        Ok(ctx.store_expr(expr))
    }

    pub fn accessed_idents(&self, ctx: &Ctx) -> Vec<Ident> {
        match self.kind {
            ExprKind::Ident(ident) => vec![ident],
            ExprKind::Bin(_, lhs, rhs) => ctx
                .expr(lhs)
                .accessed_idents(ctx)
                .into_iter()
                .chain(ctx.expr(rhs).accessed_idents(ctx).into_iter())
                .collect(),
            ExprKind::Ret(expr) => expr
                .map(|e| ctx.expr(e).accessed_idents(ctx))
                .unwrap_or_default(),
            _ => Vec::new(),
        }
    }

    pub fn constrain(&self, ctx: &mut Ctx) {
        match &self.kind {
            ExprKind::Ident(ident) => {
                for other in ctx.ty_ctx.get_vars(ident.id).to_vec().iter() {
                    ctx.ty_ctx.constrain(
                        *other,
                        Constraint {
                            span: ident.span,
                            kind: ConstraintKind::Equate(self.ty),
                        },
                    );

                    ctx.ty_ctx.constrain(
                        self.ty,
                        Constraint {
                            span: ident.span,
                            kind: ConstraintKind::Equate(*other),
                        },
                    );
                }
            }
            ExprKind::Lit(_) => ctx.ty_ctx.constrain(
                self.ty,
                Constraint {
                    span: self.span,
                    kind: ConstraintKind::Arch(Arch::Int),
                },
            ),
            ExprKind::Bin(_, lhs, rhs) => {
                ctx.expr(*lhs).constrain(ctx);
                ctx.expr(*rhs).constrain(ctx);
            }
            ExprKind::Ret(expr) => {
                if let Some(expr) = expr {
                    ctx.expr(*expr).constrain(ctx);
                }
            }
            ExprKind::Call(sig) => {
                ctx.ty_ctx.constrain(
                    self.ty,
                    Constraint {
                        span: sig.span,
                        kind: ConstraintKind::Abs(sig.ty),
                    },
                );
            }
        }
    }

    fn is_int(&self, ctx: &Ctx) -> bool {
        match self.kind {
            ExprKind::Lit(id) => {
                if !ctx.lits.lit(id).is_int() {
                    return false;
                }
            }
            ExprKind::Call(sig) => {
                if !sig.ty.is_int() {
                    return false;
                }
            }
            ExprKind::Bin(_, lhs, rhs) => {
                if !ctx.expr(lhs).is_int(ctx) && ctx.expr(rhs).is_int(ctx) {
                    return false;
                }
            }
            _ => return false,
        }

        true
    }

    pub fn hint_satisfies(&self, ctx: &Ctx, ty: Ty) -> bool {
        match self.kind {
            ExprKind::Ident(ident) => {
                if let Some(vars) = ctx.ty_ctx.try_get_vars(ident.id) {
                    vars.iter().all(|v| ctx.ty_ctx.hint_satisfies(*v, ty))
                } else {
                    false
                }
            }
            ExprKind::Lit(id) => ctx.lits.lit(id).satisfies(ty),
            ExprKind::Bin(_, lhs, rhs) => {
                ctx.expr(lhs).hint_satisfies(ctx, ty) && ctx.expr(rhs).hint_satisfies(ctx, ty)
            }
            ExprKind::Call(sig) => sig.ty == ty,

            ExprKind::Ret(expr) => {
                if let Some(expr) = expr {
                    ctx.expr(expr).hint_satisfies(ctx, ty)
                } else {
                    ty.is_unit()
                }
            }
        }
    }
}

pub fn validate_bin_ops<'a>(ctx: &Ctx<'a>) -> Result<(), Diag<'a>> {
    for expr in ctx.exprs.iter() {
        match expr.kind {
            ExprKind::Bin(op, lhs, rhs) => {
                // TODO: operators should desugar into functions with traits
                if op.kind.is_algebraic() {
                    let lhs = ctx.expr(lhs);
                    let rhs = ctx.expr(rhs);

                    if !lhs.is_int(ctx) {
                        println!("{:#?}", expr.span);
                        return Err(ctx.errors(
                            "invalid arithmetic operation",
                            [
                                Msg::error(op.span, format!("cannot {} terms", op.kind.as_str())),
                                Msg::note(lhs.span, "term is not an integer"),
                            ]
                            .into_iter(),
                        ));
                    }

                    if !rhs.is_int(ctx) {
                        println!("{:#?}", expr.span);
                        return Err(ctx.errors(
                            "invalid arithmetic operation",
                            [
                                Msg::error(op.span, format!("cannot {} terms", op.kind.as_str())),
                                Msg::note(rhs.span, "term is not an integer"),
                            ]
                            .into_iter(),
                        ));
                    }
                }
            }
            _ => {}
        }
    }

    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub enum BinOpKind {
    Add,
    AddAssign,
    Multiply,
}

impl BinOpKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::AddAssign => "add assign",
            Self::Multiply => "multiply",
        }
    }

    pub fn is_algebraic(&self) -> bool {
        match self {
            Self::AddAssign | Self::Add | Self::Multiply => true,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ExprId(usize);

#[derive(Debug, Default)]
pub struct ExprStore {
    exprs: Vec<Expr>,
}

impl ExprStore {
    pub fn store(&mut self, expr: Expr) -> ExprId {
        let idx = self.exprs.len();
        self.exprs.push(expr);
        ExprId(idx)
    }

    #[track_caller]
    pub fn expr(&self, id: ExprId) -> &Expr {
        &self.exprs.get(id.0).expect("invalid expr id")
    }

    pub fn iter(&self) -> impl Iterator<Item = &Expr> {
        self.exprs.iter()
    }
}
