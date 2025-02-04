use self::sig::Sig;
use self::ty::TyErr;
use crate::diagnostic::{self, Diag, Msg};
use crate::ir::ctx::Ctx;
use crate::ir::ident::Ident;
use crate::ir::lit::Lit;
use crate::ir::ty::{Arch, Constraint, ConstraintKind, Ty, TyCtx, TyVar, TypeKey};
use crate::lex::buffer::TokenBuffer;
use crate::lex::buffer::{Span, TokenQuery};
use crate::parse::rules::prelude as rules;
use crate::parse::Item;

pub mod ctx;
pub mod ident;
pub mod lit;
pub mod sig;
pub mod ty;

pub const SYM_DEF: &str = "undefined symbol";

pub fn lower<'a>(tokens: &'a TokenBuffer<'a>, items: &[Item]) -> Ctx<'a> {
    let mut ctx = Ctx::new(tokens);

    match items
        .iter()
        .filter_map(|i| match i {
            Item::Struct(strukt) => Some(strukt),
            _ => None,
        })
        .map(|_| Ok(()))
        .collect::<Result<(), _>>()
    {
        Ok(_) => {}
        Err(diag) => {
            diagnostic::report(diag);
            panic!()
        }
    }

    items
        .iter()
        .filter_map(|i| match i {
            Item::Func(func) => Some(func),
            _ => None,
        })
        .for_each(|f| {
            let sig = func_sig(&mut ctx, f);
            ctx.store_sig(sig);
        });

    match items
        .iter()
        .filter_map(|i| match i {
            Item::Func(func) => Some(func),
            _ => None,
        })
        .map(|f| func(&mut ctx, f))
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(funcs) => {
            ctx.store_funcs(funcs);
        }
        Err(diag) => {
            diagnostic::report(diag);
            panic!()
        }
    }

    ctx
}

#[derive(Debug, Clone)]
pub struct Func {
    pub sig: Sig,
    pub block: Block,
}

fn func_sig<'a>(ctx: &mut Ctx<'a>, func: &rules::Func) -> Sig {
    Sig {
        span: func.span,
        ident: ctx.store_ident(func.name).id,
        ty: func
            .ty
            .map(|ty| ctx.ty(ty).expect("unregistered type"))
            .unwrap_or_else(|| Ty::Unit),
    }
}

fn func<'a>(ctx: &mut Ctx<'a>, func: &rules::Func) -> Result<Func, Diag<'a>> {
    Ok(Func {
        block: block(ctx, &func.block)?,
        sig: func_sig(ctx, func),
    })
}

#[derive(Debug, Clone)]
pub struct Block {
    pub span: Span,
    pub stmts: Vec<Stmt>,
    pub end: Option<OpenStmt>,
}

fn block<'a>(ctx: &mut Ctx<'a>, block: &rules::Block) -> Result<Block, Diag<'a>> {
    let mut stmts = block
        .stmts
        .iter()
        .map(|st| stmt(ctx, st))
        .collect::<Result<Vec<_>, _>>()?;

    let end = if let Some(last) = stmts.last() {
        match last {
            Stmt::Open(end) => {
                let end = end.clone();
                stmts.pop();
                Some(end)
            }
            _ => None,
        }
    } else {
        None
    };

    Ok(Block {
        span: block.span,
        stmts,
        end,
    })
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Semi(SemiStmt),
    Open(OpenStmt),
}

#[derive(Debug, Clone)]
pub enum OpenStmt {
    Ident(Ident),
    Lit(Lit),
    Bin(BinOp),
}

#[derive(Debug, Clone)]
pub enum SemiStmt {
    Let(Let),
    Assign(Assign),
    Ret(Return),
    Bin(BinOp),
}

fn stmt<'a>(ctx: &mut Ctx<'a>, stmt: &rules::Stmt) -> Result<Stmt, Diag<'a>> {
    Ok(match stmt {
        rules::Stmt::Let { name, ty, assign } => {
            let ty = if let Some(ty) = ty {
                Some((
                    ctx.span(*ty),
                    ctx.ty(*ty).ok_or_else(|| {
                        ctx.error(
                            SYM_DEF,
                            ctx.span(*ty),
                            format!("`{}` is not a type, expected a type", ctx.ident(*ty)),
                        )
                    })?,
                ))
            } else {
                None
            };

            Stmt::Semi(SemiStmt::Let(Let {
                span: ctx.span(*name),
                lhs: let_target(ctx, &rules::Expr::Ident(*name)),
                rhs: let_expr(ctx, assign)?,
                ty,
            }))
        }
        rules::Stmt::Semi(expr) => match expr {
            rules::Expr::Assign(assign) => Stmt::Semi(SemiStmt::Assign(Assign {
                span: assign.assign_span,
                kind: assign.kind,
                lhs: assign_target(ctx, &assign.lhs),
                rhs: assign_expr(ctx, &assign.rhs)?,
            })),
            rules::Expr::Ret(span, expr) => Stmt::Semi(SemiStmt::Ret(Return {
                span: *span,
                expr: ret_expr(ctx, expr.as_deref()).unwrap(),
            })),
            rules::Expr::Bin(_, _, _) => Stmt::Semi(SemiStmt::Bin(bin_op(ctx, expr)?)),
            expr => todo!("{expr:#?}"),
        },
        rules::Stmt::Open(expr) => match expr {
            rules::Expr::Ident(ident) => Stmt::Open(OpenStmt::Ident(ctx.store_ident(*ident))),
            rules::Expr::Lit(lit) => Stmt::Open(OpenStmt::Lit(ctx.store_int(*lit))),
            rules::Expr::Bin(_, _, _) => Stmt::Open(OpenStmt::Bin(bin_op(ctx, expr)?)),
            _ => todo!(),
        },
    })
}

#[derive(Debug, Clone)]
pub struct Let {
    pub span: Span,
    pub ty: Option<(Span, Ty)>,
    pub lhs: LetTarget,
    pub rhs: LetExpr,
}

#[derive(Debug, Clone, Copy)]
pub enum LetTarget {
    Ident(Ident),
}

#[derive(Debug, Clone)]
pub enum LetExpr {
    Ident(Ident),
    Lit(Lit),
    Bin(BinOp),
}

fn let_target<'a>(ctx: &mut Ctx<'a>, expr: &rules::Expr) -> LetTarget {
    match expr {
        rules::Expr::Ident(ident) => LetTarget::Ident(ctx.store_ident(*ident)),
        _ => todo!(),
    }
}

fn let_expr<'a>(ctx: &mut Ctx<'a>, expr: &rules::Expr) -> Result<LetExpr, Diag<'a>> {
    Ok(match expr {
        rules::Expr::Ident(ident) => LetExpr::Ident(ctx.store_ident(*ident)),
        rules::Expr::Lit(lit) => LetExpr::Lit(ctx.store_int(*lit)),
        rules::Expr::Bin(_, _, _) => LetExpr::Bin(bin_op(ctx, expr)?),
        _ => todo!(),
    })
}

#[derive(Debug, Clone)]
pub struct BinOp {
    pub span: Span,
    pub kind: BinOpKind,
    pub lhs: BinOpExpr,
    pub rhs: BinOpExpr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOpKind {
    Add,
    Sub,
    Mul,
}

#[derive(Debug, Clone)]
pub enum BinOpExpr {
    Ident(Ident),
    Lit(Lit),
    Call(Call),
    Bin(Box<BinOp>),
}

fn bin_op<'a>(ctx: &mut Ctx<'a>, expr: &rules::Expr) -> Result<BinOp, Diag<'a>> {
    Ok(match expr {
        rules::Expr::Bin(op, lhs, rhs) => BinOp {
            span: op.span,
            kind: op.kind,
            lhs: bin_op_expr(ctx, lhs)?,
            rhs: bin_op_expr(ctx, rhs)?,
        },
        _ => todo!(),
    })
}

fn bin_op_expr<'a>(ctx: &mut Ctx<'a>, expr: &rules::Expr) -> Result<BinOpExpr, Diag<'a>> {
    Ok(match expr {
        rules::Expr::Ident(ident) => BinOpExpr::Ident(ctx.store_ident(*ident)),
        rules::Expr::Lit(lit) => BinOpExpr::Lit(ctx.store_int(*lit)),
        rules::Expr::Bin(_, _, _) => BinOpExpr::Bin(Box::new(bin_op(ctx, expr)?)),
        rules::Expr::Call { span, func } => {
            let id = ctx.store_ident(*func).id;
            BinOpExpr::Call(Call {
                span: *span,
                sig: *ctx.get_sig(id).ok_or_else(|| {
                    ctx.error(SYM_DEF, ctx.span(*func), "function is not defined")
                })?,
            })
        }
        _ => todo!(),
    })
}

#[derive(Debug, Clone)]
pub struct Assign {
    pub span: Span,
    pub kind: AssignKind,
    pub lhs: AssignTarget,
    pub rhs: AssignExpr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssignKind {
    Equals,
    Add,
}

#[derive(Debug, Clone)]
pub enum AssignTarget {
    Ident(Ident),
}

#[derive(Debug, Clone)]
pub enum AssignExpr {
    Ident(Ident),
    Lit(Lit),
    Bin(BinOp),
}

fn assign_target<'a>(ctx: &mut Ctx<'a>, expr: &rules::Expr) -> AssignTarget {
    match expr {
        rules::Expr::Ident(ident) => AssignTarget::Ident(ctx.store_ident(*ident)),
        //rules::Expr::Lit(lit) => AssignTarget::Lit(ctx.store_int(*lit)),
        _ => todo!(),
    }
}

fn assign_expr<'a>(ctx: &mut Ctx<'a>, expr: &rules::Expr) -> Result<AssignExpr, Diag<'a>> {
    Ok(match expr {
        rules::Expr::Ident(ident) => AssignExpr::Ident(ctx.store_ident(*ident)),
        rules::Expr::Lit(lit) => AssignExpr::Lit(ctx.store_int(*lit)),
        rules::Expr::Bin(_, _, _) => AssignExpr::Bin(bin_op(ctx, expr)?),
        _ => todo!(),
    })
}

#[derive(Debug, Clone)]
pub struct Return {
    pub span: Span,
    pub expr: RetExpr,
}

#[derive(Debug, Clone)]
pub enum RetExpr {
    Unit,
    Ident(Ident),
    Lit(Lit),
    Bin(BinOp),
}

fn ret_expr<'a>(ctx: &mut Ctx<'a>, expr: Option<&rules::Expr>) -> Result<RetExpr, Diag<'a>> {
    if let Some(expr) = expr {
        Ok(match expr {
            rules::Expr::Ident(ident) => RetExpr::Ident(ctx.store_ident(*ident)),
            rules::Expr::Lit(lit) => RetExpr::Lit(ctx.store_int(*lit)),
            rules::Expr::Bin(_, _, _) => RetExpr::Bin(bin_op(ctx, expr)?),
            _ => todo!(),
        })
    } else {
        Ok(RetExpr::Unit)
    }
}

#[derive(Debug, Clone)]
pub struct Call {
    pub span: Span,
    pub sig: Sig,
}

// TODO:
//
// garauntee:
//  - idents are declared before use
//  - no ident shadowing
//
// error reporting is terrible right now, sometimes the errors are actively bad
pub fn resolve_types<'a>(ctx: &Ctx<'a>) -> Result<TypeKey, Vec<Diag<'a>>> {
    let mut ty_ctx = TyCtx::default();
    //let sigs = funcs
    //    .iter()
    //    .map(|f| (f.sig.ident, f.sig))
    //    .collect::<HashMap<IdentId, Sig>>();

    for func in ctx.funcs.iter() {
        for stmt in func.block.stmts.iter() {
            match stmt {
                Stmt::Semi(semi) => match semi {
                    SemiStmt::Let(let_) => match let_.lhs {
                        LetTarget::Ident(ident) => {
                            let var = ty_ctx.var(ident.id);
                            let_expr_constrain(ctx, &mut ty_ctx, var, &let_.rhs);

                            if let Some((span, ty)) = let_.ty {
                                ty_ctx.constrain(
                                    var,
                                    Constraint {
                                        kind: ConstraintKind::Abs(ty),
                                        span,
                                    },
                                );
                            }
                        }
                    },
                    SemiStmt::Assign(assign) => match assign.lhs {
                        AssignTarget::Ident(ident) => {
                            let var = ty_ctx.get_var(ident.id);
                            assign_expr_constrain(ctx, &mut ty_ctx, var, &assign.rhs);
                        }
                    },
                    SemiStmt::Bin(bin) => bin_op_constrain_unkown(ctx, &mut ty_ctx, bin),
                    SemiStmt::Ret(ret) => match &ret.expr {
                        RetExpr::Unit => {
                            assert!(func.sig.ty.is_unit());
                        }
                        RetExpr::Lit(lit) => {
                            assert!(func.sig.ty.is_int());
                        }
                        RetExpr::Bin(bin) => {
                            let constraint = Constraint {
                                span: func.sig.span,
                                kind: ConstraintKind::Abs(func.sig.ty),
                            };

                            bin_op_expr_constrain_to(ctx, &mut ty_ctx, constraint, &bin.lhs)
                                .map_err(|d| vec![d])?;
                            bin_op_expr_constrain_to(ctx, &mut ty_ctx, constraint, &bin.rhs)
                                .map_err(|d| vec![d])?;
                        }
                        RetExpr::Ident(ident) => {
                            let var = ty_ctx.get_var(ident.id);
                            ty_ctx.constrain(
                                var,
                                Constraint {
                                    span: func.sig.span,
                                    kind: ConstraintKind::Abs(func.sig.ty),
                                },
                            );
                        }
                    },
                },
                Stmt::Open(_) => {
                    todo!();
                }
            }
        }

        if let Some(open) = &func.block.end {
            match open {
                OpenStmt::Ident(ident) => {
                    let var = ty_ctx.get_var(ident.id);
                    ty_ctx.constrain(
                        var,
                        Constraint {
                            span: func.sig.span,
                            kind: ConstraintKind::Abs(func.sig.ty),
                        },
                    );
                }
                OpenStmt::Lit(lit) => {
                    if !func.sig.ty.is_int() {
                        return Err(vec![ctx.errors(
                            "mismatched return type",
                            [
                                Msg::error(
                                    lit.span,
                                    format!("expected `{}`, got `int`", func.sig.ty.as_str()),
                                ),
                                Msg::note(func.block.span, "because of the signature"),
                            ],
                        )]);
                    }
                }
                OpenStmt::Bin(bin) => {
                    let constraint = Constraint {
                        span: func.sig.span,
                        kind: ConstraintKind::Abs(func.sig.ty),
                    };

                    bin_op_expr_constrain_to(ctx, &mut ty_ctx, constraint, &bin.lhs)
                        .map_err(|d| vec![d])?;
                    bin_op_expr_constrain_to(ctx, &mut ty_ctx, constraint, &bin.rhs)
                        .map_err(|d| vec![d])?;
                }
            }
        }
    }

    match ty_ctx.resolve(ctx) {
        Ok(key) => Ok(key),
        Err(errs) => Err(errs
            .into_iter()
            .map(|err| match err {
                TyErr::Arch(span, arch, ty) => ctx.error(
                    "mismatched types",
                    span,
                    format!("expected `{}`, got `{}`", ty.as_str(), arch.as_str()),
                ),
                TyErr::Abs(span) => ctx.error("mismatched types", span, format!("invalid type")),
                _ => todo!(),
            })
            .collect()),
    }
}

fn let_expr_constrain<'a>(ctx: &Ctx<'a>, ty_ctx: &mut TyCtx, var: TyVar, expr: &LetExpr) {
    match expr {
        LetExpr::Lit(lit) => {
            lit_constrain(ctx, ty_ctx, var, lit);
        }
        LetExpr::Bin(bin) => {
            bin_op_constrain(ctx, ty_ctx, var, bin);
        }
        LetExpr::Ident(_) => todo!(),
    }
}

fn assign_expr_constrain<'a>(ctx: &Ctx<'a>, ty_ctx: &mut TyCtx, var: TyVar, expr: &AssignExpr) {
    match expr {
        AssignExpr::Lit(lit) => {
            lit_constrain(ctx, ty_ctx, var, lit);
        }
        AssignExpr::Bin(bin) => {
            bin_op_constrain(ctx, ty_ctx, var, bin);
        }
        AssignExpr::Ident(_) => todo!(),
    }
}

fn bin_op_constrain<'a>(ctx: &Ctx<'a>, ty_ctx: &mut TyCtx, var: TyVar, bin: &BinOp) {
    bin_op_expr_constrain(ctx, ty_ctx, var, &bin.lhs);
    bin_op_expr_constrain(ctx, ty_ctx, var, &bin.rhs);
}

fn bin_op_expr_constrain<'a>(ctx: &Ctx<'a>, ty_ctx: &mut TyCtx, var: TyVar, bin: &BinOpExpr) {
    match bin {
        BinOpExpr::Lit(lit) => {
            lit_constrain(ctx, ty_ctx, var, lit);
        }
        BinOpExpr::Bin(bin) => {
            bin_op_constrain(ctx, ty_ctx, var, bin);
        }
        BinOpExpr::Ident(ident) => ident_constrain(ctx, ty_ctx, var, ident),
        BinOpExpr::Call(call) => {
            ty_ctx.constrain(
                var,
                Constraint {
                    span: call.span,
                    kind: ConstraintKind::Abs(call.sig.ty),
                },
            );
        }
    }
}

fn bin_op_expr_constrain_to<'a>(
    ctx: &Ctx<'a>,
    ty_ctx: &mut TyCtx,
    constraint: Constraint,
    bin: &BinOpExpr,
) -> Result<(), Diag<'a>> {
    match bin {
        BinOpExpr::Lit(lit) => {
            if constraint.kind.is_int().is_some_and(|i| !i) {
                return Err(ctx.error("invalid expression", lit.span, "mismatched types"));
            }
        }
        BinOpExpr::Bin(bin) => {
            bin_op_expr_constrain_to(ctx, ty_ctx, constraint, &bin.rhs)?;
            bin_op_expr_constrain_to(ctx, ty_ctx, constraint, &bin.lhs)?;
        }
        BinOpExpr::Ident(ident) => ident_constrain_to(ctx, ty_ctx, constraint, ident),
        BinOpExpr::Call(call) => {
            if constraint
                .kind
                .hint_satisfies(call.sig.ty)
                .is_some_and(|s| !s)
            {
                return Err(ctx.error("invalid expression", call.span, "mismatched types"));
            }
        }
    }

    Ok(())
}

/// Recursively descend binary op tree to find constraint targets.
fn bin_op_constrain_unkown<'a>(ctx: &Ctx<'a>, ty_ctx: &mut TyCtx, bin: &BinOp) {
    if let Some(target) = bin_op_expr_find_constrain_target(ctx, ty_ctx, &bin.lhs)
        .or_else(|| bin_op_expr_find_constrain_target(ctx, ty_ctx, &bin.rhs))
    {
        // finding one constraint target is fine, it will equate with other variables
        bin_op_constrain(ctx, ty_ctx, target, bin);
    }
}

fn bin_op_expr_find_constrain_target<'a>(
    ctx: &Ctx<'a>,
    ty_ctx: &mut TyCtx,
    bin: &BinOpExpr,
) -> Option<TyVar> {
    match bin {
        BinOpExpr::Bin(bin) => bin_op_expr_find_constrain_target(ctx, ty_ctx, &bin.lhs)
            .or_else(|| bin_op_expr_find_constrain_target(ctx, ty_ctx, &bin.rhs)),
        BinOpExpr::Ident(ident) => Some(ty_ctx.get_var(ident.id)),
        BinOpExpr::Lit(_) => None,
        BinOpExpr::Call(_) => None,
    }
}

fn ident_constrain<'a>(_ctx: &Ctx<'a>, ty_ctx: &mut TyCtx, var: TyVar, ident: &Ident) {
    ty_ctx.constrain(
        var,
        Constraint {
            span: ident.span,
            kind: ConstraintKind::Equate(ty_ctx.get_var(ident.id)),
        },
    );
}

fn ident_constrain_to<'a>(
    _ctx: &Ctx<'a>,
    ty_ctx: &mut TyCtx,
    constraint: Constraint,
    ident: &Ident,
) {
    let var = ty_ctx.get_var(ident.id);
    ty_ctx.constrain(var, constraint);
}

fn lit_constrain<'a>(_ctx: &Ctx<'a>, ty_ctx: &mut TyCtx, var: TyVar, lit: &Lit) {
    ty_ctx.constrain(
        var,
        Constraint {
            span: lit.span,
            kind: ConstraintKind::Arch(Arch::Int),
        },
    );
}
