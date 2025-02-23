use super::Stmt;
use super::*;
use crate::diagnostic::{Diag, Msg};
use crate::ir::ctx::{Ctx, CtxFmt};
use crate::ir::lit::LitKind;
use crate::ir::sig::Param;
use crate::ir::sig::Sig;
use crate::ir::ty::infer::InferCtx;
use crate::ir::ty::store::TyId;
use crate::ir::ty::Ty;
use crate::ir::ty::{TyVar, TypeKey};
use crate::lex::buffer::Span;
use crate::{err, note};

pub fn resolve_types<'a>(ctx: &mut Ctx<'a>) -> Result<TypeKey, Vec<Diag<'a>>> {
    let mut errors = Vec::new();
    let mut infer = InferCtx::default();

    for func in ctx.funcs.iter() {
        infer.for_func(func);
        init_params(&mut infer, func);

        for stmt in func.block.stmts.iter() {
            match stmt {
                Stmt::Semi(semi) => {
                    if let Err(diag) = constrain_semi_stmt(ctx, &mut infer, semi, &func.sig) {
                        errors.push(diag);
                    }
                }
                Stmt::Open(_) => {
                    unreachable!();
                }
            }
        }

        if let Some(end) = &func.block.end {
            if let Err(diag) = constrain_open_stmt(ctx, &mut infer, end, &func.sig) {
                errors.push(diag);
            }
        }

        match infer.finish(ctx) {
            Ok(_) => {}
            Err(diags) => errors.extend(diags),
        }
    }

    if errors.is_empty() {
        Ok(infer.into_key())
    } else {
        Err(errors)
    }
}

fn init_params(infer: &mut InferCtx, func: &Func) {
    for Param {
        ident,
        ty,
        ty_binding,
        ..
    } in func.sig.params.iter()
    {
        let var = infer.new_var(*ident);
        infer.eq(var, *ty, *ty_binding);
    }
}

fn constrain_semi_stmt<'a>(
    ctx: &Ctx<'a>,
    infer: &mut InferCtx,
    stmt: &SemiStmt,
    sig: &Sig,
) -> Result<(), Diag<'a>> {
    match stmt {
        SemiStmt::Let(let_) => match let_.lhs {
            LetTarget::Ident(ident) => {
                let var = infer.new_var(ident);

                if let Some((span, ty)) = let_.ty {
                    infer.eq(var, ty, span);
                } else {
                    match &let_.rhs {
                        Expr::Struct(def) => {
                            infer.eq(var, ctx.tys.struct_ty_id(def.id), def.span);
                        }
                        Expr::Call(call @ Call { sig, span, .. }) => {
                            infer.eq(var, sig.ty, *span);
                            constrain_call_args(ctx, infer, call)?;
                        }
                        Expr::Ident(other) => {
                            let Some(other_var) = infer.get_var(other.id) else {
                                return Err(ctx.undeclared(other));
                            };

                            infer.var_eq(ctx, var, other_var);
                        }
                        Expr::Bin(bin) => {
                            if bin.kind.is_field() {
                                let (_, ty) = aquire_bin_field_ty(ctx, infer, &bin)?;
                                infer.eq(var, ty, bin.span);
                            } else {
                                constrain_bin_op(ctx, infer, bin, var)?;
                            }
                        }
                        Expr::Lit(lit) => match ctx.expect_lit(lit.kind) {
                            LitKind::Int(_) => {
                                infer.integral(var, lit.span);
                            }
                            _ => todo!(),
                        },
                        Expr::Enum(_) => todo!(),
                    }
                }
            }
        },
        SemiStmt::Ret(r) => {
            if let Some(expr) = &r.expr {
                constrain_open_stmt(ctx, infer, expr, sig)
                    .map_err(|err| err.msg(Msg::note(sig.span, "from fn signature")))?;
            } else {
                if !ctx.tys.is_unit(sig.ty) {
                    return Err(ctx.errors(
                        "mismatched type",
                        [
                            Msg::error(
                                r.span,
                                format!("expected `{}`, got `()`", ctx.ty_str(sig.ty)),
                            ),
                            Msg::note(sig.span, "from fn signature"),
                        ],
                    ));
                }
            }
        }
        SemiStmt::Assign(assign) => match &assign.lhs {
            AssignTarget::Ident(ident) => match infer.get_var(ident.id) {
                Some(var) => {
                    constrain_expr(ctx, infer, &assign.rhs, var)?;
                }
                None => {
                    return Err(ctx.undeclared(ident));
                }
            },
            AssignTarget::Field(field_bin) => {
                let (span, ty) = aquire_bin_field_ty(ctx, infer, field_bin)?;
                constrain_expr_to(ctx, infer, &assign.rhs, ty, span)?;
            }
        },
        SemiStmt::Bin(bin) => {
            constrain_unbounded_bin_op(ctx, infer, bin)?;
        }
        SemiStmt::Call(_) => {}
    }

    Ok(())
}

fn constrain_open_stmt<'a>(
    ctx: &Ctx<'a>,
    infer: &mut InferCtx,
    stmt: &OpenStmt,
    sig: &Sig,
) -> Result<(), Diag<'a>> {
    match stmt {
        OpenStmt::Ident(ident) => match infer.get_var(ident.id) {
            Some(var) => {
                infer.eq(var, sig.ty, sig.span);
            }
            None => {
                return Err(ctx.undeclared(ident));
            }
        },
        OpenStmt::Lit(lit) => match ctx.expect_lit(lit.kind) {
            LitKind::Int(_) => {
                if !ctx.tys.ty(sig.ty).is_int() {
                    return Err(ctx.mismatch(lit.span, sig.ty, "an integer"));
                }
            }
            _ => todo!(),
        },
        OpenStmt::Bin(bin) => {
            constrain_bin_op_to(ctx, infer, bin, sig.ty, sig.span)?;
        }
        OpenStmt::Call(call) => {
            if call.sig.ty != sig.ty {
                return Err(ctx.report_error(
                    call.span,
                    format!(
                        "invalid type: expected `{}`, got `{}`",
                        ctx.ty_str(sig.ty),
                        ctx.ty_str(call.sig.ty)
                    ),
                ));
            }
        }
        OpenStmt::Struct(def) => {
            if ctx.tys.struct_ty_id(def.id) != sig.ty {
                return Err(ctx.report_error(
                    def.span,
                    format!(
                        "invalid type: expected `{}`, got `{}`",
                        ctx.ty_str(sig.ty),
                        ctx.struct_name(def.id)
                    ),
                ));
            }
        }
    }

    Ok(())
}

fn constrain_expr<'a>(
    ctx: &Ctx<'a>,
    infer: &mut InferCtx,
    expr: &Expr,
    var: TyVar,
) -> Result<(), Diag<'a>> {
    match &expr {
        Expr::Struct(def) => {
            infer.eq(var, ctx.tys.struct_ty_id(def.id), def.span);
        }
        Expr::Call(call @ Call { sig, span, .. }) => {
            infer.eq(var, sig.ty, *span);
            constrain_call_args(ctx, infer, call)?;
        }
        Expr::Ident(other) => {
            let Some(other_ty) = infer.get_var(other.id) else {
                return Err(ctx.undeclared(other));
            };

            infer.var_eq(ctx, var, other_ty);
        }
        Expr::Bin(bin) => {
            constrain_bin_op(ctx, infer, bin, var)?;
        }
        Expr::Lit(lit) => match ctx.expect_lit(lit.kind) {
            LitKind::Int(_) => {
                infer.integral(var, lit.span);
            }
            _ => todo!(),
        },
        Expr::Enum(_) => todo!(),
    }

    Ok(())
}

fn constrain_expr_to<'a>(
    ctx: &Ctx<'a>,
    infer: &mut InferCtx,
    expr: &Expr,
    ty: TyId,
    source: Span,
) -> Result<(), Diag<'a>> {
    match &expr {
        Expr::Struct(def) => {
            let struct_ty = ctx.tys.struct_ty_id(def.id);
            if ty != struct_ty {
                return Err(ctx.mismatch(def.span, ty, struct_ty));
            }
        }
        Expr::Call(call @ Call { sig, span, .. }) => {
            if sig.ty != ty {
                return Err(ctx.mismatch(*span, ty, sig.ty));
            }
            constrain_call_args(ctx, infer, call)?;
        }
        Expr::Ident(other) => {
            let Some(var) = infer.get_var(other.id) else {
                return Err(ctx.undeclared(other));
            };

            infer.eq(var, ty, source);
        }
        Expr::Bin(bin) => {
            constrain_bin_op_to(ctx, infer, bin, ty, source)?;
        }
        Expr::Lit(lit) => match ctx.expect_lit(lit.kind) {
            LitKind::Int(_) => {
                if !ctx.tys.ty(ty).is_int() {
                    return Err(ctx.mismatch(lit.span, ty, "an integer"));
                }
            }
            _ => todo!(),
        },
        Expr::Enum(_) => todo!(),
    }

    Ok(())
}

fn constrain_bin_op<'a>(
    ctx: &Ctx<'a>,
    infer: &mut InferCtx,
    bin: &BinOp,
    var: TyVar,
) -> Result<(), Diag<'a>> {
    if bin.kind.is_field() {
        let (span, field_ty) = aquire_bin_field_ty(ctx, infer, &bin)?;
        infer.eq(var, field_ty, span);
    } else {
        constrain_expr(ctx, infer, &bin.lhs, var)?;
        constrain_expr(ctx, infer, &bin.rhs, var)?;
    }

    Ok(())
}

fn constrain_bin_op_to<'a>(
    ctx: &Ctx<'a>,
    infer: &mut InferCtx,
    bin: &BinOp,
    ty: TyId,
    source: Span,
) -> Result<(), Diag<'a>> {
    if bin.kind.is_field() {
        let (span, field_ty) = aquire_bin_field_ty(ctx, infer, bin)?;
        if ty != field_ty {
            return Err(ctx.mismatch(span, ty, field_ty));
        }
    } else {
        constrain_expr_to(ctx, infer, &bin.lhs, ty, source)?;
        constrain_expr_to(ctx, infer, &bin.rhs, ty, source)?;
    }

    Ok(())
}

fn constrain_unbounded_bin_op<'a>(
    ctx: &Ctx<'a>,
    infer: &mut InferCtx,
    bin: &BinOp,
) -> Result<(), Diag<'a>> {
    if !bin.kind.is_field() {
        if let Some(var) = find_ty_var_in_bin_op(ctx, infer, bin)? {
            constrain_bin_op(ctx, infer, bin, var)?;
        } else if let Some((span, ty)) = find_ty_in_bin_op(ctx, infer, bin)? {
            constrain_bin_op_to(ctx, infer, bin, ty, span)?;
        } else {
            if bin.is_integral(ctx).is_none_or(|i| !i) {
                return Err(ctx.report_error(bin.span, "expected every term to be an integer"));
            }
        }
    }

    Ok(())
}

fn find_ty_var_in_bin_op<'a>(
    ctx: &Ctx<'a>,
    infer: &mut InferCtx,
    bin: &BinOp,
) -> Result<Option<TyVar>, Diag<'a>> {
    if bin.kind.is_field() {
        Ok(None)
    } else {
        if let Some(var) = find_ty_var_in_expr(ctx, infer, &bin.lhs)? {
            Ok(Some(var))
        } else {
            find_ty_var_in_expr(ctx, infer, &bin.rhs)
        }
    }
}

fn find_ty_var_in_expr<'a>(
    ctx: &Ctx<'a>,
    infer: &mut InferCtx,
    expr: &Expr,
) -> Result<Option<TyVar>, Diag<'a>> {
    match expr {
        Expr::Ident(ident) => infer
            .get_var(ident.id)
            .map(|var| Some(var))
            .ok_or_else(|| ctx.undeclared(ident)),
        Expr::Bin(bin) => find_ty_var_in_bin_op(ctx, infer, bin),
        Expr::Call(_) | Expr::Struct(_) | Expr::Enum(_) | Expr::Lit(_) => Ok(None),
    }
}

fn find_ty_in_bin_op<'a>(
    ctx: &Ctx<'a>,
    infer: &mut InferCtx,
    bin: &BinOp,
) -> Result<Option<(Span, TyId)>, Diag<'a>> {
    if bin.kind.is_field() {
        let (span, ty) = aquire_bin_field_ty(ctx, infer, bin)?;
        Ok(Some((span, ty)))
    } else {
        if let Some(ty) = find_ty_in_expr(ctx, infer, &bin.lhs)? {
            Ok(Some(ty))
        } else {
            find_ty_in_expr(ctx, infer, &bin.rhs)
        }
    }
}

fn find_ty_in_expr<'a>(
    ctx: &Ctx<'a>,
    infer: &mut InferCtx,
    expr: &Expr,
) -> Result<Option<(Span, TyId)>, Diag<'a>> {
    match expr {
        Expr::Bin(bin) => find_ty_in_bin_op(ctx, infer, bin),
        Expr::Call(call) => Ok(Some((call.span, call.sig.ty))),
        Expr::Struct(def) => Ok(Some((def.span, ctx.tys.struct_ty_id(def.id)))),
        Expr::Enum(_) => unimplemented!(),
        Expr::Ident(_) | Expr::Lit(_) => Ok(None),
    }
}

#[track_caller]
pub fn aquire_bin_field_ty<'a>(
    ctx: &Ctx<'a>,
    infer: &InferCtx,
    bin: &BinOp,
) -> Result<(Span, TyId), Diag<'a>> {
    assert_eq!(bin.kind, BinOpKind::Field);

    let mut accesses = Vec::new();
    descend_bin_op_field(ctx, bin, &mut accesses);

    let var = accesses.first().unwrap();
    let ty_var = infer.get_var(var.id).ok_or_else(|| {
        ctx.report_error(
            var.span,
            format!("`{}` is undeclared", ctx.expect_ident(var.id)),
        )
    })?;
    let Some(ty) = infer.guess_var_ty(ctx, ty_var) else {
        return Err(ctx.report_error(
            bin.span,
            format!("failed to infer type of `{}`", infer.var_ident(ctx, ty_var)),
        ));
    };
    let ty = ctx.tys.ty(ty);

    let id = match ty {
        Ty::Struct(id) => *id,
        Ty::Int(_) | Ty::Unit => {
            return Err(ctx.report_error(
                bin.span,
                format!(
                    "invalid access: `{}` is of type `{}`, which has no fields",
                    var.ctx_fmt(ctx),
                    ty.ctx_fmt(ctx)
                ),
            ));
        }
    };
    let mut strukt = ctx.tys.strukt(id);

    for (i, access) in accesses.iter().skip(1).enumerate() {
        let ty = strukt.field_ty(access.id);
        if i == accesses.len() - 2 {
            return Ok((access.span, ty));
        }

        match ctx.tys.ty(ty) {
            Ty::Struct(id) => {
                strukt = ctx.tys.strukt(*id);
            }
            ty @ Ty::Int(_) | ty @ Ty::Unit => {
                return Err(ctx.report_error(
                    access,
                    format!(
                        "invalid field access, `{}` is of type `{}`, which has no fields",
                        access.ctx_fmt(ctx),
                        ty.ctx_fmt(ctx)
                    ),
                ));
            }
        }
    }

    // there must be atleast one access, since bin is of type field
    unreachable!()
}

fn constrain_call_args<'a>(
    ctx: &Ctx<'a>,
    infer: &mut InferCtx,
    call: &Call,
) -> Result<(), Diag<'a>> {
    let params = call.sig.params.len();
    let args = call.args.args.len();
    if params != args {
        return Err(err!(
            ctx,
            call.span,
            format!("expected `{}` arguments, got `{}`", params, args)
        )
        .wrap(note!(ctx, call.sig.span, "function defined here")));
    }

    for (expr, ty) in call.args.args.iter().zip(call.sig.params.iter()) {
        constrain_expr_to(ctx, infer, expr, ty.ty, ty.ty_binding)?;
    }

    Ok(())
}
