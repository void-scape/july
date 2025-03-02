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

        if let Err(err) = constrain_block(ctx, &mut infer, &func.block, func.sig) {
            errors.push(err);
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

fn constrain_block<'a>(
    ctx: &Ctx<'a>,
    infer: &mut InferCtx,
    block: &Block<'a>,
    sig: &Sig<'a>,
) -> Result<(), Diag<'a>> {
    let mut errors = Vec::new();

    for stmt in block.stmts.iter() {
        match stmt {
            Stmt::Semi(semi) => {
                if let Err(diag) = constrain_semi_stmt(ctx, infer, semi, sig) {
                    errors.push(diag);
                }
            }
            Stmt::Open(expr) => {
                // TODO: do we have to check for invalid open statements here? Are those even
                // possible?
                if let Err(diag) = constrain_expr_no_var(ctx, infer, sig, expr) {
                    errors.push(diag);
                }
            }
        }
    }

    if let Some(end) = &block.end {
        if let Err(diag) = constrain_return_expr(ctx, infer, end, &sig) {
            errors.push(diag);
        }
    }

    if !errors.is_empty() {
        Err(Diag::bundle(errors))
    } else {
        Ok(())
    }
}

fn constrain_block_to<'a>(
    ctx: &Ctx<'a>,
    infer: &mut InferCtx,
    block: &Block<'a>,
    sig: &Sig<'a>,
    ty: TyId,
    source: Span,
) -> Result<(), Diag<'a>> {
    let mut errors = Vec::new();

    for stmt in block.stmts.iter() {
        match stmt {
            Stmt::Semi(semi) => {
                if let Err(diag) = constrain_semi_stmt(ctx, infer, semi, sig) {
                    errors.push(diag);
                }
            }
            Stmt::Open(expr) => {
                // TODO: do we have to check for invalid open statements here? Are those even
                // possible?
                if let Err(diag) = constrain_expr_no_var(ctx, infer, sig, expr) {
                    errors.push(diag);
                }
            }
        }
    }

    if let Some(end) = &block.end {
        if let Err(diag) = constrain_expr_to(ctx, infer, end, sig, ty, source) {
            errors.push(diag);
        }
    }

    if !errors.is_empty() {
        Err(Diag::bundle(errors))
    } else {
        Ok(())
    }
}

fn constrain_semi_stmt<'a>(
    ctx: &Ctx<'a>,
    infer: &mut InferCtx,
    stmt: &SemiStmt<'a>,
    sig: &Sig<'a>,
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
                            constrain_call_args(ctx, infer, sig, call)?;
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
                                constrain_bin_op(ctx, infer, sig, bin, var)?;
                            }
                        }
                        Expr::Lit(lit) => match lit.kind {
                            LitKind::Int(_) => {
                                infer.integral(var, lit.span);
                            }
                            _ => todo!(),
                        },
                        Expr::Bool(span, _) => {
                            infer.eq(var, ctx.tys.bool(), *span);
                        }
                        Expr::Enum(_) => todo!(),
                        Expr::If(_) => todo!(),
                        Expr::Block(_) => todo!(),
                    }
                }
            }
        },
        SemiStmt::Ret(r) => {
            if let Some(expr) = &r.expr {
                constrain_return_expr(ctx, infer, expr, sig)
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
                    constrain_expr(ctx, infer, sig, &assign.rhs, var)?;
                }
                None => {
                    return Err(ctx.undeclared(ident));
                }
            },
            AssignTarget::Field(field_bin) => {
                let (span, ty) = aquire_bin_field_ty(ctx, infer, field_bin)?;
                constrain_expr_to(ctx, infer, &assign.rhs, sig, ty, span)?;
            }
        },
        SemiStmt::Expr(expr) => constrain_expr_no_var(ctx, infer, sig, expr)?,
    }

    Ok(())
}

fn constrain_return_expr<'a>(
    ctx: &Ctx<'a>,
    infer: &mut InferCtx,
    expr: &Expr<'a>,
    sig: &Sig<'a>,
) -> Result<(), Diag<'a>> {
    match expr {
        Expr::Ident(ident) => match infer.get_var(ident.id) {
            Some(var) => {
                infer.eq(var, sig.ty, sig.span);
            }
            None => {
                return Err(ctx.undeclared(ident));
            }
        },
        Expr::Lit(lit) => match lit.kind {
            LitKind::Int(_) => {
                if !ctx.tys.ty(sig.ty).is_int() {
                    return Err(ctx.mismatch(lit.span, sig.ty, "an integer"));
                }
            }
            _ => todo!(),
        },
        Expr::Bin(bin) => {
            constrain_bin_op_to(ctx, infer, bin, sig, sig.ty, sig.span)?;
        }
        Expr::Call(call) => {
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
            constrain_call_args(ctx, infer, sig, call)?;
        }
        Expr::Struct(def) => {
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
        Expr::Bool(span, _) => {
            if sig.ty != ctx.tys.bool() {
                return Err(ctx.report_error(
                    span,
                    format!(
                        "invalid type: expected: `{}`, got `bool`",
                        sig.ty.ctx_fmt(ctx)
                    ),
                ));
            }
        }
        Expr::If(if_) => {
            constrain_return_expr(ctx, infer, if_.block, sig)?;
            if let Some(otherwise) = if_.otherwise {
                constrain_return_expr(ctx, infer, otherwise, sig)?;
            }

            constrain_expr_to(ctx, infer, if_.condition, sig, ctx.tys.bool(), if_.span).map_err(
                |_| ctx.report_error(if_.condition.span(), "mismatched types: expected a `bool`"),
            )?;
            constrain_expr_to(ctx, infer, if_.block, sig, sig.ty, if_.span)?;
            if let Some(otherwise) = if_.otherwise {
                constrain_expr_to(ctx, infer, otherwise, sig, sig.ty, if_.span)?;
            }
        }
        Expr::Block(blck) => {
            constrain_block_to(ctx, infer, blck, sig, sig.ty, sig.span)?;
        }
        Expr::Enum(_) => todo!(),
    }

    Ok(())
}

fn constrain_expr<'a>(
    ctx: &Ctx<'a>,
    infer: &mut InferCtx,
    sig: &Sig<'a>,
    expr: &Expr<'a>,
    var: TyVar,
) -> Result<(), Diag<'a>> {
    match &expr {
        Expr::Struct(def) => {
            infer.eq(var, ctx.tys.struct_ty_id(def.id), def.span);
        }
        Expr::Call(call @ Call { sig, span, .. }) => {
            infer.eq(var, sig.ty, *span);
            constrain_call_args(ctx, infer, sig, call)?;
        }
        Expr::Ident(other) => {
            let Some(other_ty) = infer.get_var(other.id) else {
                return Err(ctx.undeclared(other));
            };

            infer.var_eq(ctx, var, other_ty);
        }
        Expr::Bin(bin) => {
            constrain_bin_op(ctx, infer, sig, bin, var)?;
        }
        Expr::Lit(lit) => match lit.kind {
            LitKind::Int(_) => {
                infer.integral(var, lit.span);
            }
            _ => todo!(),
        },
        Expr::Bool(span, _) => {
            infer.eq(var, ctx.tys.bool(), *span);
        }
        Expr::Block(block) => {
            constrain_block(ctx, infer, block, sig)?;
        }
        Expr::If(if_) => {
            todo!()
        }
        Expr::Enum(_) => todo!(),
    }

    Ok(())
}

fn constrain_expr_no_var<'a>(
    ctx: &Ctx<'a>,
    infer: &mut InferCtx,
    sig: &Sig<'a>,
    expr: &Expr<'a>,
) -> Result<(), Diag<'a>> {
    match &expr {
        Expr::Bool(_, _) | Expr::Lit(_) | Expr::Struct(_) => {}
        Expr::Call(call) => {
            constrain_call_args(ctx, infer, sig, call)?;
        }
        Expr::Ident(other) => {
            if infer.get_var(other.id).is_none() {
                return Err(ctx.undeclared(other));
            };
        }
        Expr::Bin(bin) => {
            constrain_bin_op_no_var(ctx, infer, sig, bin)?;
        }
        Expr::Enum(_) => todo!(),
        Expr::If(if_) => {
            todo!();
            //constrain_expr_to(ctx, infer, &if_.expr, ctx.tys.bool(), if_.span)?;
        }
        Expr::Block(_) => todo!(),
    }

    Ok(())
}

fn constrain_expr_to<'a>(
    ctx: &Ctx<'a>,
    infer: &mut InferCtx,
    expr: &Expr<'a>,
    sig: &Sig<'a>,
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
            constrain_call_args(ctx, infer, sig, call)?;
        }
        Expr::Ident(other) => {
            let Some(var) = infer.get_var(other.id) else {
                return Err(ctx.undeclared(other));
            };

            infer.eq(var, ty, source);
        }
        Expr::Bin(bin) => {
            constrain_bin_op_to(ctx, infer, bin, sig, ty, source)?;
        }
        Expr::Lit(lit) => match lit.kind {
            LitKind::Int(_) => {
                if !ctx.tys.ty(ty).is_int() {
                    return Err(ctx.mismatch(lit.span, ty, "an integer"));
                }
            }
            _ => todo!(),
        },
        Expr::Bool(span, _) => {
            if ty != ctx.tys.bool() {
                return Err(ctx.mismatch(*span, ty, "bool"));
            }
        }
        Expr::If(if_) => {
            constrain_expr_to(ctx, infer, if_.block, sig, ty, source)?;
            if let Some(otherwise) = if_.otherwise {
                constrain_expr_to(ctx, infer, otherwise, sig, ty, source)?;
            }
        }
        Expr::Block(block) => constrain_block_to(ctx, infer, block, sig, ty, source)?,
        Expr::Enum(_) => todo!(),
    }

    Ok(())
}

fn constrain_bin_op<'a>(
    ctx: &Ctx<'a>,
    infer: &mut InferCtx,
    sig: &Sig<'a>,
    bin: &BinOp<'a>,
    var: TyVar,
) -> Result<(), Diag<'a>> {
    if bin.kind.is_field() {
        let (span, field_ty) = aquire_bin_field_ty(ctx, infer, &bin)?;
        infer.eq(var, field_ty, span);
    } else {
        constrain_expr(ctx, infer, sig, &bin.lhs, var)?;
        constrain_expr(ctx, infer, sig, &bin.rhs, var)?;
    }

    Ok(())
}

fn constrain_bin_op_no_var<'a>(
    ctx: &Ctx<'a>,
    infer: &mut InferCtx,
    sig: &Sig<'a>,
    bin: &BinOp<'a>,
) -> Result<(), Diag<'a>> {
    if !bin.kind.is_field() {
        constrain_expr_no_var(ctx, infer, sig, &bin.lhs)?;
        constrain_expr_no_var(ctx, infer, sig, &bin.rhs)?;
    }

    Ok(())
}

fn constrain_bin_op_to<'a>(
    ctx: &Ctx<'a>,
    infer: &mut InferCtx,
    bin: &BinOp<'a>,
    sig: &Sig<'a>,
    ty: TyId,
    source: Span,
) -> Result<(), Diag<'a>> {
    match bin.kind {
        BinOpKind::Field => {
            let (span, field_ty) = aquire_bin_field_ty(ctx, infer, bin)?;
            if ty != field_ty {
                return Err(ctx.mismatch(span, ty, field_ty));
            }
        }
        BinOpKind::Eq => {
            if ty != ctx.tys.bool() {
                return Err(ctx.mismatch(bin.span, ctx.tys.bool(), ty));
            }
        }
        _ => {
            constrain_expr_to(ctx, infer, &bin.lhs, sig, ty, source)?;
            constrain_expr_to(ctx, infer, &bin.rhs, sig, ty, source)?;
        }
    }

    Ok(())
}

fn constrain_unbounded_bin_op<'a>(
    ctx: &Ctx<'a>,
    infer: &mut InferCtx,
    sig: &Sig<'a>,
    bin: &BinOp<'a>,
) -> Result<(), Diag<'a>> {
    if !bin.kind.is_field() {
        if let Some(var) = find_ty_var_in_bin_op(ctx, infer, bin)? {
            constrain_bin_op(ctx, infer, sig, bin, var)?;
        } else if let Some((span, ty)) = find_ty_in_bin_op(ctx, infer, bin)? {
            constrain_bin_op_to(ctx, infer, bin, sig, ty, span)?;
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
        Expr::Bool(_, _) => Ok(None),
        Expr::If(_) => todo!(),
        Expr::Block(_) => todo!(),
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
        Expr::Bool(span, _) => Ok(Some((*span, ctx.tys.bool()))),
        Expr::Bin(bin) => find_ty_in_bin_op(ctx, infer, bin),
        Expr::Call(call) => Ok(Some((call.span, call.sig.ty))),
        Expr::Struct(def) => Ok(Some((def.span, ctx.tys.struct_ty_id(def.id)))),
        Expr::Enum(_) => unimplemented!(),
        Expr::Ident(_) | Expr::Lit(_) => Ok(None),
        Expr::If(_) => todo!(),
        Expr::Block(_) => todo!(),
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
        Ty::Struct(id) => id,
        Ty::Int(_) | Ty::Unit | Ty::Bool => {
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
                strukt = ctx.tys.strukt(id);
            }
            ty @ Ty::Int(_) | ty @ Ty::Unit | ty @ Ty::Bool => {
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
    sig: &Sig<'a>,
    call: &Call<'a>,
) -> Result<(), Diag<'a>> {
    let params = call.sig.params.len();
    let args = call.args.len();
    if params != args {
        return Err(err!(
            ctx,
            call.span,
            format!("expected `{}` arguments, got `{}`", params, args)
        )
        .wrap(note!(ctx, call.sig.span, "function defined here")));
    }

    for (expr, ty) in call.args.iter().zip(call.sig.params.iter()) {
        constrain_expr_to(ctx, infer, expr, sig, ty.ty, ty.ty_binding)?;
    }

    Ok(())
}
