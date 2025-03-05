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

    // TODO: do better
    let funcs = std::mem::take(&mut ctx.funcs);
    for func in funcs.iter() {
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

    ctx.funcs = funcs;

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
    ctx: &mut Ctx<'a>,
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
                if let Err(diag) = constrain_expr_no_var(ctx, infer, sig, expr) {
                    errors.push(diag);
                }
            }
        }
    }

    if let Some(end) = &block.end {
        if let Err(diag) = constrain_expr_to(ctx, infer, end, &sig, sig.ty, sig.span) {
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
    ctx: &mut Ctx<'a>,
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
    ctx: &mut Ctx<'a>,
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
                            constrain_bin_op(ctx, infer, sig, bin, var)?;
                        }
                        Expr::Access(access) => {
                            let (_, ty) = aquire_access_ty(ctx, infer, access)?;
                            infer.eq(var, ty, access.span);
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
                        Expr::Str(span, _) => {
                            infer.eq(var, ctx.tys.str_lit(), *span);
                        }
                        Expr::Ref(_) => {}
                        Expr::Enum(_) => todo!(),
                        Expr::If(_) => todo!(),
                        Expr::Block(_) => todo!(),
                        Expr::Loop(_) => todo!(),
                    }
                }
            }
        },
        SemiStmt::Ret(r) => {
            if let Some(expr) = &r.expr {
                constrain_expr_to(ctx, infer, expr, sig, sig.ty, sig.span)?;
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
            AssignTarget::Access(access) => {
                let (span, ty) = aquire_access_ty(ctx, infer, access)?;
                constrain_expr_to(ctx, infer, &assign.rhs, sig, ty, span)?;
            }
        },
        SemiStmt::Expr(expr) => constrain_expr_no_var(ctx, infer, sig, expr)?,
    }

    Ok(())
}

// TODO: this is not well defined, should just use
//fn constrain_return_expr<'a>(
//    ctx: &mut Ctx<'a>,
//    infer: &mut InferCtx,
//    expr: &Expr<'a>,
//    sig: &Sig<'a>,
//) -> Result<(), Diag<'a>> {
//    match expr {
//        Expr::Ref(ref_) => {
//            constrain_return_expr(ctx, infer, ref_.inner, sig)?;
//        }
//        Expr::Ident(ident) => match infer.get_var(ident.id) {
//            Some(var) => {
//                infer.eq(var, sig.ty, sig.span);
//            }
//            None => {
//                return Err(ctx.undeclared(ident));
//            }
//        },
//        Expr::Lit(lit) => match lit.kind {
//            LitKind::Int(_) => {
//                if !ctx.tys.ty(sig.ty).is_int() {
//                    return Err(ctx.mismatch(lit.span, sig.ty, "an integer"));
//                }
//            }
//            _ => todo!(),
//        },
//        Expr::Bin(bin) => {
//            constrain_bin_op_to(ctx, infer, bin, sig, sig.ty, sig.span)?;
//        }
//        Expr::Call(call) => {
//            if call.sig.ty != sig.ty {
//                return Err(ctx.report_error(
//                    call.span,
//                    format!(
//                        "invalid type: expected `{}`, got `{}`",
//                        ctx.ty_str(sig.ty),
//                        ctx.ty_str(call.sig.ty)
//                    ),
//                ));
//            }
//            constrain_call_args(ctx, infer, sig, call)?;
//        }
//        Expr::Struct(def) => {
//            if ctx.tys.struct_ty_id(def.id) != sig.ty {
//                return Err(ctx.report_error(
//                    def.span,
//                    format!(
//                        "invalid type: expected `{}`, got `{}`",
//                        ctx.ty_str(sig.ty),
//                        ctx.struct_name(def.id)
//                    ),
//                ));
//            }
//        }
//        Expr::Bool(span, _) => {
//            if sig.ty != ctx.tys.bool() {
//                return Err(ctx.mismatch(*span, sig.ty, "bool"));
//            }
//        }
//        Expr::Str(span, _) => {
//            if sig.ty != ctx.tys.str_lit() {
//                return Err(ctx.mismatch(*span, sig.ty, "str"));
//            }
//        }
//        Expr::If(if_) => {
//            constrain_return_expr(ctx, infer, if_.block, sig)?;
//            if let Some(otherwise) = if_.otherwise {
//                constrain_return_expr(ctx, infer, otherwise, sig)?;
//            }
//
//            constrain_expr_to(ctx, infer, if_.condition, sig, ctx.tys.bool(), if_.span).map_err(
//                |_| ctx.report_error(if_.condition.span(), "mismatched types: expected a `bool`"),
//            )?;
//            constrain_expr_to(ctx, infer, if_.block, sig, sig.ty, if_.span)?;
//            if let Some(otherwise) = if_.otherwise {
//                constrain_expr_to(ctx, infer, otherwise, sig, sig.ty, if_.span)?;
//            }
//        }
//        Expr::Block(blck) => {
//            constrain_block_to(ctx, infer, blck, sig, sig.ty, sig.span)?;
//        }
//        Expr::Loop(block) => {
//            validate_loop_block(ctx, sig, block)?;
//
//            for stmt in block.stmts.iter() {
//                if let Stmt::Semi(SemiStmt::Ret(ret)) = stmt {
//                    if let Some(expr) = &ret.expr {
//                        constrain_expr_to(ctx, infer, expr, sig, sig.ty, sig.span)?;
//                    }
//                }
//            }
//
//            if sig.ty. {
//                // TODO: ctx.expr_ty(end), or something
//                return Err(ctx.report_error(end.span(), "mismatched types: expected `()`"));
//            }
//        }
//        Expr::Enum(_) => todo!(),
//    }
//
//    Ok(())
//}

// TODO: this sort of thing should be automated.
//
// Have a function like `verify_all_returns`, and `verify_openness`
fn validate_loop_block<'a>(
    ctx: &Ctx<'a>,
    sig: &Sig<'a>,
    block: &Block<'a>,
) -> Result<(), Diag<'a>> {
    if let Some(end) = block.end {
        // TODO: ctx.expr_ty(end), or something
        return Err(ctx.report_error(end.span(), "mismatched types: expected `()`"));
    }

    for stmt in block.stmts.iter() {
        if let Stmt::Semi(SemiStmt::Ret(ret)) = stmt {
            match ret.expr {
                None => {
                    if !ctx.tys.is_unit(sig.ty) {
                        return Err(ctx.mismatch(ret.span, sig.ty, TyId::UNIT));
                    }
                }
                Some(_) => {
                    if ctx.tys.is_unit(sig.ty) {
                        // TODO: ctx.expr_ty(end), or something
                        return Err(ctx.report_error(ret.span, "mismatched types: expected `()`"));
                    }
                }
            }
        }
    }

    Ok(())
}

fn constrain_expr<'a>(
    ctx: &mut Ctx<'a>,
    infer: &mut InferCtx,
    sig: &Sig<'a>,
    expr: &Expr<'a>,
    var: TyVar,
) -> Result<(), Diag<'a>> {
    match &expr {
        Expr::Ref(ref_) => {
            constrain_expr(ctx, infer, sig, ref_.inner, var)?;
        }
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
        Expr::Str(span, _) => {
            infer.eq(var, ctx.tys.str_lit(), *span);
        }
        Expr::Block(block) => {
            constrain_block(ctx, infer, block, sig)?;
        }
        Expr::Access(access) => {
            let (span, ty) = aquire_access_ty(ctx, infer, access)?;
            infer.eq(var, ty, span);
        }
        Expr::If(_) => {
            todo!()
        }
        Expr::Enum(_) => todo!(),
        Expr::Loop(_) => todo!(),
    }

    Ok(())
}

fn constrain_expr_no_var<'a>(
    ctx: &mut Ctx<'a>,
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
        Expr::Enum(_) | Expr::Str(_, _) => todo!(),
        Expr::If(if_) => {
            constrain_expr_to(ctx, infer, if_.condition, sig, TyId::BOOL, if_.span)?;
            constrain_expr_to(ctx, infer, if_.block, sig, TyId::UNIT, if_.span)?;
            if let Some(expr) = if_.otherwise {
                constrain_expr_to(ctx, infer, expr, sig, TyId::UNIT, if_.span)?;
            }
        }
        Expr::Access(access) => {
            _ = aquire_access_ty(ctx, infer, access)?;
        }
        Expr::Ref(_) => todo!(),
        Expr::Block(_) => todo!(),
        Expr::Loop(block) => constrain_loop_block(ctx, infer, block, sig)?,
    }

    Ok(())
}

fn constrain_expr_to<'a>(
    ctx: &mut Ctx<'a>,
    infer: &mut InferCtx,
    expr: &Expr<'a>,
    sig: &Sig<'a>,
    ty: TyId,
    source: Span,
) -> Result<(), Diag<'a>> {
    match &expr {
        Expr::Ref(ref_) => {
            let ty = ctx.tys.ty(ty);
            match ty {
                Ty::Ref(ty) => {
                    let ty = ctx.tys.ty_id(ty);
                    constrain_expr_to(ctx, infer, ref_.inner, sig, ty, source)?;
                }
                inner => {
                    return Err(ctx.mismatch(ref_.span, Ty::Ref(&ty), inner));
                }
            }
        }
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
            if bin.kind == BinOpKind::Eq && ty == TyId::BOOL {
                constrain_unbounded_bin_op(ctx, infer, sig, bin)?;
            } else {
                constrain_bin_op_to(ctx, infer, bin, sig, ty, source)?;
            }
        }
        Expr::Access(access) => {
            constrain_access_to(ctx, infer, access, ty)?;
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
        Expr::Str(span, _) => {
            if ty != ctx.tys.str_lit() {
                return Err(ctx.mismatch(*span, ty, "str"));
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
        Expr::Loop(block) => {
            constrain_loop_block(ctx, infer, block, sig)?;

            //if !ctx.tys.is_unit(sig.ty) {
            //    return Err(ctx
            //        .mismatch(block.span, ty, TyId::UNIT)
            //        .msg(Msg::error_span(source)));
            //}
        }
    }

    Ok(())
}

fn constrain_loop_block<'a>(
    ctx: &mut Ctx<'a>,
    infer: &mut InferCtx,
    block: &Block<'a>,
    sig: &Sig<'a>,
) -> Result<(), Diag<'a>> {
    validate_loop_block(ctx, sig, block)?;

    for stmt in block.stmts.iter() {
        if let Stmt::Semi(SemiStmt::Ret(ret)) = stmt {
            if let Some(expr) = &ret.expr {
                constrain_expr_to(ctx, infer, expr, sig, sig.ty, sig.span)?;
            }
        }
    }

    Ok(())
}

fn constrain_bin_op<'a>(
    ctx: &mut Ctx<'a>,
    infer: &mut InferCtx,
    sig: &Sig<'a>,
    bin: &BinOp<'a>,
    var: TyVar,
) -> Result<(), Diag<'a>> {
    constrain_expr(ctx, infer, sig, &bin.lhs, var)?;
    constrain_expr(ctx, infer, sig, &bin.rhs, var)?;
    Ok(())
}

fn constrain_bin_op_no_var<'a>(
    ctx: &mut Ctx<'a>,
    infer: &mut InferCtx,
    sig: &Sig<'a>,
    bin: &BinOp<'a>,
) -> Result<(), Diag<'a>> {
    constrain_expr_no_var(ctx, infer, sig, &bin.lhs)?;
    constrain_expr_no_var(ctx, infer, sig, &bin.rhs)?;
    Ok(())
}

fn constrain_access_to<'a>(
    ctx: &mut Ctx<'a>,
    infer: &mut InferCtx,
    access: &Access<'a>,
    ty: TyId,
) -> Result<(), Diag<'a>> {
    let (span, field_ty) = aquire_access_ty(ctx, infer, access)?;
    if ty != field_ty {
        return Err(ctx.mismatch(span, ty, field_ty));
    }

    Ok(())
}

#[track_caller]
fn constrain_bin_op_to<'a>(
    ctx: &mut Ctx<'a>,
    infer: &mut InferCtx,
    bin: &BinOp<'a>,
    sig: &Sig<'a>,
    ty: TyId,
    source: Span,
) -> Result<(), Diag<'a>> {
    constrain_expr_to(ctx, infer, &bin.lhs, sig, ty, source)?;
    constrain_expr_to(ctx, infer, &bin.rhs, sig, ty, source)?;
    Ok(())
}

fn constrain_unbounded_bin_op<'a>(
    ctx: &mut Ctx<'a>,
    infer: &mut InferCtx,
    sig: &Sig<'a>,
    bin: &BinOp<'a>,
) -> Result<(), Diag<'a>> {
    if let Some(var) = find_ty_var_in_bin_op(ctx, infer, bin)? {
        constrain_bin_op(ctx, infer, sig, bin, var)?;
    } else if let Some((span, ty)) = find_ty_in_bin_op(ctx, infer, bin)? {
        constrain_bin_op_to(ctx, infer, bin, sig, ty, span)?;
    } else {
        if bin.is_integral(ctx).is_none_or(|i| !i) {
            return Err(ctx.report_error(bin.span, "expected every term to be an integer"));
        }
    }
    Ok(())
}

fn find_ty_var_in_bin_op<'a>(
    ctx: &mut Ctx<'a>,
    infer: &mut InferCtx,
    bin: &BinOp,
) -> Result<Option<TyVar>, Diag<'a>> {
    if let Some(var) = find_ty_var_in_expr(ctx, infer, &bin.lhs)? {
        Ok(Some(var))
    } else {
        find_ty_var_in_expr(ctx, infer, &bin.rhs)
    }
}

fn find_ty_var_in_expr<'a>(
    ctx: &mut Ctx<'a>,
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
        Expr::Access(_) | Expr::Str(_, _) | Expr::Bool(_, _) => Ok(None),
        Expr::Ref(_) => todo!(),
        //Expr::Ref(inner) => find_ty_var_in_expr(ctx, infer, inner.inner),
        Expr::If(_) => todo!(),
        Expr::Block(_) => todo!(),
        Expr::Loop(_) => todo!(),
    }
}

fn find_ty_in_bin_op<'a>(
    ctx: &mut Ctx<'a>,
    infer: &mut InferCtx,
    bin: &BinOp,
) -> Result<Option<(Span, TyId)>, Diag<'a>> {
    if let Some(ty) = find_ty_in_expr(ctx, infer, &bin.lhs)? {
        Ok(Some(ty))
    } else {
        find_ty_in_expr(ctx, infer, &bin.rhs)
    }
}

fn find_ty_in_expr<'a>(
    ctx: &mut Ctx<'a>,
    infer: &mut InferCtx,
    expr: &Expr,
) -> Result<Option<(Span, TyId)>, Diag<'a>> {
    match expr {
        Expr::Bool(span, _) => Ok(Some((*span, ctx.tys.bool()))),
        Expr::Bin(bin) => find_ty_in_bin_op(ctx, infer, bin),
        Expr::Call(call) => Ok(Some((call.span, call.sig.ty))),
        Expr::Struct(def) => Ok(Some((def.span, ctx.tys.struct_ty_id(def.id)))),
        Expr::Ident(_) | Expr::Lit(_) => Ok(None),
        Expr::Str(span, _) => Ok(Some((*span, ctx.tys.str_lit()))),
        Expr::Access(access) => Ok(Some(aquire_access_ty(ctx, infer, access)?)),
        Expr::Ref(ref_) => Ok(find_ty_in_expr(ctx, infer, ref_.inner)?.map(|(_, ty)| {
            (
                ref_.span,
                ctx.tys.ty_id(&Ty::Ref(ctx.intern(ctx.tys.ty(ty)))),
            )
        })),
        Expr::Enum(_) => unimplemented!(),
        Expr::If(_) => todo!(),
        Expr::Block(_) => todo!(),
        Expr::Loop(_) => todo!(),
    }
}

pub fn aquire_access_ty<'a>(
    ctx: &Ctx<'a>,
    infer: &InferCtx,
    access: &Access,
) -> Result<(Span, TyId), Diag<'a>> {
    let ty_var = match access.lhs {
        Expr::Ident(ident) => infer
            .get_var(ident.id)
            .ok_or_else(|| ctx.undeclared(ident))?,
        _ => unimplemented!(),
    };

    let Some(ty) = infer.guess_var_ty(ctx, ty_var) else {
        return Err(ctx.report_error(
            access.lhs.span(),
            format!("failed to infer type of `{}`", infer.var_ident(ctx, ty_var)),
        ));
    };
    let ty = ctx.tys.ty(ty);

    let id = match ty.peel_refs().0 {
        Ty::Struct(id) => id,
        Ty::Int(_) | Ty::Unit | Ty::Bool | Ty::Ref(_) | Ty::Str => {
            return Err(ctx.report_error(
                access.lhs.span(),
                format!(
                    "invalid access: `{}` is of type `{}`, which has no fields",
                    match access.lhs {
                        Expr::Ident(ident) => ident.to_string(ctx),
                        _ => unreachable!(),
                    },
                    ty.to_string(ctx)
                ),
            ));
        }
    };
    let mut strukt = ctx.tys.strukt(*id);

    for (i, acc) in access.accessors.iter().enumerate() {
        let ty = strukt.field_ty(acc.id);
        if i == access.accessors.len() - 1 {
            return Ok((access.span, ty));
        }

        match ctx.tys.ty(ty) {
            Ty::Struct(id) => {
                strukt = ctx.tys.strukt(id);
            }
            Ty::Int(_) | Ty::Unit | Ty::Bool | Ty::Ref(_) | Ty::Str => {
                return Err(ctx.report_error(
                    acc,
                    format!(
                        "invalid field access, `{}` is of type `{}`, which has no fields",
                        acc.to_string(ctx),
                        ty.to_string(ctx)
                    ),
                ));
            }
        }
    }

    // there must be atleast one access, since bin is of type field
    unreachable!()
}

fn constrain_call_args<'a>(
    ctx: &mut Ctx<'a>,
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
