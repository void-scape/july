use super::ty::infer::Integral;
use super::Stmt;
use super::*;
use crate::air::IntKind;
use crate::diagnostic::{Diag, Msg};
use crate::ir::ctx::{Ctx, CtxFmt};
use crate::ir::sig::Param;
use crate::ir::sig::Sig;
use crate::ir::ty::infer::InferCtx;
use crate::ir::ty::store::TyId;
use crate::ir::ty::Ty;
use crate::ir::ty::{TyVar, TypeKey};
use crate::lex::buffer::Span;
use std::fmt::Debug;

pub fn resolve_types<'a>(ctx: &mut Ctx<'a>) -> Result<TypeKey, Diag<'a>> {
    let mut errors = Vec::new();
    let mut infer = InferCtx::default();

    for konst in ctx.tys.consts() {
        // place the global consts in highest scope
        let var = infer.new_var(konst.name);
        infer.eq(var, konst.ty, konst.span);
    }

    // TODO: do better
    let funcs = std::mem::take(&mut ctx.funcs);
    for func in funcs.iter().filter(|f| !f.is_intrinsic()) {
        if let Err(diag) = infer.in_scope(ctx, |ctx, infer| {
            init_params(infer, func);
            func.block.block_constrain(ctx, infer, func.sig)?;
            if let Some(end) = func.block.end {
                if let Err(diag) =
                    end.constrain_with(ctx, infer, func.sig, func.sig.ty, func.sig.span)
                {
                    errors.push(diag);
                }
            } else {
                if func.sig.ty != TyId::UNIT {
                    errors.push(ctx.mismatch(func.block.span, func.sig.ty, TyId::UNIT));
                }
            }
            Ok(())
        }) {
            errors.push(diag);
        }
    }

    ctx.funcs = funcs;

    if !errors.is_empty() {
        Err(Diag::bundle(errors))
    } else {
        infer.unify_top_scope(ctx).map(|_| infer.into_key())
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

impl<'a> Expr<'a> {
    /// Fails when:
    ///     aquiring field access type Fails
    ///     ident is undefined
    pub fn resolve_infer(&self, ctx: &mut Ctx<'a>, infer: &InferCtx) -> Result<InferTy, Diag<'a>> {
        Ok(match self {
            Self::Lit(lit) => match lit.kind {
                LitKind::Int(_) => InferTy::Int,
                LitKind::Float(_) => InferTy::Float,
            },
            Self::Ident(ident) => {
                let Some(var) = infer.var(ident.id) else {
                    return Err(ctx.undeclared(ident));
                };

                infer
                    .guess_var_ty(ctx, var)
                    .map(|t| InferTy::Ty(t))
                    .or_else(|| infer.is_var_integral_int(var).then_some(InferTy::Int))
                    .or_else(|| infer.is_var_integral_float(var).then_some(InferTy::Float))
                    .ok_or_else(|| ctx.report_error(ident.span, "could not infer type"))?
            }
            Self::Access(access) => InferTy::Ty(aquire_access_ty(ctx, infer, access)?.1),
            Self::Call(call) => InferTy::Ty(call.sig.ty),
            Self::Str(_) => InferTy::Ty(TyId::STR_LIT),
            Self::Bin(bin) => {
                let lhs = bin.lhs.resolve_infer(ctx, infer)?;
                let rhs = bin.lhs.resolve_infer(ctx, infer)?;

                //assert_eq!(lhs, rhs);
                match bin.kind {
                    BinOpKind::Add
                    | BinOpKind::Sub
                    | BinOpKind::Mul
                    | BinOpKind::Div
                    | BinOpKind::Xor => {
                        // I know this is not right, but I am too lazy
                        assert_eq!(lhs, rhs);
                        lhs
                    }
                    BinOpKind::Eq | BinOpKind::Ne => InferTy::Ty(TyId::BOOL),
                }
            }
            Self::Bool(_) => InferTy::Ty(TyId::BOOL),
            Self::IndexOf(index) => match index.array.resolve_infer(ctx, infer)? {
                InferTy::Ty(arr_ty) => match ctx.tys.ty(arr_ty) {
                    Ty::Array(_, inner) => InferTy::Ty(ctx.tys.get_ty_id(inner).unwrap()),
                    other => panic!("invalid array type for index of: {:?}", other),
                },
                other => panic!("invalid array type for index of: {:?}", other),
            },
            Self::Struct(def) => InferTy::Ty(ctx.tys.struct_ty_id(def.id)),
            Self::Block(block) => match block.end {
                None => InferTy::Ty(TyId::UNIT),
                Some(end) => end.resolve_infer(ctx, infer)?,
            },
            Self::For(_) | Self::Loop(_) => InferTy::Ty(TyId::UNIT),
            Self::If(if_) => {
                let block_infer = if_.block.resolve_infer(ctx, infer)?;
                if let Some(otherwise) = if_.otherwise {
                    let otherwise_infer = otherwise.resolve_infer(ctx, infer)?;
                    if block_infer != otherwise_infer {
                        return Err(ctx.report_error(if_.span, "branches return different types"));
                    }
                }
                block_infer
            }
            Self::Array(arr_def) => {
                if let Some(first) = arr_def.exprs.first() {
                    let infer_ty = first.resolve_infer(ctx, infer)?;
                    for expr in arr_def.exprs.iter().skip(1) {
                        let next_infer_ty = expr.resolve_infer(ctx, infer)?;
                        if next_infer_ty != infer_ty {
                            return Err(ctx.mismatch(
                                expr.span(),
                                infer_ty.to_string(ctx),
                                next_infer_ty.to_string(ctx),
                            ));
                        }
                    }

                    InferTy::Ty(match infer_ty {
                        InferTy::Ty(ty) => ctx
                            .tys
                            .ty_id(&Ty::Array(arr_def.exprs.len(), ctx.intern(ctx.tys.ty(ty)))),
                        InferTy::Int => ctx
                            .tys
                            .ty_id(&Ty::Array(arr_def.exprs.len(), &Ty::Int(IntTy::ISIZE))),
                        InferTy::Float => ctx
                            .tys
                            .ty_id(&Ty::Array(arr_def.exprs.len(), &Ty::Float(FloatTy::F64))),
                    })
                } else {
                    return Err(ctx.report_error(arr_def.span, "could not infer type of {array}"));
                }
            }
            Self::Unary(unary) => match unary.kind {
                UOpKind::Not => unary.inner.resolve_infer(ctx, infer)?,
                UOpKind::Ref => InferTy::Ty(match unary.inner.resolve_infer(ctx, infer)? {
                    InferTy::Ty(inner_ty) =>
                    // TODO: make ctx.tys.ty return a reference instead of interning
                    {
                        ctx.tys.ty_id(&Ty::Ref(&ctx.intern(ctx.tys.ty(inner_ty))))
                    }
                    InferTy::Int => ctx.tys.ty_id(&Ty::Ref(&Ty::Int(IntTy::ISIZE))),
                    InferTy::Float => ctx.tys.ty_id(&Ty::Ref(&Ty::Float(FloatTy::F64))),
                }),
                UOpKind::Deref => InferTy::Ty(match unary.inner.resolve_infer(ctx, infer)? {
                    InferTy::Ty(inner_ty) => match ctx.tys.ty(inner_ty) {
                        Ty::Ref(inner) => ctx.tys.ty_id(inner),
                        _ => panic!("invalid deref type"),
                    },
                    InferTy::Float | InferTy::Int => panic!("invalid deref type"),
                }),
            },
            Self::Cast(cast) => InferTy::Ty(cast.ty),
            ty => todo!("{ty:?}"),
        })
    }

    pub fn infer_equality(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        ty: TyId,
        source: Span,
    ) -> Result<(), Diag<'a>> {
        let span = self.span();
        match self.resolve_infer(ctx, infer)? {
            InferTy::Int => {
                if !ctx.tys.ty(ty).is_int() {
                    return Err(ctx
                        .mismatch(span, ty, "{int}")
                        .msg(Msg::help(source, "from this binding")));
                }
            }
            InferTy::Float => {
                if !ctx.tys.ty(ty).is_int() {
                    return Err(ctx
                        .mismatch(span, ty, "{float}")
                        .msg(Msg::help(source, "from this binding")));
                }
            }
            InferTy::Ty(infer_ty) => {
                if ty != infer_ty {
                    return Err(ctx
                        .mismatch(span, ty, infer_ty.to_string(ctx))
                        .msg(Msg::help(source, "from this binding")));
                }
            }
        }
        Ok(())
    }

    pub fn constrain_var(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        var: TyVar,
    ) -> Result<(), Diag<'a>> {
        let span = self.span();
        match self.resolve_infer(ctx, infer)? {
            InferTy::Int => infer.integral(Integral::Int, var, span),
            InferTy::Float => infer.integral(Integral::Float, var, span),
            InferTy::Ty(infer_ty) => infer.eq(var, infer_ty, span),
        }
        Ok(())
    }

    pub fn constrain_var_with(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        ty: TyId,
        source: Span,
        var: TyVar,
    ) -> Result<(), Diag<'a>> {
        let span = self.span();
        match self.resolve_infer(ctx, infer)? {
            InferTy::Int => {
                if !ctx.tys.ty(ty).is_int() {
                    return Err(ctx.mismatch(span, ty, "{int}"));
                }
                infer.integral(Integral::Int, var, source)
            }
            InferTy::Float => {
                if !ctx.tys.ty(ty).is_int() {
                    return Err(ctx.mismatch(span, ty, "{float}"));
                }
                infer.integral(Integral::Float, var, source)
            }
            InferTy::Ty(infer_ty) => {
                if ty != infer_ty {
                    return Err(ctx.mismatch(span, ty, infer_ty.to_string(ctx)));
                }
                infer.eq(var, ty, source)
            }
        }
        Ok(())
    }

    #[track_caller]
    pub fn constrain_with(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
        ty: TyId,
        source: Span,
    ) -> Result<(), Diag<'a>> {
        match self {
            Expr::Ident(ident) => {
                let var = infer.var(ident.id).ok_or_else(|| ctx.undeclared(ident))?;
                self.infer_equality(ctx, infer, ty, source)?;
                infer.eq(var, ty, source);

                Ok(())
            }
            Expr::Block(block) => match block.end {
                Some(end) => end.constrain_with(ctx, infer, sig, ty, source),
                None => {
                    if ty == TyId::UNIT {
                        Ok(())
                    } else {
                        Err(ctx.mismatch(block.span, ty, TyId::UNIT))
                    }
                }
            },
            Expr::If(if_) => {
                infer.in_scope(ctx, |ctx, infer| {
                    match if_.block {
                        Expr::Block(block) => block.block_constrain(ctx, infer, sig)?,
                        _ => unreachable!(),
                    }
                    if_.block.constrain_with(ctx, infer, sig, ty, source)
                })?;
                infer.in_scope(ctx, |ctx, infer| {
                    if ty != TyId::UNIT {
                        let otherwise = if_
                            .otherwise
                            .ok_or_else(|| ctx.report_error(if_.span, "missing `else` branch"))?;
                        match otherwise {
                            Expr::Block(block) => block.block_constrain(ctx, infer, sig)?,
                            _ => unreachable!(),
                        }
                        otherwise.constrain_with(ctx, infer, sig, ty, source)
                    } else if let Some(otherwise) = if_.otherwise {
                        match otherwise {
                            Expr::Block(block) => block.block_constrain(ctx, infer, sig)?,
                            _ => unreachable!(),
                        }
                        otherwise.constrain_with(ctx, infer, sig, ty, source)
                    } else {
                        Ok(())
                    }
                })
            }
            Expr::Lit(lit) => match lit.kind {
                LitKind::Int(_) => {
                    if ctx.tys.ty(ty).is_int() {
                        Ok(())
                    } else {
                        Err(ctx.mismatch(lit.span, ty, "{int}"))
                    }
                }
                LitKind::Float(_) => {
                    if ctx.tys.ty(ty).is_float() {
                        Ok(())
                    } else {
                        Err(ctx.mismatch(lit.span, ty, "{float}"))
                    }
                }
            },
            Expr::Bin(bin) => {
                match bin.kind {
                    BinOpKind::Eq | BinOpKind::Ne => {
                        assert!(!bin.kind.output_is_input());
                        if ty != TyId::BOOL {
                            Err(ctx.mismatch(bin.span, ty, TyId::BOOL))
                        } else {
                            let infer_lhs = bin.lhs.resolve_infer(ctx, infer)?;
                            let infer_rhs = bin.rhs.resolve_infer(ctx, infer)?;
                            if !infer_lhs.equiv(infer_rhs, ctx) {
                                Err(ctx.report_error(bin.span, "left hand side if of a different type than the right hand side"))
                            } else {
                                Ok(())
                            }
                        }
                    }
                    _ => {
                        assert!(bin.kind.output_is_input());
                        bin.lhs.constrain_with(ctx, infer, sig, ty, source)?;
                        bin.rhs.constrain_with(ctx, infer, sig, ty, source)
                    }
                }
            }
            _ => self.infer_equality(ctx, infer, ty, source),
        }
    }
}

pub trait Constrain<'a>: Debug {
    fn constrain(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
    ) -> Result<(), Diag<'a>>;
}

impl<'a> Constrain<'a> for Stmt<'a> {
    fn constrain(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
    ) -> Result<(), Diag<'a>> {
        match self {
            Self::Semi(semi) => semi.constrain(ctx, infer, sig),
            Self::Open(open) => open.constrain(ctx, infer, sig),
        }
    }
}

impl<'a> Constrain<'a> for SemiStmt<'a> {
    fn constrain(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
    ) -> Result<(), Diag<'a>> {
        match self {
            SemiStmt::Let(let_) => match let_.lhs {
                LetTarget::Ident(ident) => {
                    let var = infer.new_var(ident);

                    let_.rhs.constrain(ctx, infer, sig)?;
                    if let Some((span, ty)) = let_.ty {
                        let_.rhs.constrain_var_with(ctx, infer, ty, span, var)?;
                    }
                    let_.rhs.constrain_var(ctx, infer, var)?;
                }
            },
            SemiStmt::Ret(r) => {
                if let Some(expr) = &r.expr {
                    expr.constrain(ctx, infer, sig)?;
                    expr.infer_equality(ctx, infer, sig.ty, sig.span)?;
                } else if !ctx.tys.is_unit(sig.ty) {
                    return Err(ctx
                        .mismatch(r.span, sig.ty, TyId::UNIT)
                        .msg(Msg::help(sig.span, "from fn signature")));
                }
            }
            SemiStmt::Assign(assign) => {
                assign.lhs.constrain(ctx, infer, sig)?;
                assign.rhs.constrain(ctx, infer, sig)?;

                match assign.lhs.resolve_infer(ctx, infer)? {
                    InferTy::Ty(ty) => {
                        if assign.kind != AssignKind::Equals {
                            match ctx.tys.ty(ty) {
                                Ty::Int(_) | Ty::Float(_) => {}
                                other => {
                                    return Err(ctx.report_error(
                                        assign.span,
                                        format!(
                                            "cannot `{}` to a value of type `{}`",
                                            assign.kind.as_str(),
                                            other.to_string(ctx)
                                        ),
                                    ))
                                }
                            }
                        }
                    }
                    InferTy::Int | InferTy::Float => {
                        // assignable
                    }
                }
                if assign.rhs.is_unit(ctx) {
                    return Err(ctx.report_error(
                        assign.rhs.span(),
                        "cannot assign value to an expression of type `()`",
                    ));
                }
            }
            SemiStmt::Expr(expr) => expr.constrain(ctx, infer, sig)?,
        }

        Ok(())
    }
}

impl<'a> Constrain<'a> for Expr<'a> {
    fn constrain(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
    ) -> Result<(), Diag<'a>> {
        match self {
            Self::Lit(_) | Self::Str(_) | Self::Bool(_) => Ok(()),
            Self::Ident(ident) => ident.constrain(ctx, infer, sig),
            Self::Bin(bin) => bin.constrain(ctx, infer, sig),
            Self::Access(access) => access.constrain(ctx, infer, sig),
            Self::Struct(def) => def.constrain(ctx, infer, sig),
            Self::Enum(def) => def.constrain(ctx, infer, sig),
            Self::Call(call) => call.constrain(ctx, infer, sig),
            Self::Block(block) => block.constrain(ctx, infer, sig),
            Self::If(if_) => if_.constrain(ctx, infer, sig),
            Self::Loop(loop_) => loop_.constrain(ctx, infer, sig),
            Self::For(for_) => for_.constrain(ctx, infer, sig),
            Self::Unary(unary) => unary.constrain(ctx, infer, sig),
            Self::Array(arr) => arr.constrain(ctx, infer, sig),
            Self::IndexOf(index) => index.constrain(ctx, infer, sig),
            Self::Range(range) => range.constrain(ctx, infer, sig),
            Self::Cast(cast) => cast.constrain(ctx, infer, sig),
            Self::Continue(_) | Self::Break(_) => Ok(()),
        }
    }
}

impl<'a> Constrain<'a> for BinOp<'a> {
    fn constrain(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
    ) -> Result<(), Diag<'a>> {
        match self.kind {
            BinOpKind::Eq | BinOpKind::Ne => {
                assert!(!self.kind.output_is_input());
                self.lhs.constrain(ctx, infer, sig)?;
                self.rhs.constrain(ctx, infer, sig)?;

                let infer_lhs = self.lhs.resolve_infer(ctx, infer)?;
                let infer_rhs = self.rhs.resolve_infer(ctx, infer)?;
                if !infer_lhs.equiv(infer_rhs, ctx) {
                    Err(ctx.report_error(
                        self.span,
                        format!(
                            "operation terms are of different types: `{}` {} `{}`",
                            infer_lhs.to_string(ctx),
                            self.kind.as_str(),
                            infer_rhs.to_string(ctx)
                        ),
                    ))
                } else {
                    Ok(())
                }
            }
            kind => {
                assert!(self.kind.output_is_input(), "{kind:?}");
                self.lhs.constrain(ctx, infer, sig)?;
                self.rhs.constrain(ctx, infer, sig)
            }
        }
    }
}

impl<'a> Constrain<'a> for Access<'a> {
    fn constrain(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        _: &Sig<'a>,
    ) -> Result<(), Diag<'a>> {
        assert!(matches!(self.lhs, Expr::Ident(_)));
        let _ = aquire_access_ty(ctx, infer, self)?;
        Ok(())
    }
}

impl<'a> Constrain<'a> for StructDef<'a> {
    fn constrain(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
    ) -> Result<(), Diag<'a>> {
        for field_def in self.fields.iter() {
            let field_map = ctx.tys.fields(self.id);
            let field_ty = field_map.field_ty(field_def.name.id).unwrap();
            field_def
                .expr
                .constrain_with(ctx, infer, sig, field_ty, field_def.name.span)?;
        }

        Ok(())
    }
}

impl<'a> Constrain<'a> for EnumDef {
    fn constrain(
        &self,
        _ctx: &mut Ctx<'a>,
        _infer: &mut InferCtx,
        _sig: &Sig<'a>,
    ) -> Result<(), Diag<'a>> {
        todo!();
    }
}

impl<'a> Constrain<'a> for ArrDef<'a> {
    fn constrain(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
    ) -> Result<(), Diag<'a>> {
        for expr in self.exprs.iter() {
            expr.constrain(ctx, infer, sig)?;
        }

        Ok(())
    }
}

impl<'a> Constrain<'a> for Call<'a> {
    fn constrain(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
    ) -> Result<(), Diag<'a>> {
        let mut errors = Vec::new();
        let params = self.sig.params.len();
        let args = self.args.len();

        let name = ctx.expect_ident(self.sig.ident);
        if params != args && (name != "printf" || (name == "printf" && args == 0)) {
            errors.push(
                ctx.report_error(
                    self.span,
                    format!("expected `{}` arguments, got `{}`", params, args),
                )
                .msg(Msg::help(self.sig.span, "function defined here")),
            );
        }

        if name == "printf" {
            for expr in self.args.iter() {
                expr.constrain(ctx, infer, sig)?;
            }
        } else {
            for (expr, param) in self.args.iter().zip(self.sig.params.iter()) {
                expr.constrain(ctx, infer, sig)?;
                match expr {
                    Expr::Ident(ident) => match infer.var(ident.id) {
                        Some(var) => {
                            if ctx.expect_ident(ident.id) != "NULL" {
                                infer.eq(var, param.ty, param.span);
                            }
                        }
                        None => {
                            errors.push(ctx.undeclared(ident));
                        }
                    },
                    _ => {
                        if let Err(diag) = expr.infer_equality(ctx, infer, param.ty, param.span) {
                            errors.push(diag);
                        }
                    }
                }
            }
        }

        if !errors.is_empty() {
            Err(Diag::bundle(errors))
        } else {
            Ok(())
        }
    }
}

impl<'a> Block<'a> {
    fn block_constrain(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
    ) -> Result<(), Diag<'a>> {
        let mut errors = Vec::new();

        for stmt in self.stmts.iter() {
            if let Err(diag) = stmt.constrain(ctx, infer, sig) {
                errors.push(diag);
            }
        }

        if let Some(end) = &self.end {
            if let Err(diag) = end.constrain(ctx, infer, sig) {
                errors.push(diag);
            }
        }

        if !errors.is_empty() {
            Err(Diag::bundle(errors))
        } else {
            Ok(())
        }
    }
}

impl<'a> Constrain<'a> for Block<'a> {
    fn constrain(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
    ) -> Result<(), Diag<'a>> {
        infer.in_scope(ctx, |ctx, infer| self.block_constrain(ctx, infer, sig))
    }
}

impl<'a> Constrain<'a> for If<'a> {
    fn constrain(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
    ) -> Result<(), Diag<'a>> {
        self.condition.constrain(ctx, infer, sig)?;
        self.condition
            .infer_equality(ctx, infer, TyId::BOOL, self.span)?;
        infer.in_scope(ctx, |ctx, infer| match self.block {
            Expr::Block(block) => {
                block.block_constrain(ctx, infer, sig)?;
                if let Some(otherwise) = self.otherwise {
                    otherwise.constrain(ctx, infer, sig)?;
                }
                Ok(())
            }
            _ => unreachable!(),
        })
    }
}

impl<'a> Constrain<'a> for Loop<'a> {
    fn constrain(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
    ) -> Result<(), Diag<'a>> {
        validate_loop_block(ctx, sig, &self.block)?;
        self.block.constrain(ctx, infer, sig)?;
        Expr::Block(self.block).infer_equality(ctx, infer, TyId::UNIT, self.span)
    }
}

impl<'a> Constrain<'a> for ForLoop<'a> {
    fn constrain(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
    ) -> Result<(), Diag<'a>> {
        validate_loop_block(ctx, sig, &self.block)?;
        Expr::Block(self.block).infer_equality(ctx, infer, TyId::UNIT, self.span)?;
        infer.in_scope(ctx, |ctx, infer| {
            match self.iterable {
                Expr::Range(range) => {
                    match (range.start, range.end) {
                        (Some(start), Some(end)) => {
                            let var = infer.new_var(self.iter);
                            start.constrain_var(ctx, infer, var)?;
                            end.constrain_var(ctx, infer, var)?;
                        }
                        _ => {
                            return Err(ctx.report_error(
                                self.iterable.span(),
                                "expression is not iterable: range must have a start and end",
                            ));
                        }
                    }
                    Ok(())
                }
                Expr::Ident(_) => match self.iterable.resolve_infer(ctx, infer)? {
                    InferTy::Ty(ty) => {
                        match ctx.tys.ty(ty) {
                            Ty::Array(_, inner) => {
                                let var = infer.new_var(self.iter);
                                infer.eq(var, ctx.tys.ty_id(&Ty::Ref(inner)), self.iterable.span());
                                Ok(())
                            }
                            _ => Err(ctx
                                .report_error(self.iterable.span(), "expression is not iterable")),
                        }
                    }
                    InferTy::Int | InferTy::Float => {
                        Err(ctx.report_error(self.iterable.span(), "expression is not iterable"))
                    } //None => Err(ctx.report_error(
                      //    self.iterable.span(),
                      //    "expression is not iterable: failed to infer expression type",
                      //)),
                },
                _ => Err(ctx.report_error(self.iterable.span(), "expression is not iterable")),
            }?;

            self.iter.constrain(ctx, infer, sig)?;
            self.iterable.constrain(ctx, infer, sig)?;
            self.block.constrain(ctx, infer, sig)
        })
    }
}

// TODO: this sort of thing should be automated.
//
// Have a function like `verify_all_returns`, and `verify_openness`
fn validate_loop_block<'a>(
    ctx: &Ctx<'a>,
    sig: &Sig<'a>,
    block: &Block<'a>,
) -> Result<(), Diag<'a>> {
    if let Some(end) = block.end {
        if !end.is_unit(ctx) {
            return Err(ctx.report_error(end.span(), "mismatched types: expected `()`"));
        }
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

impl<'a> Constrain<'a> for Cast<'a> {
    fn constrain(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
    ) -> Result<(), Diag<'a>> {
        self.lhs.constrain(ctx, infer, sig)?;
        let target = ctx.tys.ty(self.ty);
        match self.lhs.resolve_infer(ctx, infer)? {
            InferTy::Int => {
                if !target.is_float() && !target.is_int() {
                    Err(ctx.report_error(
                        self.span,
                        format!(
                            "an value of type `{{integer}}` cannot be cast to `{}`",
                            target.to_string(ctx)
                        ),
                    ))
                } else {
                    Ok(())
                }
            }
            InferTy::Float => {
                if !target.is_float() && !target.is_int() {
                    Err(ctx.report_error(
                        self.span,
                        format!(
                            "a value of type `{{float}}` cannot be cast to `{}`",
                            target.to_string(ctx)
                        ),
                    ))
                } else {
                    Ok(())
                }
            }
            InferTy::Ty(ty) => {
                let src = ctx.tys.ty(ty);
                match src {
                    Ty::Int(int_ty) => {
                        // hard coded case for casting an int (NULL) to a ref
                        if int_ty == IntTy::USIZE && target.is_ref() {
                            return Ok(());
                        }

                        if !target.is_float() && !target.is_int() {
                            Err(ctx.report_error(
                                self.span,
                                format!(
                                    "an value of type `{{integer}}` cannot be cast to `{}`",
                                    target.to_string(ctx)
                                ),
                            ))
                        } else {
                            Ok(())
                        }
                    }
                    Ty::Float(_) => {
                        if !target.is_float() && !target.is_int() {
                            Err(ctx.report_error(
                                self.span,
                                format!(
                                    "a value of type `{{float}}` cannot be cast to `{}`",
                                    target.to_string(ctx)
                                ),
                            ))
                        } else {
                            Ok(())
                        }
                    }
                    Ty::Bool => {
                        if !target.is_int() {
                            Err(ctx.report_error(
                                self.span,
                                format!(
                                    "a value of type `bool` cannot be cast to `{}`",
                                    target.to_string(ctx)
                                ),
                            ))
                        } else {
                            Ok(())
                        }
                    }
                    // hard coded case for casting ref to int to check for null
                    Ty::Ref(ty) if *ty != Ty::Str => {
                        if !matches!(target, Ty::Int(IntTy::USIZE)) {
                            Err(ctx.report_error(
                                self.lhs.span(),
                                format!(
                                    "a value of type `{}` cannot be cast to `{}`",
                                    src.to_string(ctx),
                                    target.to_string(ctx)
                                ),
                            ))
                        } else {
                            Ok(())
                        }
                    }
                    _ => Err(ctx.report_error(
                        self.lhs.span(),
                        format!(
                            "a value of type `{}` cannot be cast to `{}`",
                            src.to_string(ctx),
                            target.to_string(ctx)
                        ),
                    )),
                }
            }
        }
    }
}

impl<'a> Constrain<'a> for Unary<'a> {
    fn constrain(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
    ) -> Result<(), Diag<'a>> {
        match self.kind {
            UOpKind::Ref => Ref {
                span: self.span,
                inner: self.inner,
            }
            .constrain(ctx, infer, sig),
            UOpKind::Deref => Deref {
                span: self.span,
                inner: self.inner,
            }
            .constrain(ctx, infer, sig),
            UOpKind::Not => Not {
                span: self.span,
                inner: self.inner,
            }
            .constrain(ctx, infer, sig),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub struct Ref<'a> {
    pub span: Span,
    pub inner: &'a Expr<'a>,
}

impl<'a> Constrain<'a> for Ref<'a> {
    fn constrain(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
    ) -> Result<(), Diag<'a>> {
        self.inner.constrain(ctx, infer, sig)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub struct Deref<'a> {
    pub span: Span,
    pub inner: &'a Expr<'a>,
}

impl<'a> Constrain<'a> for Deref<'a> {
    fn constrain(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
    ) -> Result<(), Diag<'a>> {
        self.inner.constrain(ctx, infer, sig)?;
        match self.inner.resolve_infer(ctx, infer)? {
            InferTy::Ty(ty) => {
                match ctx.tys.ty(ty) {
                    Ty::Ref(&Ty::Str) => Err(ctx
                        .report_error(self.span, "expression of ty `&str` cannot be dereferenced")),
                    Ty::Ref(_) => Ok(()),
                    ty => Err(ctx.report_error(
                        self.span,
                        format!(
                            "expression of ty `{}` cannot be dereferenced",
                            ty.to_string(ctx)
                        ),
                    )),
                }
            }
            InferTy::Int | InferTy::Float => {
                Err(ctx.report_error(self.span, "expression cannot be dereferenced"))
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub struct Not<'a> {
    pub span: Span,
    pub inner: &'a Expr<'a>,
}

impl<'a> Constrain<'a> for Not<'a> {
    fn constrain(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
    ) -> Result<(), Diag<'a>> {
        self.inner.constrain(ctx, infer, sig)
    }
}

impl<'a> Constrain<'a> for IndexOf<'a> {
    fn constrain(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
    ) -> Result<(), Diag<'a>> {
        self.array.constrain(ctx, infer, sig)?;
        self.index.constrain(ctx, infer, sig)?;
        let index_ty = ctx.tys.builtin(Ty::Int(IntTy::USIZE));
        match self.index {
            Expr::Ident(ident) => {
                let Some(var) = infer.var(ident.id) else {
                    return Err(ctx.undeclared(ident));
                };

                infer.eq(var, index_ty, self.index.span());
            }
            _ => {
                self.index
                    .infer_equality(ctx, infer, index_ty, self.index.span())?;
            }
        }

        match self.array.resolve_infer(ctx, infer)? {
            InferTy::Ty(infer_ty) => match ctx.tys.ty(infer_ty) {
                Ty::Array(_, _) => Ok(()),
                _ => Err(ctx.report_error(
                    self.array.span(),
                    format!(
                        "expression of type `{}` cannot be indexed",
                        infer_ty.to_string(ctx)
                    ),
                )),
            },
            infer_ty @ InferTy::Int | infer_ty @ InferTy::Float => Err(ctx.report_error(
                self.array.span(),
                format!(
                    "expression of type `{}` cannot be indexed",
                    infer_ty.to_string(ctx)
                ),
            )),
        }
    }
}

impl<'a> Constrain<'a> for Range<'a> {
    fn constrain(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
    ) -> Result<(), Diag<'a>> {
        if let Some(start) = self.start {
            start.constrain(ctx, infer, sig)?;
            match start.resolve_infer(ctx, infer)? {
                InferTy::Int => {}
                InferTy::Float => {
                    return Err(ctx.mismatch(start.span(), "{integer}", "{float}"));
                }
                InferTy::Ty(ty) => match ctx.tys.ty(ty) {
                    Ty::Int(_) => {}
                    got => {
                        return Err(ctx.mismatch(start.span(), "{integer}", got));
                    }
                },
            }
        }

        if let Some(end) = self.end {
            end.constrain(ctx, infer, sig)?;
            match end.resolve_infer(ctx, infer)? {
                InferTy::Int => {}
                InferTy::Float => {
                    return Err(ctx.mismatch(end.span(), "{integer}", "{float}"));
                }
                InferTy::Ty(ty) => match ctx.tys.ty(ty) {
                    Ty::Int(_) => {}
                    got => {
                        return Err(ctx.mismatch(end.span(), "{integer}", got));
                    }
                },
            }
        }

        Ok(())
    }
}

impl<'a> Constrain<'a> for Ident {
    fn constrain(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        _sig: &Sig<'a>,
    ) -> Result<(), Diag<'a>> {
        if infer.var(self.id).is_none() {
            Err(ctx.undeclared(self))
        } else {
            Ok(())
        }
    }
}

impl<'a> Constrain<'a> for Lit<'a> {
    fn constrain(
        &self,
        _ctx: &mut Ctx<'a>,
        _infer: &mut InferCtx,
        _sig: &Sig<'a>,
    ) -> Result<(), Diag<'a>> {
        Ok(())
    }
}

impl<'a> Constrain<'a> for BoolLit {
    fn constrain(
        &self,
        _ctx: &mut Ctx<'a>,
        _infer: &mut InferCtx,
        _sig: &Sig<'a>,
    ) -> Result<(), Diag<'a>> {
        Ok(())
    }
}

impl<'a> Constrain<'a> for StrLit<'a> {
    fn constrain(
        &self,
        _ctx: &mut Ctx<'a>,
        _infer: &mut InferCtx,
        _sig: &Sig<'a>,
    ) -> Result<(), Diag<'a>> {
        Ok(())
    }
}

pub fn aquire_access_ty<'a>(
    ctx: &Ctx<'a>,
    infer: &InferCtx,
    access: &Access,
) -> Result<(Span, TyId), Diag<'a>> {
    let ty_var = match access.lhs {
        Expr::Ident(ident) => infer.var(ident.id).ok_or_else(|| ctx.undeclared(ident))?,
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
        Ty::Array(_, _)
        | Ty::Int(_)
        | Ty::Unit
        | Ty::Bool
        | Ty::Ref(_)
        | Ty::Str
        | Ty::Float(_) => {
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

    for (i, acc) in access.accessors.iter().rev().enumerate() {
        let Some(ty) = strukt.get_field_ty(acc.id) else {
            return Err(ctx.report_error(
                acc.span,
                format!(
                    "invalid access: `{}` has no field `{}`",
                    ctx.expect_ident(strukt.name.id),
                    ctx.expect_ident(acc.id)
                ),
            ));
        };

        if i == access.accessors.len() - 1 {
            return Ok((access.span, ty));
        }

        match ctx.tys.ty(ty) {
            Ty::Struct(id) => {
                strukt = ctx.tys.strukt(id);
            }
            Ty::Array(_, _)
            | Ty::Int(_)
            | Ty::Unit
            | Ty::Bool
            | Ty::Ref(_)
            | Ty::Str
            | Ty::Float(_) => {
                let access = access.accessors[i];
                return Err(ctx.report_error(
                    access.span,
                    format!(
                        "invalid access: `{}` is of type `{}`, which has no field `{}`",
                        acc.to_string(ctx),
                        ty.to_string(ctx),
                        access.to_string(ctx),
                    ),
                ));
            }
        }
    }

    unreachable!()
}
