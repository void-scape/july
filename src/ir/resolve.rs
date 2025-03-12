use super::ty::infer::Integral;
use super::Stmt;
use super::*;
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

pub fn resolve_types<'a>(ctx: &mut Ctx<'a>) -> Result<TypeKey, Vec<Diag<'a>>> {
    let mut errors = Vec::new();
    let mut infer = InferCtx::default();

    for konst in ctx.tys.consts() {
        // the `Ident` constraint expected the consts to already be registered
        infer.register_const(konst.name, konst.ty, konst.span);
    }

    // TODO: do better
    let funcs = std::mem::take(&mut ctx.funcs);
    for func in funcs.iter() {
        infer.for_func(func);
        init_params(&mut infer, func);

        if let Err(err) =
            func.block
                .ty_constrain(ctx, &mut infer, func.sig, func.sig.ty, func.sig.span)
        {
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

impl<'a> Expr<'a> {
    /// Fails when:
    ///     aquiring field access type Fails
    ///     ident is undefined
    pub fn resolve_infer(
        &self,
        ctx: &Ctx<'a>,
        infer: &InferCtx,
    ) -> Result<Option<InferTy>, Diag<'a>> {
        Ok(match self {
            Self::Lit(lit) => Some(match lit.kind {
                LitKind::Int(_) => InferTy::Int,
                LitKind::Float(_) => InferTy::Float,
            }),
            Self::Ident(ident) => {
                let Some(var) = infer.get_var(ident.id) else {
                    return Err(ctx.undeclared(ident));
                };

                infer
                    .guess_var_ty(ctx, var)
                    .map(|t| InferTy::Ty(t))
                    .or_else(|| infer.is_var_integral(ctx, var).then_some(InferTy::Int))
            }
            Self::Access(access) => Some(InferTy::Ty(aquire_access_ty(ctx, infer, access)?.1)),
            Self::Call(call) => Some(InferTy::Ty(call.sig.ty)),
            Self::Str(_) => Some(InferTy::Ty(TyId::STR_LIT)),
            Self::Bin(bin) => {
                let lhs = bin.lhs.resolve_infer(ctx, infer)?;
                let rhs = bin.lhs.resolve_infer(ctx, infer)?;

                //assert_eq!(lhs, rhs);
                match bin.kind {
                    BinOpKind::Add
                    | BinOpKind::Sub
                    | BinOpKind::Mul
                    | BinOpKind::Div
                    | BinOpKind::Xor => match (lhs, rhs) {
                        (Some(lhs), Some(rhs)) => {
                            // I know this is not right, but I am too lazy
                            assert_eq!(lhs, rhs);
                            Some(lhs)
                        }
                        (Some(ty), None) | (None, Some(ty)) => Some(ty),
                        _ => None,
                    },
                    BinOpKind::Eq => Some(InferTy::Ty(TyId::BOOL)),
                }
            }
            Self::Bool(_) => Some(InferTy::Ty(TyId::BOOL)),
            Self::IndexOf(index) => match index.array.resolve_infer(ctx, infer)? {
                Some(InferTy::Ty(arr_ty)) => match ctx.tys.ty(arr_ty) {
                    Ty::Array(_, inner) => Some(InferTy::Ty(ctx.tys.get_ty_id(inner).unwrap())),
                    other => panic!("invalid array type for index of: {:?}", other),
                },
                other => panic!("invalid array type for index of: {:?}", other),
            },
            Self::Struct(def) => Some(InferTy::Ty(ctx.tys.struct_ty_id(def.id))),
            ty => todo!("{ty:?}"),
        })
    }
}

pub trait Constrain<'a>: Debug {
    fn constrain(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
    ) -> Result<(), Diag<'a>>;

    fn ty_constrain(
        &self,
        _ctx: &mut Ctx<'a>,
        _infer: &mut InferCtx,
        _sig: &Sig<'a>,
        _ty: TyId,
        _source: Span,
    ) -> Result<(), Diag<'a>> {
        unimplemented!("{:#?}", self);
    }

    fn apply_constraint(
        &self,
        _ctx: &mut Ctx<'a>,
        _infer: &mut InferCtx,
        _sig: &Sig<'a>,
        _var: TyVar,
    ) -> Result<(), Diag<'a>> {
        unimplemented!("{:#?}", self);
    }
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

                    if let Some((span, ty)) = let_.ty {
                        infer.eq(var, ty, span);
                        let_.rhs.ty_constrain(ctx, infer, sig, ty, span)?;
                    }
                    let_.rhs.apply_constraint(ctx, infer, sig, var)?;
                }
            },
            SemiStmt::Ret(r) => {
                if let Some(expr) = &r.expr {
                    expr.ty_constrain(ctx, infer, sig, sig.ty, sig.span)?;
                } else if !ctx.tys.is_unit(sig.ty) {
                    return Err(ctx
                        .mismatch(r.span, sig.ty, TyId::UNIT)
                        .msg(Msg::help(sig.span, "from fn signature")));
                }
            }
            SemiStmt::Assign(assign) => {
                assign.lhs.constrain(ctx, infer, sig)?;
                match assign.lhs.resolve_infer(ctx, infer)? {
                    Some(InferTy::Ty(ty)) => {
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
                    Some(InferTy::Int) | Some(InferTy::Float) => {
                        // assignable
                    }
                    None => {
                        // TODO: must determine later is this is assignable
                    }
                }
                if assign.rhs.is_unit(ctx) {
                    return Err(ctx.report_error(
                        assign.rhs.span(),
                        "cannot assign value to an expression of type `()`",
                    ));
                }
                assign.rhs.constrain(ctx, infer, sig)?;
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
            Self::Continue(_) | Self::Break(_) => Ok(()),
        }
    }

    fn ty_constrain(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
        ty: TyId,
        source: Span,
    ) -> Result<(), Diag<'a>> {
        self.constrain(ctx, infer, sig)?;
        match self {
            Self::Ident(ident) => ident.ty_constrain(ctx, infer, sig, ty, source),
            Self::Lit(lit) => lit.ty_constrain(ctx, infer, sig, ty, source),
            Self::Str(str) => str.ty_constrain(ctx, infer, sig, ty, source),
            Self::Bool(bool) => bool.ty_constrain(ctx, infer, sig, ty, source),
            Self::Bin(bin) => bin.ty_constrain(ctx, infer, sig, ty, source),
            Self::Access(access) => access.ty_constrain(ctx, infer, sig, ty, source),
            Self::Struct(def) => def.ty_constrain(ctx, infer, sig, ty, source),
            Self::Enum(def) => def.ty_constrain(ctx, infer, sig, ty, source),
            Self::Call(call) => call.ty_constrain(ctx, infer, sig, ty, source),
            Self::Block(block) => block.ty_constrain(ctx, infer, sig, ty, source),
            Self::If(if_) => if_.ty_constrain(ctx, infer, sig, ty, source),
            Self::Loop(loop_) => loop_.ty_constrain(ctx, infer, sig, ty, source),
            Self::For(for_) => for_.ty_constrain(ctx, infer, sig, ty, source),
            Self::Unary(unary) => unary.ty_constrain(ctx, infer, sig, ty, source),
            Self::Array(arr) => arr.ty_constrain(ctx, infer, sig, ty, source),
            Self::IndexOf(index) => index.ty_constrain(ctx, infer, sig, ty, source),
            Self::Range(range) => range.ty_constrain(ctx, infer, sig, ty, source),
            Self::Continue(_) | Self::Break(_) => unreachable!(),
        }
    }

    fn apply_constraint(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
        var: TyVar,
    ) -> Result<(), Diag<'a>> {
        self.constrain(ctx, infer, sig)?;
        match self {
            Self::Ident(ident) => ident.apply_constraint(ctx, infer, sig, var),
            Self::Lit(lit) => lit.apply_constraint(ctx, infer, sig, var),
            Self::Str(str) => str.apply_constraint(ctx, infer, sig, var),
            Self::Bool(bool) => bool.apply_constraint(ctx, infer, sig, var),
            Self::Bin(bin) => bin.apply_constraint(ctx, infer, sig, var),
            Self::Access(access) => access.apply_constraint(ctx, infer, sig, var),
            Self::Struct(def) => def.apply_constraint(ctx, infer, sig, var),
            Self::Enum(def) => def.apply_constraint(ctx, infer, sig, var),
            Self::Call(call) => call.apply_constraint(ctx, infer, sig, var),
            Self::Block(block) => block.apply_constraint(ctx, infer, sig, var),
            Self::If(if_) => if_.apply_constraint(ctx, infer, sig, var),
            Self::Loop(loop_) => loop_.apply_constraint(ctx, infer, sig, var),
            Self::For(for_) => for_.apply_constraint(ctx, infer, sig, var),
            Self::Unary(unary) => unary.apply_constraint(ctx, infer, sig, var),
            Self::Array(arr) => arr.apply_constraint(ctx, infer, sig, var),
            Self::IndexOf(index) => index.apply_constraint(ctx, infer, sig, var),
            Self::Range(range) => range.apply_constraint(ctx, infer, sig, var),
            Self::Continue(_) | Self::Break(_) => unreachable!(),
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
        self.lhs.constrain(ctx, infer, sig)?;
        self.rhs.constrain(ctx, infer, sig)?;
        // attempt to bind arguments WITH eachother
        constrain_unbounded_bin_op(ctx, infer, sig, self)
    }

    fn ty_constrain(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
        ty: TyId,
        source: Span,
    ) -> Result<(), Diag<'a>> {
        self.constrain(ctx, infer, sig)?;
        self.bin_ty_constrain(ctx, infer, sig, ty, source)
    }

    fn apply_constraint(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
        var: TyVar,
    ) -> Result<(), Diag<'a>> {
        self.constrain(ctx, infer, sig)?;
        self.bin_apply_constraint(ctx, infer, sig, var)
    }
}

impl<'a> BinOp<'a> {
    pub fn bin_ty_constrain(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
        ty: TyId,
        source: Span,
    ) -> Result<(), Diag<'a>> {
        if self.kind == BinOpKind::Eq && ty == TyId::BOOL {
            Ok(())
        } else {
            self.lhs.ty_constrain(ctx, infer, sig, ty, source)?;
            self.rhs.ty_constrain(ctx, infer, sig, ty, source)
        }
    }

    pub fn bin_apply_constraint(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
        var: TyVar,
    ) -> Result<(), Diag<'a>> {
        if self.kind == BinOpKind::Eq {
            //infer.eq(var, TyId::BOOL, self.span);
            Ok(())
        } else {
            self.lhs.apply_constraint(ctx, infer, sig, var)?;
            self.rhs.apply_constraint(ctx, infer, sig, var)
        }
    }
}

fn constrain_unbounded_bin_op<'a>(
    ctx: &mut Ctx<'a>,
    infer: &mut InferCtx,
    sig: &Sig<'a>,
    bin: &BinOp<'a>,
) -> Result<(), Diag<'a>> {
    if let Some(var) = find_ty_var_in_bin_op(ctx, infer, bin)? {
        bin.bin_apply_constraint(ctx, infer, sig, var)?;
    } else if let Some((span, ty)) = find_ty_in_bin_op(ctx, infer, bin)? {
        bin.bin_ty_constrain(ctx, infer, sig, ty, span)?;
    } else {
        //if bin.is_integral(ctx).is_none_or(|i| !i) {
        //    return Err(ctx.report_error(bin.span, "expected every term to be an integer"));
        //}
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
        Expr::Ident(ident) => match infer.get_var(ident.id) {
            Some(var) => Ok(Some(var)),
            None => match ctx.tys.get_const(ident.id) {
                Some(konst) => Ok(Some(infer.get_const(konst.name.id).unwrap())),
                None => Err(ctx.undeclared(ident)),
            },
        },
        Expr::Bin(bin) => find_ty_var_in_bin_op(ctx, infer, bin),
        Expr::Call(_)
        | Expr::Struct(_)
        | Expr::Enum(_)
        | Expr::Lit(_)
        | Expr::Continue(_)
        | Expr::Break(_)
        | Expr::Access(_)
        | Expr::Str(_)
        | Expr::Range(_)
        | Expr::Bool(_) => Ok(None),
        Expr::If(_) => todo!(),
        Expr::Block(_) => todo!(),
        Expr::Loop(_) => todo!(),
        Expr::Unary(_) => todo!(),
        //Expr::Ref(inner) => find_ty_var_in_expr(ctx, infer, inner.inner),
        Expr::Array(_) => todo!(),
        Expr::IndexOf(_) => todo!(),
        Expr::For(_) => todo!(),
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
        Expr::Bool(bool) => Ok(Some((bool.span, ctx.tys.bool()))),
        Expr::Bin(bin) => find_ty_in_bin_op(ctx, infer, bin),
        Expr::Call(call) => Ok(Some((call.span, call.sig.ty))),
        Expr::Struct(def) => Ok(Some((def.span, ctx.tys.struct_ty_id(def.id)))),
        Expr::Range(_) | Expr::Continue(_) | Expr::Break(_) | Expr::Ident(_) | Expr::Lit(_) => {
            Ok(None)
        }
        Expr::Str(str) => Ok(Some((str.span, ctx.tys.str_lit()))),
        Expr::Access(access) => Ok(Some(aquire_access_ty(ctx, infer, access)?)),
        //Expr::Ref(ref_) => Ok(find_ty_in_expr(ctx, infer, ref_.inner)?.map(|(_, ty)| {
        //    (
        //        ref_.span,
        //        ctx.tys.ty_id(&Ty::Ref(ctx.intern(ctx.tys.ty(ty)))),
        //    )
        //})),
        Expr::Array(_) => todo!(),
        Expr::Enum(_) => unimplemented!(),
        Expr::If(_) => todo!(),
        Expr::Block(_) => todo!(),
        Expr::Loop(_) => todo!(),
        Expr::Unary(_) => todo!(),
        Expr::IndexOf(_) => todo!(),
        Expr::For(_) => todo!(),
    }
}

impl<'a> Constrain<'a> for Access<'a> {
    fn constrain(&self, _: &mut Ctx<'a>, _: &mut InferCtx, _: &Sig<'a>) -> Result<(), Diag<'a>> {
        assert!(matches!(self.lhs, Expr::Ident(_)));
        Ok(())
    }

    fn ty_constrain(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
        ty: TyId,
        source: Span,
    ) -> Result<(), Diag<'a>> {
        self.constrain(ctx, infer, sig)?;
        let (span, field_ty) = aquire_access_ty(ctx, infer, self)?;
        if ty != field_ty {
            Err(ctx
                .mismatch(span, ty, field_ty)
                .msg(Msg::help(source, "from this binding")))
        } else {
            Ok(())
        }
    }

    fn apply_constraint(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
        var: TyVar,
    ) -> Result<(), Diag<'a>> {
        self.constrain(ctx, infer, sig)?;
        let (span, ty) = aquire_access_ty(ctx, infer, self)?;
        infer.eq(var, ty, span);
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
                .ty_constrain(ctx, infer, sig, field_ty, field_def.name.span)?;
        }

        Ok(())
    }

    fn ty_constrain(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
        ty: TyId,
        source: Span,
    ) -> Result<(), Diag<'a>> {
        self.constrain(ctx, infer, sig)?;
        let struct_ty = ctx.tys.struct_ty_id(self.id);
        if ty != struct_ty {
            Err(ctx
                .mismatch(self.span, ty, struct_ty)
                .msg(Msg::help(source, "from this binding")))
        } else {
            Ok(())
        }
    }

    fn apply_constraint(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
        var: TyVar,
    ) -> Result<(), Diag<'a>> {
        self.constrain(ctx, infer, sig)?;
        infer.eq(var, ctx.tys.struct_ty_id(self.id), self.span);

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

impl<'a> ArrDef<'a> {
    pub fn constrain_with_infer(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
        infer_ty: Option<InferTy>,
    ) -> Result<TyId, Diag<'a>> {
        match infer_ty {
            Some(infer_ty) => match infer_ty {
                InferTy::Ty(infer_ty) => match ctx.tys.ty(infer_ty) {
                    Ty::Array(len, inner) => {
                        let inner = ctx.tys.ty_id(inner);
                        let mut errors = Vec::new();
                        for expr in self.exprs.iter() {
                            if let Err(diag) = expr.ty_constrain(ctx, infer, sig, inner, self.span)
                            {
                                errors.push(diag);
                            }
                        }

                        if len != self.exprs.len() {
                            errors.push(ctx.report_error(
                                self.span,
                                format!(
                                    "expected an array of len `{}`, got `{}`",
                                    len,
                                    self.exprs.len()
                                ),
                            ));
                        }

                        if !errors.is_empty() {
                            Err(Diag::bundle(errors))
                        } else {
                            Ok(infer_ty)
                        }
                    }
                    _ => Err(ctx.mismatch(self.span, infer_ty.to_string(ctx), "{array}")),
                },
                InferTy::Int | InferTy::Float => {
                    Err(ctx.mismatch(self.span, infer_ty.to_string(ctx), "{array}"))
                }
            },
            None => Err(ctx.report_error(
                self.span,
                "the type of `{expr}` cannot be infered: array types must be statically known",
            )),
        }
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

    fn ty_constrain(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
        ty: TyId,
        source: Span,
    ) -> Result<(), Diag<'a>> {
        self.constrain(ctx, infer, sig)?;
        match ctx.tys.ty(ty) {
            Ty::Array(len, inner) => {
                let inner = ctx.tys.ty_id(inner);
                let mut errors = Vec::new();
                for expr in self.exprs.iter() {
                    if let Err(diag) = expr.ty_constrain(ctx, infer, sig, inner, source) {
                        errors.push(diag);
                    }
                }

                if len != self.exprs.len() {
                    errors.push(ctx.report_error(
                        self.span,
                        format!(
                            "expected an array of len `{}`, got `{}`",
                            len,
                            self.exprs.len()
                        ),
                    ));
                }

                if !errors.is_empty() {
                    Err(Diag::bundle(errors))
                } else {
                    Ok(())
                }
            }
            ty => Err(ctx.mismatch(self.span, "{array}", ty)),
        }
    }

    fn apply_constraint(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
        var: TyVar,
    ) -> Result<(), Diag<'a>> {
        self.constrain(ctx, infer, sig)?;
        let results = self
            .exprs
            .iter()
            .map(|e| e.resolve_infer(ctx, infer))
            .collect::<Result<Vec<_>, _>>()?;
        match results.iter().find_map(|ty| *ty) {
            Some(inner_ty) => {
                match inner_ty {
                    InferTy::Int => {
                        infer.eq(
                            var,
                            ctx.tys
                                .ty_id(&Ty::Array(self.exprs.len(), &Ty::Int(IntTy::ISIZE))),
                            self.span,
                        );
                    }
                    InferTy::Float => {
                        infer.eq(
                            var,
                            ctx.tys
                                .ty_id(&Ty::Array(self.exprs.len(), &Ty::Float(FloatTy::F64))),
                            self.span,
                        );
                    }

                    InferTy::Ty(inner) => {
                        infer.eq(
                            var,
                            ctx.tys
                                .ty_id(&Ty::Array(self.exprs.len(), ctx.intern(ctx.tys.ty(inner)))),
                            self.span,
                        );
                    }
                }

                Ok(())
            }
            None => Err(ctx.report_error(self.span, "cannot infer the element type")),
        }
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
                    Expr::Ident(ident) => {
                        match infer
                            .get_var(ident.id)
                            .or_else(|| infer.get_const(ident.id))
                        {
                            Some(var) => {
                                if ctx.expect_ident(ident.id) != "NULL" {
                                    infer.eq(var, param.ty, param.span);
                                }
                            }
                            None => {
                                errors.push(ctx.undeclared(ident));
                            }
                        }
                    }
                    _ => {
                        if let Err(diag) = expr.ty_constrain(ctx, infer, sig, param.ty, param.span)
                        {
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

    fn ty_constrain(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
        ty: TyId,
        _: Span,
    ) -> Result<(), Diag<'a>> {
        self.constrain(ctx, infer, sig)?;
        if self.sig.ty != ty {
            Err(ctx
                .mismatch(self.span, ty, self.sig.ty)
                .msg(Msg::help(self.sig.span, "function defined here")))
        } else {
            Ok(())
        }
    }

    fn apply_constraint(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
        var: TyVar,
    ) -> Result<(), Diag<'a>> {
        self.constrain(ctx, infer, sig)?;
        infer.eq(var, self.sig.ty, self.span);
        Ok(())
    }
}

impl<'a> Constrain<'a> for Block<'a> {
    fn constrain(
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

    fn ty_constrain(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
        ty: TyId,
        source: Span,
    ) -> Result<(), Diag<'a>> {
        self.constrain(ctx, infer, sig)?;
        if let Some(end) = self.end {
            match end {
                Expr::Ident(ident) => {
                    let var = infer
                        .get_var(ident.id)
                        .ok_or_else(|| ctx.undeclared(ident))?;
                    infer.eq(var, ty, source);
                }
                _ => {
                    end.ty_constrain(ctx, infer, sig, ty, source)?;
                }
            }
        }

        Ok(())
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
            .ty_constrain(ctx, infer, sig, TyId::BOOL, self.span)?;
        self.block.constrain(ctx, infer, sig)?;
        if let Some(otherwise) = self.otherwise {
            otherwise.constrain(ctx, infer, sig)?;
        }

        Ok(())
    }

    fn ty_constrain(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
        ty: TyId,
        source: Span,
    ) -> Result<(), Diag<'a>> {
        self.constrain(ctx, infer, sig)?;
        self.block.ty_constrain(ctx, infer, sig, ty, source)?;
        if let Some(otherwise) = self.otherwise {
            otherwise.ty_constrain(ctx, infer, sig, ty, source)
        } else {
            Ok(())
        }
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
        self.block.constrain(ctx, infer, sig)
    }

    fn ty_constrain(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
        ty: TyId,
        source: Span,
    ) -> Result<(), Diag<'a>> {
        self.constrain(ctx, infer, sig)?;
        if TyId::UNIT != ty {
            return Err(ctx
                .mismatch(self.block.span, ty, TyId::UNIT)
                .msg(Msg::help(source, "from this binding")));
        }
        self.block
            .ty_constrain(ctx, infer, sig, TyId::UNIT, self.span)
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
        match self.iterable {
            Expr::Range(range) => {
                match (range.start, range.end) {
                    (Some(start), Some(end)) => {
                        let var = infer.new_var(self.iter);
                        start.apply_constraint(ctx, infer, sig, var)?;
                        end.apply_constraint(ctx, infer, sig, var)?;
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
                Some(InferTy::Int) | Some(InferTy::Float) => {
                    Err(ctx.report_error(self.iterable.span(), "expression is not iterable"))
                }
                Some(InferTy::Ty(ty)) => match ctx.tys.ty(ty) {
                    Ty::Array(_, inner) => {
                        let var = infer.new_var(self.iter);
                        infer.eq(var, ctx.tys.ty_id(&Ty::Ref(inner)), self.iterable.span());
                        Ok(())
                    }
                    _ => Err(ctx.report_error(self.iterable.span(), "expression is not iterable")),
                },
                None => Err(ctx.report_error(
                    self.iterable.span(),
                    "expression is not iterable: failed to infer expression type",
                )),
            },
            _ => Err(ctx.report_error(self.iterable.span(), "expression is not iterable")),
        }?;

        self.iterable.constrain(ctx, infer, sig)?;
        self.block.constrain(ctx, infer, sig)
    }

    fn ty_constrain(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
        ty: TyId,
        source: Span,
    ) -> Result<(), Diag<'a>> {
        self.constrain(ctx, infer, sig)?;
        if TyId::UNIT != ty {
            return Err(ctx
                .mismatch(self.block.span, ty, TyId::UNIT)
                .msg(Msg::help(source, "from this binding")));
        }
        self.block
            .ty_constrain(ctx, infer, sig, TyId::UNIT, self.span)
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

    fn ty_constrain(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
        ty: TyId,
        source: Span,
    ) -> Result<(), Diag<'a>> {
        match self.kind {
            UOpKind::Ref => Ref {
                span: self.span,
                inner: self.inner,
            }
            .ty_constrain(ctx, infer, sig, ty, source),
            UOpKind::Deref => Deref {
                span: self.span,
                inner: self.inner,
            }
            .ty_constrain(ctx, infer, sig, ty, source),
            UOpKind::Not => Not {
                span: self.span,
                inner: self.inner,
            }
            .ty_constrain(ctx, infer, sig, ty, source),
        }
    }

    fn apply_constraint(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
        var: TyVar,
    ) -> Result<(), Diag<'a>> {
        match self.kind {
            UOpKind::Ref => Ref {
                span: self.span,
                inner: self.inner,
            }
            .apply_constraint(ctx, infer, sig, var),
            UOpKind::Deref => Deref {
                span: self.span,
                inner: self.inner,
            }
            .apply_constraint(ctx, infer, sig, var),
            UOpKind::Not => Not {
                span: self.span,
                inner: self.inner,
            }
            .apply_constraint(ctx, infer, sig, var),
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

    fn ty_constrain(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
        ty: TyId,
        source: Span,
    ) -> Result<(), Diag<'a>> {
        self.constrain(ctx, infer, sig)?;

        let ty = ctx.tys.ty(ty);
        match ty {
            Ty::Ref(ty) => {
                let ty = ctx.tys.ty_id(ty);
                self.inner.ty_constrain(ctx, infer, sig, ty, source)?;
            }
            inner => {
                return Err(ctx.mismatch(self.span, Ty::Ref(&ty), inner));
            }
        }

        Ok(())
    }

    fn apply_constraint(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
        var: TyVar,
    ) -> Result<(), Diag<'a>> {
        self.constrain(ctx, infer, sig)?;
        match self.inner.resolve_infer(ctx, infer)? {
            Some(InferTy::Ty(ty)) => {
                infer.eq(
                    var,
                    ctx.tys.ty_id(&Ty::Ref(ctx.intern(ctx.tys.ty(ty)))),
                    self.span,
                );
                Ok(())
            }
            _ => Err(ctx.report_error(self.span, "failed to infer the expression type")),
        }
        //self.inner.apply_constraint(ctx, infer, sig, var)
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
            Some(InferTy::Ty(ty)) => {
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
            Some(InferTy::Int) | Some(InferTy::Float) => {
                Err(ctx.report_error(self.span, "expression cannot be dereferenced"))
            }
            None => Err(ctx.report_error(self.inner.span(), "could not infer type of expression")),
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

    fn ty_constrain(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
        ty: TyId,
        source: Span,
    ) -> Result<(), Diag<'a>> {
        match ctx.tys.ty(ty) {
            Ty::Int(_) | Ty::Bool => {}
            ty => {
                return Err(ctx.report_error(
                    self.span,
                    format!(
                        "cannot apply unary operator `!` to type `{}`",
                        ty.to_string(ctx)
                    ),
                ))
            }
        }

        self.constrain(ctx, infer, sig)?;
        self.inner.ty_constrain(ctx, infer, sig, ty, source)?;

        Ok(())
    }

    fn apply_constraint(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
        var: TyVar,
    ) -> Result<(), Diag<'a>> {
        self.constrain(ctx, infer, sig)?;
        self.inner.apply_constraint(ctx, infer, sig, var)
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
                let Some(var) = infer
                    .get_var(ident.id)
                    .or_else(|| infer.get_const(ident.id))
                else {
                    return Err(ctx.undeclared(ident));
                };

                infer.eq(var, index_ty, self.index.span());
            }
            _ => {
                self.index
                    .ty_constrain(ctx, infer, sig, index_ty, self.index.span())?;
            }
        }

        match self.array.resolve_infer(ctx, infer)? {
            Some(infer_ty) => match infer_ty {
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
                InferTy::Int | InferTy::Float => Err(ctx.report_error(
                    self.array.span(),
                    format!(
                        "expression of type `{}` cannot be indexed",
                        infer_ty.to_string(ctx)
                    ),
                )),
            },
            None => Err(ctx.report_error(
                self.span,
                "the type of `{expr}` cannot be infered: array types must be statically known",
            )),
        }
    }

    fn ty_constrain(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
        ty: TyId,
        source: Span,
    ) -> Result<(), Diag<'a>> {
        self.constrain(ctx, infer, sig)?;
        let infer_ty = self.array.resolve_infer(ctx, infer).unwrap().unwrap();
        match infer_ty {
            InferTy::Ty(infer_ty) => match ctx.tys.ty(infer_ty) {
                Ty::Array(_, inner) => {
                    if ctx.tys.ty_id(inner) != ty {
                        Err(ctx
                            .mismatch(self.array.span(), inner, ty)
                            .msg(Msg::help(source, "from this binding")))
                    } else {
                        Ok(())
                    }
                }
                _ => unreachable!(),
            },
            _ => unreachable!(),
        }
    }

    fn apply_constraint(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        sig: &Sig<'a>,
        var: TyVar,
    ) -> Result<(), Diag<'a>> {
        self.constrain(ctx, infer, sig)?;
        let infer_ty = self.array.resolve_infer(ctx, infer).unwrap().unwrap();
        match infer_ty {
            InferTy::Ty(infer_ty) => match ctx.tys.ty(infer_ty) {
                Ty::Array(_, inner) => {
                    infer.eq(var, ctx.tys.ty_id(inner), self.index.span());
                    Ok(())
                }
                _ => unreachable!(),
            },
            _ => unreachable!(),
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
                Some(InferTy::Int) => {}
                Some(InferTy::Float) => {
                    return Err(ctx.mismatch(start.span(), "{integer}", "{float}"));
                }
                Some(InferTy::Ty(ty)) => match ctx.tys.ty(ty) {
                    Ty::Int(_) => {}
                    got => {
                        return Err(ctx.mismatch(start.span(), "{integer}", got));
                    }
                },
                None => {
                    // constrain to throw the error with helpful information
                    start
                        .ty_constrain(
                            ctx,
                            infer,
                            sig,
                            // this might not always report the correct error
                            ctx.tys.builtin(Ty::Int(IntTy::USIZE)),
                            self.span,
                        )
                        .map_err(|diag| {
                            diag.wrap(
                                ctx.report_help(self.span, "range must produce integer values"),
                            )
                        })?;
                }
            }
        }

        if let Some(end) = self.end {
            end.constrain(ctx, infer, sig)?;
            match end.resolve_infer(ctx, infer)? {
                Some(InferTy::Int) => {}
                Some(InferTy::Float) => {
                    return Err(ctx.mismatch(end.span(), "{integer}", "{float}"));
                }
                Some(InferTy::Ty(ty)) => match ctx.tys.ty(ty) {
                    Ty::Int(_) => {}
                    got => {
                        return Err(ctx.mismatch(end.span(), "{integer}", got));
                    }
                },
                None => {
                    // constrain to throw the error with helpful information
                    end.ty_constrain(
                        ctx,
                        infer,
                        sig,
                        // this might not always report the correct error
                        ctx.tys.builtin(Ty::Int(IntTy::USIZE)),
                        self.span,
                    )
                    .map_err(|diag| {
                        diag.wrap(ctx.report_help(self.span, "range must produce integer values"))
                    })?;
                }
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
        if infer
            .get_var(self.id)
            .or_else(|| infer.get_const(self.id))
            .is_none()
        {
            Err(ctx.undeclared(self))
        } else {
            Ok(())
        }
    }

    fn ty_constrain(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        _: &Sig<'a>,
        ty: TyId,
        source: Span,
    ) -> Result<(), Diag<'a>> {
        match infer.get_var(self.id) {
            Some(var) => {
                if let Some(_ty_var) = infer.guess_var_ty(ctx, var) {
                    // TODO: if there are only integral constraints on a type up to this point, it
                    // will confidently return the default integer type, even if later it will be
                    // constrained to some other integer type
                    //if ty_var != ty {
                    //    return Err(ctx.mismatch(self.span, ty, ty_var));
                    //}
                    infer.eq(var, ty, source);
                }
            }
            None => match ctx.tys.get_const(self.id) {
                Some(konst) => {
                    if ctx.tys.ty(konst.ty).is_int()
                        && ctx.expect_ident(konst.name.id) == "NULL"
                        && ctx.tys.ty(ty).is_ref()
                    {
                        // hard coded special case for null atm
                    } else if konst.ty != ty {
                        return Err(ctx
                            .mismatch(self.span, ty, konst.ty)
                            .msg(Msg::help(source, "from this binding")));
                    }
                }
                None => {
                    return Err(ctx.undeclared(self));
                }
            },
        }

        Ok(())
    }

    fn apply_constraint(
        &self,
        ctx: &mut Ctx<'a>,
        infer: &mut InferCtx,
        _: &Sig<'a>,
        var: TyVar,
    ) -> Result<(), Diag<'a>> {
        if ctx.expect_ident(self.id) == "NULL" || infer.var_ident(ctx, var) == "NULL"
        //&& infer
        //    .guess_var_ty(ctx, var)
        //    .is_some_and(|ty| ctx.tys.ty(ty).is_ref())
        {
            // hard coded special case for null atm
        } else {
            let other_ty = infer
                .get_var(self.id)
                .or_else(|| infer.get_const(self.id))
                .ok_or_else(|| ctx.undeclared(self))?;
            infer.var_eq(ctx, var, other_ty);
        }
        Ok(())
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

    fn ty_constrain(
        &self,
        ctx: &mut Ctx<'a>,
        _: &mut InferCtx,
        _: &Sig<'a>,
        ty: TyId,
        source: Span,
    ) -> Result<(), Diag<'a>> {
        match self.kind {
            LitKind::Int(_) => {
                if !ctx.tys.ty(ty).is_int() {
                    return Err(ctx
                        .mismatch(self.span, ty, "{integer}")
                        .msg(Msg::help(source, "from this binding")));
                }
            }
            LitKind::Float(_) => {
                if !ctx.tys.ty(ty).is_float() {
                    return Err(ctx
                        .mismatch(self.span, ty, "{float}")
                        .msg(Msg::help(source, "from this binding")));
                }
            }
        }
        Ok(())
    }

    fn apply_constraint(
        &self,
        _: &mut Ctx<'a>,
        infer: &mut InferCtx,
        _: &Sig<'a>,
        var: TyVar,
    ) -> Result<(), Diag<'a>> {
        match self.kind {
            LitKind::Int(_) => infer.integral(Integral::Int, var, self.span),
            LitKind::Float(_) => infer.integral(Integral::Float, var, self.span),
        }
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

    fn ty_constrain(
        &self,
        ctx: &mut Ctx<'a>,
        _: &mut InferCtx,
        _: &Sig<'a>,
        ty: TyId,
        source: Span,
    ) -> Result<(), Diag<'a>> {
        if ty != TyId::BOOL {
            Err(ctx
                .mismatch(self.span, ty, TyId::BOOL)
                .msg(Msg::help(source, "from this binding")))
        } else {
            Ok(())
        }
    }

    fn apply_constraint(
        &self,
        _: &mut Ctx<'a>,
        infer: &mut InferCtx,
        _: &Sig<'a>,
        var: TyVar,
    ) -> Result<(), Diag<'a>> {
        infer.eq(var, TyId::BOOL, self.span);
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

    fn ty_constrain(
        &self,
        ctx: &mut Ctx<'a>,
        _: &mut InferCtx,
        _: &Sig<'a>,
        ty: TyId,
        source: Span,
    ) -> Result<(), Diag<'a>> {
        if ty != TyId::STR_LIT {
            Err(ctx
                .mismatch(self.span, ty, "string literal")
                .msg(Msg::help(source, "from this binding")))
        } else {
            Ok(())
        }
    }

    fn apply_constraint(
        &self,
        _: &mut Ctx<'a>,
        infer: &mut InferCtx,
        _: &Sig<'a>,
        var: TyVar,
    ) -> Result<(), Diag<'a>> {
        infer.eq(var, TyId::STR_LIT, self.span);
        Ok(())
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
                //Span::from_spans(acc.span, access.accessors.last().unwrap().span),
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
