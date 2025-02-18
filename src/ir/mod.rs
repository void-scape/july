use self::enom::{Enum, EnumDef, Variant};
use self::ident::IdentId;
use self::sig::Sig;
use self::strukt::{Field, FieldDef, Struct, StructDef, StructId};
use self::ty::{FullTy, TyErr};
use crate::diagnostic::{self, Diag, Msg};
use crate::ir::ctx::Ctx;
use crate::ir::ident::Ident;
use crate::ir::lit::Lit;
use crate::ir::ty::{Arch, Constraint, ConstraintKind, Ty, TyCtx, TyVar, TypeKey};
use crate::lex::buffer::{Span, TokenQuery};
use crate::lex::buffer::{TokenBuffer, TokenId};
use crate::parse::rules::prelude as rules;
use crate::parse::Item;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

pub mod ctx;
pub mod enom;
pub mod ident;
pub mod lit;
pub mod mem;
pub mod sig;
pub mod strukt;
pub mod ty;

pub const SYM_DEF: &str = "undefined symbol";

pub fn lower<'a>(tokens: &'a TokenBuffer<'a>, items: &[Item]) -> Result<Ctx<'a>, ()> {
    let mut ctx = Ctx::new(tokens);

    let enoms = lower_set(
        items
            .iter()
            .filter_map(|i| match i {
                Item::Enum(enom) => Some(enom),
                _ => None,
            })
            .map(|e| enom(&mut ctx, e)),
    )?;
    ctx.store_enums(enoms);

    let struct_ids = items
        .iter()
        .filter_map(|i| match i {
            Item::Struct(strukt) => Some(strukt),
            _ => None,
        })
        .enumerate()
        .map(|(i, strukt)| (ctx.store_ident(strukt.name).id, StructId(i)))
        .collect();
    let structs = lower_set(
        items
            .iter()
            .filter_map(|i| match i {
                Item::Struct(strukt) => Some(strukt),
                _ => None,
            })
            .map(|s| strukt(&mut ctx, &struct_ids, s)),
    )?;
    ctx.structs.set_storage(struct_ids, structs);
    ctx.build_type_layouts();

    let sigs = lower_set(
        items
            .iter()
            .filter_map(|i| match i {
                Item::Func(func) => Some(func),
                _ => None,
            })
            .map(|f| func_sig(&mut ctx, f)),
    )?;
    ctx.store_sigs(sigs);

    let funcs = lower_set(
        items
            .iter()
            .filter_map(|i| match i {
                Item::Func(func) => Some(func),
                _ => None,
            })
            .map(|f| func(&mut ctx, f)),
    )?;
    ctx.store_funcs(funcs);

    Ok(ctx)
}

pub fn lower_set<'a, O>(items: impl Iterator<Item = Result<O, Diag<'a>>>) -> Result<Vec<O>, ()> {
    let mut error = false;
    let mut set = Vec::new();
    for item in items {
        match item {
            Ok(item) => {
                set.push(item);
            }
            Err(diag) => {
                diagnostic::report(diag);
                error = true;
            }
        }
    }
    (!error).then_some(set).ok_or(())
}

fn enom<'a>(ctx: &mut Ctx<'a>, enom: &rules::Enum) -> Result<Enum, Diag<'a>> {
    let mut variant_names = Vec::with_capacity(enom.variants.len());

    for field in enom.variants.iter() {
        if variant_names.contains(&ctx.ident(field.name)) {
            return Err(ctx.errors(
                "failed to parse enum",
                [
                    Msg::error(ctx.span(field.name), "variant already declared"),
                    Msg::note(
                        ctx.span(enom.name),
                        format!("while parsing `{}`", ctx.ident(enom.name)),
                    ),
                ],
            ));
        }
        variant_names.push(ctx.ident(field.name));
    }

    Ok(Enum {
        span: enom.span,
        name: ctx.store_ident(enom.name),
        variants: enom.variants.iter().map(|f| variant(ctx, f)).collect(),
    })
}

fn variant<'a>(ctx: &mut Ctx<'a>, variant: &rules::Variant) -> Variant {
    Variant {
        span: variant.span,
        name: ctx.store_ident(variant.name),
        //ty: ctx
        //    .ty(field.ty)
        //    .map(|t| FullTy::Ty(t))
        //    .unwrap_or_else(|| FullTy::Struct(ctx.store_ident(field.ty).id)),
    }
}

fn strukt<'a>(
    ctx: &mut Ctx<'a>,
    struct_ids: &HashMap<IdentId, StructId>,
    strukt: &rules::Struct,
) -> Result<Struct, Diag<'a>> {
    let mut field_names = Vec::with_capacity(strukt.fields.len());

    for field in strukt.fields.iter() {
        if field_names.contains(&ctx.ident(field.name)) {
            return Err(ctx.errors(
                "failed to parse struct",
                [
                    Msg::error(ctx.span(field.name), "field already declared"),
                    Msg::note(
                        ctx.span(strukt.name),
                        format!("while parsing `{}`", ctx.ident(strukt.name)),
                    ),
                ],
            ));
        }
        field_names.push(ctx.ident(field.name));
    }

    Ok(Struct {
        span: strukt.span,
        name: ctx.store_ident(strukt.name),
        fields: strukt
            .fields
            .iter()
            .map(|f| field(ctx, struct_ids, f))
            .collect(),
    })
}

fn field<'a>(
    ctx: &mut Ctx<'a>,
    struct_ids: &HashMap<IdentId, StructId>,
    field: &rules::Field,
) -> Field {
    let name = ctx.store_ident(field.ty).id;

    Field {
        span: field.span,
        name: ctx.store_ident(field.name),
        ty: ctx.ty(field.ty).map(|t| FullTy::Ty(t)).unwrap_or_else(|| {
            FullTy::Struct(
                *struct_ids
                    .get(&name)
                    .unwrap_or_else(|| panic!("invalid struct: `{}`", ctx.expect_ident(name),)),
            )
        }),
    }
}

#[derive(Debug, Clone)]
pub struct Func {
    pub sig: Sig,
    pub block: Block,
}

impl Func {
    pub fn hash(&self) -> FuncHash {
        let mut hash = DefaultHasher::new();
        self.sig.hash(&mut hash);
        FuncHash(hash.finish())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FuncHash(u64);

fn func_sig<'a>(ctx: &mut Ctx<'a>, func: &rules::Func) -> Result<Sig, Diag<'a>> {
    let ty = if let Some(ty) = func.ty {
        if let Some(ty) = ctx.ty(ty) {
            FullTy::Ty(ty)
        } else {
            let ident = ctx.store_ident(ty).id;
            if ctx.struct_id(ident).is_none() {
                return Err(ctx.error(
                    SYM_DEF,
                    ctx.span(ty),
                    format!("expected type, got `{}`", ctx.ident(ty)),
                ));
            }

            FullTy::Struct(ctx.expect_struct_id(ident))
        }
    } else {
        FullTy::Ty(Ty::Unit)
    };

    Ok(Sig {
        span: func.span,
        ident: ctx.store_ident(func.name).id,
        ty,
    })
}

fn func<'a>(ctx: &mut Ctx<'a>, func: &rules::Func) -> Result<Func, Diag<'a>> {
    Ok(Func {
        block: block(ctx, &func.block)
            .map_err(|err| err.msg(Msg::note(func.block.span, "while parsing this function")))?,
        sig: func_sig(ctx, func)?,
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
    Call(Call),
    Struct(StructDef),
}

#[derive(Debug, Clone)]
pub enum SemiStmt {
    Let(Let),
    Assign(Assign),
    Ret(Return),
    Bin(BinOp),
    Call(Call),
}

fn stmt<'a>(ctx: &mut Ctx<'a>, stmt: &rules::Stmt) -> Result<Stmt, Diag<'a>> {
    Ok(match stmt {
        rules::Stmt::Let { name, ty, assign } => {
            let ty = if let Some(ty) = ty {
                let ident = ctx.store_ident(*ty).id;
                match ctx.ty(*ty) {
                    Some(prim_ty) => Some((ctx.span(*ty), FullTy::Ty(prim_ty))),
                    None => match ctx.structs.struct_id(ident) {
                        Some(id) => Some((ctx.span(*ty), FullTy::Struct(id))),
                        None => {
                            return Err(ctx.error(
                                SYM_DEF,
                                ctx.span(*ty),
                                format!("`{}` is not a type, expected a type", ctx.ident(*ty)),
                            ));
                        }
                    },
                }
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
                lhs: assign_target(ctx, &assign.lhs)?,
                rhs: let_expr(ctx, &assign.rhs)?,
            })),
            rules::Expr::Ret(span, expr) => Stmt::Semi(SemiStmt::Ret(Return {
                span: *span,
                expr: match expr {
                    Some(expr) => Some(open_stmt(ctx, &expr)?),
                    None => None,
                },
            })),
            rules::Expr::Bin(_, _, _) => Stmt::Semi(SemiStmt::Bin(bin_op(ctx, expr)?)),
            rules::Expr::Call { span, func } => {
                Stmt::Semi(SemiStmt::Call(call(ctx, *span, *func)?))
            }
            expr => todo!("{expr:#?}"),
        },
        rules::Stmt::Open(expr) => Stmt::Open(open_stmt(ctx, &expr)?),
    })
}

fn open_stmt<'a>(ctx: &mut Ctx<'a>, expr: &rules::Expr) -> Result<OpenStmt, Diag<'a>> {
    Ok(match expr {
        rules::Expr::Ident(ident) => OpenStmt::Ident(ctx.store_ident(*ident)),
        rules::Expr::Lit(lit) => OpenStmt::Lit(ctx.store_int(*lit)),
        rules::Expr::Bin(_, _, _) => OpenStmt::Bin(bin_op(ctx, expr)?),
        rules::Expr::Call { span, func } => OpenStmt::Call(call(ctx, *span, *func)?),
        rules::Expr::StructDef(def) => OpenStmt::Struct(struct_def(ctx, def)?),
        _ => todo!(),
    })
}

#[derive(Debug, Clone)]
pub struct Let {
    pub span: Span,
    pub ty: Option<(Span, FullTy)>,
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
    Struct(StructDef),
    Enum(EnumDef),
    Call(Call),
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
        rules::Expr::StructDef(def) => LetExpr::Struct(struct_def(ctx, def)?),
        rules::Expr::EnumDef(def) => LetExpr::Enum(enum_def(ctx, def)?),
        rules::Expr::Call { span, func } => LetExpr::Call(call(ctx, *span, *func)?),
        _ => todo!(),
    })
}

fn enum_def<'a>(ctx: &mut Ctx<'a>, def: &rules::EnumDef) -> Result<EnumDef, Diag<'a>> {
    Ok(EnumDef {
        span: def.span,
        name: ctx.store_ident(def.name),
        variant: variant(ctx, &def.variant),
    })
}

fn struct_def<'a>(ctx: &mut Ctx<'a>, def: &rules::StructDef) -> Result<StructDef, Diag<'a>> {
    let id = ctx.store_ident(def.name).id;

    Ok(StructDef {
        span: def.span,
        id: ctx.expect_struct_id(id),
        fields: def
            .fields
            .iter()
            .map(|f| field_def(ctx, f))
            .collect::<Result<_, _>>()?,
    })
}

fn field_def<'a>(ctx: &mut Ctx<'a>, def: &rules::FieldDef) -> Result<FieldDef, Diag<'a>> {
    Ok(FieldDef {
        span: def.span,
        name: ctx.store_ident(def.name),
        expr: let_expr(ctx, &def.expr)?,
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
    Field,
}

impl BinOpKind {
    pub fn is_primitive(&self) -> bool {
        match self {
            Self::Add | Self::Sub | Self::Mul => true,
            Self::Field => false,
        }
    }

    pub fn is_field(&self) -> bool {
        matches!(self, Self::Field)
    }
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
        rules::Expr::Call { span, func } => BinOpExpr::Call(call(ctx, *span, *func)?),
        _ => todo!(),
    })
}

#[derive(Debug, Clone)]
pub struct Assign {
    pub span: Span,
    pub kind: AssignKind,
    pub lhs: AssignTarget,
    pub rhs: LetExpr,
}

// TODO: more assignment kinds
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssignKind {
    Equals,
    Add,
}

#[derive(Debug, Clone)]
pub enum AssignTarget {
    Ident(Ident),
    Field(BinOp),
}

fn assign_target<'a>(ctx: &mut Ctx<'a>, expr: &rules::Expr) -> Result<AssignTarget, Diag<'a>> {
    Ok(match expr {
        rules::Expr::Ident(ident) => AssignTarget::Ident(ctx.store_ident(*ident)),
        rules::Expr::Bin(bin, _, _) => {
            if bin.kind == BinOpKind::Field {
                AssignTarget::Field(bin_op(ctx, expr)?)
            } else {
                todo!()
            }
        }
        _ => todo!(),
    })
}

#[derive(Debug, Clone)]
pub struct Return {
    pub span: Span,
    pub expr: Option<OpenStmt>,
}

#[derive(Debug, Clone)]
pub struct Call {
    pub span: Span,
    pub sig: Sig,
}

fn call<'a>(ctx: &mut Ctx<'a>, span: Span, name: TokenId) -> Result<Call, Diag<'a>> {
    let id = ctx.store_ident(name).id;
    Ok(Call {
        span,
        sig: *ctx
            .get_sig(id)
            .ok_or_else(|| ctx.error(SYM_DEF, ctx.span(name), "function is not defined"))?,
    })
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
    for func in ctx.funcs.iter() {
        let hash = func.hash();
        for stmt in func.block.stmts.iter() {
            match stmt {
                Stmt::Semi(semi) => match semi {
                    SemiStmt::Let(let_) => match let_.lhs {
                        LetTarget::Ident(ident) => {
                            let var = ty_ctx.var(ident.id, hash);
                            let_expr_constrain(ctx, &mut ty_ctx, var, &let_.rhs, hash);

                            if let Some((span, ty)) = let_.ty {
                                ty_ctx.constrain(
                                    var,
                                    Constraint {
                                        kind: ConstraintKind::full(ty),
                                        span,
                                    },
                                );
                            }
                        }
                    },
                    SemiStmt::Assign(assign) => match &assign.lhs {
                        AssignTarget::Ident(ident) => {
                            let var = ty_ctx.get_var(ident.id, hash);
                            let_expr_constrain(ctx, &mut ty_ctx, var, &assign.rhs, hash);
                        }
                        AssignTarget::Field(field) => match field.lhs {
                            BinOpExpr::Ident(ident) => {
                                //let var = ty_ctx.get_var(ident.id, hash);
                                //let mut accesses = Vec::new();
                                //descend_bin_op_field(ctx, &mut ty_ctx, &field, &mut accesses);
                                //ty_ctx.constrain(
                                //    var,
                                //    Constraint {
                                //        span: ident.span,
                                //        kind: ConstraintKind::Field(
                                //            accesses.split_off(1),
                                //            Box::new(Con),
                                //        ),
                                //    },
                                //);
                                //
                                //assign_expr_constrain_to(
                                //    ctx,
                                //    &mut ty_ctx,
                                //    constraint,
                                //    &assign.rhs,
                                //    hash,
                                //);
                            }
                            _ => todo!(),
                        },
                    },
                    SemiStmt::Bin(bin) => bin_op_constrain_unkown(ctx, &mut ty_ctx, bin, hash),
                    SemiStmt::Ret(ret) => match &ret.expr {
                        Some(OpenStmt::Lit(_)) => {
                            assert!(func.sig.ty.is_ty_and(|ty| ty.is_int()));
                        }
                        Some(OpenStmt::Bin(bin)) => {
                            let constraint = Constraint {
                                span: func.sig.span,
                                kind: ConstraintKind::full(func.sig.ty),
                            };

                            bin_op_constrain_to(ctx, &mut ty_ctx, constraint, bin, hash)
                                .map_err(|d| vec![d])?
                        }
                        Some(OpenStmt::Ident(ident)) => {
                            let var = ty_ctx.get_var(ident.id, hash);
                            ty_ctx.constrain(
                                var,
                                Constraint {
                                    span: func.sig.span,
                                    kind: ConstraintKind::full(func.sig.ty),
                                },
                            );
                        }
                        Some(OpenStmt::Call(Call { sig, .. })) => {
                            if sig.ty != func.sig.ty {
                                todo!("error")
                            }
                        }
                        Some(OpenStmt::Struct(def)) => match func.sig.ty {
                            FullTy::Struct(sig_id) => {
                                assert_eq!(def.id, sig_id);
                            }
                            _ => todo!("error"),
                        },
                        None => {}
                    },
                    SemiStmt::Call(_) => {}
                },
                Stmt::Open(_) => {
                    todo!();
                }
            }
        }

        if let Some(open) = &func.block.end {
            match open {
                OpenStmt::Ident(ident) => {
                    let var = ty_ctx.get_var(ident.id, hash);
                    ty_ctx.constrain(
                        var,
                        Constraint {
                            span: func.sig.span,
                            kind: ConstraintKind::full(func.sig.ty),
                        },
                    );
                }
                OpenStmt::Lit(lit) => {
                    if !func.sig.ty.is_ty_and(|ty| ty.is_int()) {
                        return Err(vec![ctx.errors(
                            "mismatched return type",
                            [
                                Msg::error(
                                    lit.span,
                                    format!("expected `{}`, got `int`", func.sig.ty.as_str(ctx)),
                                ),
                                Msg::note(func.block.span, "because of the signature"),
                            ],
                        )]);
                    }
                }
                OpenStmt::Bin(bin) => {
                    let constraint = Constraint {
                        span: func.sig.span,
                        kind: ConstraintKind::full(func.sig.ty),
                    };

                    bin_op_constrain_to(ctx, &mut ty_ctx, constraint, bin, hash)
                        .map_err(|d| vec![d])?
                }
                OpenStmt::Call(call) => {
                    if call.sig.ty != func.sig.ty {
                        return Err(vec![ctx.errors(
                            "invalid return type",
                            [Msg::error(
                                call.span,
                                format!(
                                    "expected `{}`, got `{}`",
                                    func.sig.ty.as_str(ctx),
                                    call.sig.ty.as_str(ctx)
                                ),
                            )],
                        )]);
                    }
                }
                OpenStmt::Struct(def) => {
                    if FullTy::Struct(def.id) != func.sig.ty {
                        return Err(vec![ctx.errors(
                            "invalid return type",
                            [Msg::error(
                                def.span,
                                format!(
                                    "expected `{}`, got `{}`",
                                    func.sig.ty.as_str(ctx),
                                    ctx.struct_name(def.id)
                                ),
                            )],
                        )]);
                    }
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
                //TyErr::Struct(name) => ctx.error("mismatched types", span, msg)
                _ => todo!(),
            })
            .collect()),
    }
}

fn let_expr_constrain<'a>(
    ctx: &Ctx<'a>,
    ty_ctx: &mut TyCtx,
    var: TyVar,
    expr: &LetExpr,
    hash: FuncHash,
) {
    match expr {
        LetExpr::Lit(lit) => {
            lit_constrain(ctx, ty_ctx, var, lit);
        }
        LetExpr::Bin(bin) => {
            bin_op_constrain(ctx, ty_ctx, var, bin, hash);
        }
        LetExpr::Struct(def) => {
            ty_ctx.constrain(
                var,
                Constraint {
                    span: ctx.structs.strukt(def.id).name.span,
                    kind: ConstraintKind::Struct(def.id),
                },
            );
        }
        LetExpr::Enum(def) => {
            ty_ctx.constrain(
                var,
                Constraint {
                    span: def.name.span,
                    kind: ConstraintKind::EnumVariant(def.name.id, def.variant.name.id),
                },
            );
        }
        LetExpr::Call(call) => {
            ty_ctx.constrain(
                var,
                Constraint {
                    span: call.span,
                    kind: ConstraintKind::full(call.sig.ty),
                },
            );
        }
        LetExpr::Ident(_) => {}
    }
}

fn bin_op_constrain<'a>(
    ctx: &Ctx<'a>,
    ty_ctx: &mut TyCtx,
    var: TyVar,
    bin: &BinOp,
    hash: FuncHash,
) {
    if bin.kind != BinOpKind::Field {
        bin_op_expr_constrain(ctx, ty_ctx, var, &bin.lhs, hash);
        bin_op_expr_constrain(ctx, ty_ctx, var, &bin.rhs, hash);
    }
}

fn bin_op_expr_constrain<'a>(
    ctx: &Ctx<'a>,
    ty_ctx: &mut TyCtx,
    var: TyVar,
    bin: &BinOpExpr,
    hash: FuncHash,
) {
    match bin {
        BinOpExpr::Lit(lit) => {
            lit_constrain(ctx, ty_ctx, var, lit);
        }
        BinOpExpr::Bin(bin) => {
            bin_op_constrain(ctx, ty_ctx, var, bin, hash);
        }
        BinOpExpr::Ident(ident) => ident_constrain(ctx, ty_ctx, var, ident, hash),
        BinOpExpr::Call(call) => {
            ty_ctx.constrain(
                var,
                Constraint {
                    span: call.span,
                    kind: ConstraintKind::full(call.sig.ty),
                },
            );
        }
    }
}

fn bin_op_constrain_to<'a>(
    ctx: &Ctx<'a>,
    ty_ctx: &mut TyCtx,
    constraint: Constraint,
    bin: &BinOp,
    hash: FuncHash,
) -> Result<(), Diag<'a>> {
    match bin.kind {
        BinOpKind::Field => {
            let mut accesses = Vec::new();
            descend_bin_op_field(ctx, ty_ctx, bin, &mut accesses);
            let ident = accesses.first().unwrap();
            let var = ty_ctx.get_var(*ident, hash);
            let span = match bin.lhs {
                BinOpExpr::Ident(ident) => ident.span,
                _ => unreachable!(),
            };

            ty_ctx.constrain(
                var,
                Constraint {
                    kind: ConstraintKind::Field(accesses.split_off(1), Box::new(constraint)),
                    span,
                },
            );
        }
        _ => {
            bin_op_expr_constrain_to(ctx, ty_ctx, constraint.clone(), &bin.lhs, hash)?;
            bin_op_expr_constrain_to(ctx, ty_ctx, constraint, &bin.rhs, hash)?;
        }
    }

    Ok(())
}

fn descend_bin_op_field<'a>(
    ctx: &Ctx<'a>,
    ty_ctx: &mut TyCtx,
    bin: &BinOp,
    accesses: &mut Vec<IdentId>,
) {
    if bin.kind == BinOpKind::Field {
        match bin.lhs {
            BinOpExpr::Ident(ident) => {
                if let BinOpExpr::Bin(bin) = &bin.rhs {
                    accesses.push(ident.id);
                    descend_bin_op_field(ctx, ty_ctx, bin, accesses);
                } else {
                    let BinOpExpr::Ident(other) = bin.rhs else {
                        panic!()
                    };

                    accesses.push(other.id);
                    accesses.push(ident.id);
                }
            }
            _ => {}
        }
    }
}

/// DO NOT CALL
///
/// Use [`bin_op_constrain_to`] instead.
fn bin_op_expr_constrain_to<'a>(
    ctx: &Ctx<'a>,
    ty_ctx: &mut TyCtx,
    constraint: Constraint,
    bin: &BinOpExpr,
    hash: FuncHash,
) -> Result<(), Diag<'a>> {
    match bin {
        BinOpExpr::Lit(lit) => {
            if constraint.kind.is_int().is_some_and(|i| !i) {
                return Err(ctx.error("invalid expression", lit.span, "mismatched types"));
            }
        }
        BinOpExpr::Bin(bin) => {
            bin_op_constrain_to(ctx, ty_ctx, constraint, bin, hash)?;
        }
        BinOpExpr::Ident(ident) => ident_constrain_to(ctx, ty_ctx, constraint, ident, hash),
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
fn bin_op_constrain_unkown<'a>(ctx: &Ctx<'a>, ty_ctx: &mut TyCtx, bin: &BinOp, hash: FuncHash) {
    if let Some(target) = bin_op_expr_find_constrain_target(ctx, ty_ctx, &bin.lhs, hash)
        .or_else(|| bin_op_expr_find_constrain_target(ctx, ty_ctx, &bin.rhs, hash))
    {
        // finding one constraint target is fine, it will equate with other variables
        bin_op_constrain(ctx, ty_ctx, target, bin, hash);
    }
}

fn bin_op_expr_find_constrain_target<'a>(
    ctx: &Ctx<'a>,
    ty_ctx: &mut TyCtx,
    bin: &BinOpExpr,
    hash: FuncHash,
) -> Option<TyVar> {
    match bin {
        BinOpExpr::Bin(bin) => bin_op_expr_find_constrain_target(ctx, ty_ctx, &bin.lhs, hash)
            .or_else(|| bin_op_expr_find_constrain_target(ctx, ty_ctx, &bin.rhs, hash)),
        BinOpExpr::Ident(ident) => Some(ty_ctx.get_var(ident.id, hash)),
        BinOpExpr::Lit(_) => None,
        BinOpExpr::Call(_) => None,
    }
}

fn ident_constrain<'a>(
    _ctx: &Ctx<'a>,
    ty_ctx: &mut TyCtx,
    var: TyVar,
    ident: &Ident,
    hash: FuncHash,
) {
    ty_ctx.constrain(
        var,
        Constraint {
            span: ident.span,
            kind: ConstraintKind::Equate(ty_ctx.get_var(ident.id, hash)),
        },
    );
}

fn ident_constrain_to<'a>(
    _ctx: &Ctx<'a>,
    ty_ctx: &mut TyCtx,
    constraint: Constraint,
    ident: &Ident,
    hash: FuncHash,
) {
    let var = ty_ctx.get_var(ident.id, hash);
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
