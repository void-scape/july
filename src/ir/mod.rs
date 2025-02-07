use self::ident::IdentId;
use self::sig::Sig;
use self::strukt::{Field, FieldDef, Struct, StructDef};
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
use std::hash::{DefaultHasher, Hash, Hasher};

pub mod ctx;
pub mod ident;
pub mod lit;
pub mod sig;
pub mod strukt;
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
        .map(|s| strukt(&mut ctx, s))
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(structs) => {
            ctx.store_structs(structs);
            ctx.layout_structs();
        }
        Err(diag) => {
            diagnostic::report(diag);
            panic!()
        }
    }

    match items
        .iter()
        .filter_map(|i| match i {
            Item::Func(func) => Some(func),
            _ => None,
        })
        .map(|f| func_sig(&mut ctx, f))
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(sigs) => {
            for sig in sigs.into_iter() {
                ctx.store_sig(sig);
            }
        }
        Err(diag) => {
            diagnostic::report(diag);
            panic!()
        }
    }

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

fn strukt<'a>(ctx: &mut Ctx<'a>, strukt: &rules::Struct) -> Result<Struct, Diag<'a>> {
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
        fields: strukt.fields.iter().map(|f| field(ctx, f)).collect(),
    })
}

fn field<'a>(ctx: &mut Ctx<'a>, field: &rules::Field) -> Field {
    Field {
        span: field.span,
        name: ctx.store_ident(field.name),
        ty: ctx
            .ty(field.ty)
            .map(|t| FullTy::Ty(t))
            .unwrap_or_else(|| FullTy::Struct(ctx.store_ident(field.ty).id)),
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

            FullTy::Struct(ident)
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
            rules::Expr::Call { span, func } => {
                Stmt::Semi(SemiStmt::Call(call(ctx, *span, *func)?))
            }
            expr => todo!("{expr:#?}"),
        },
        rules::Stmt::Open(expr) => match expr {
            rules::Expr::Ident(ident) => Stmt::Open(OpenStmt::Ident(ctx.store_ident(*ident))),
            rules::Expr::Lit(lit) => Stmt::Open(OpenStmt::Lit(ctx.store_int(*lit))),
            rules::Expr::Bin(_, _, _) => Stmt::Open(OpenStmt::Bin(bin_op(ctx, expr)?)),
            rules::Expr::Call { span, func } => {
                Stmt::Open(OpenStmt::Call(call(ctx, *span, *func)?))
            }
            rules::Expr::StructDef(def) => Stmt::Open(OpenStmt::Struct(struct_def(ctx, def)?)),
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
    Struct(StructDef),
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
        rules::Expr::Call { span, func } => LetExpr::Call(call(ctx, *span, *func)?),
        _ => todo!(),
    })
}

fn struct_def<'a>(ctx: &mut Ctx<'a>, def: &rules::StructDef) -> Result<StructDef, Diag<'a>> {
    Ok(StructDef {
        span: def.span,
        name: ctx.store_ident(def.name),
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
                                        kind: ConstraintKind::Abs(ty),
                                        span,
                                    },
                                );
                            }
                        }
                    },
                    SemiStmt::Assign(assign) => match assign.lhs {
                        AssignTarget::Ident(ident) => {
                            let var = ty_ctx.get_var(ident.id, hash);
                            assign_expr_constrain(ctx, &mut ty_ctx, var, &assign.rhs, hash);
                        }
                    },
                    SemiStmt::Bin(bin) => bin_op_constrain_unkown(ctx, &mut ty_ctx, bin, hash),
                    SemiStmt::Ret(ret) => match &ret.expr {
                        RetExpr::Unit => {
                            assert!(func.sig.ty.is_ty_and(|ty| ty.is_unit()));
                        }
                        RetExpr::Lit(_) => {
                            assert!(func.sig.ty.is_ty_and(|ty| ty.is_int()));
                        }
                        RetExpr::Bin(bin) => {
                            let constraint = Constraint {
                                span: func.sig.span,
                                kind: ConstraintKind::full(func.sig.ty),
                            };

                            bin_op_constrain_to(ctx, &mut ty_ctx, constraint, bin, hash)
                                .map_err(|d| vec![d])?
                        }
                        RetExpr::Ident(ident) => {
                            let var = ty_ctx.get_var(ident.id, hash);
                            ty_ctx.constrain(
                                var,
                                Constraint {
                                    span: func.sig.span,
                                    kind: ConstraintKind::full(func.sig.ty),
                                },
                            );
                        }
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
                    if FullTy::Struct(def.name.id) != func.sig.ty {
                        return Err(vec![ctx.errors(
                            "invalid return type",
                            [Msg::error(
                                def.span,
                                format!(
                                    "expected `{}`, got `{}`",
                                    func.sig.ty.as_str(ctx),
                                    ctx.expect_ident(def.name.id)
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
                    span: def.name.span,
                    kind: ConstraintKind::Struct(def.name.id),
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
        LetExpr::Ident(_) => todo!(),
    }
}

fn assign_expr_constrain<'a>(
    ctx: &Ctx<'a>,
    ty_ctx: &mut TyCtx,
    var: TyVar,
    expr: &AssignExpr,
    hash: FuncHash,
) {
    match expr {
        AssignExpr::Lit(lit) => {
            lit_constrain(ctx, ty_ctx, var, lit);
        }
        AssignExpr::Bin(bin) => {
            bin_op_constrain(ctx, ty_ctx, var, bin, hash);
        }
        AssignExpr::Ident(_) => todo!(),
    }
}

fn bin_op_constrain<'a>(
    ctx: &Ctx<'a>,
    ty_ctx: &mut TyCtx,
    var: TyVar,
    bin: &BinOp,
    hash: FuncHash,
) {
    bin_op_expr_constrain(ctx, ty_ctx, var, &bin.lhs, hash);
    bin_op_expr_constrain(ctx, ty_ctx, var, &bin.rhs, hash);
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
