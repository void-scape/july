use crate::ir::lit::{Lit, LitKind};
use crate::ir::sig::Sig;
use crate::ir::strukt::StructDef;
use crate::ir::ty::{FullTy, IntKind, Ty};
use crate::ir::*;
use bin::*;
use ctx::*;

mod bin;
pub mod ctx;

/// Analyzed Intermediate Representation.
#[derive(Debug, Clone, Copy)]
pub enum Air {
    Ret,
    Call(Sig),

    /// Swap the A and B registers.
    SwapReg,
    MovIVar(Reg, OffsetVar, IntKind),
    MovIConst(Reg, i64),

    SAlloc(Var, Bytes),

    /// Load address of `Var` into `Reg`.
    Addr(Reg, OffsetVar),

    /// Non overlapping copy.
    MemCpy {
        dst: Reg,
        src: Reg,
        bytes: usize,
    },

    // TODO: rename Push
    PushIConst(OffsetVar, IntKind, i64),
    PushIReg {
        dst: OffsetVar,
        kind: IntKind,
        src: Reg,
    },
    PushIVar {
        dst: OffsetVar,
        kind: IntKind,
        src: OffsetVar,
    },

    /// Binary operations use registers A and B, then store result in A.
    AddAB,
    MulAB,
    SubAB,
}

/// A collection of [`Air`] instructions for a [`crate::ir::Func`].
#[derive(Debug)]
pub struct AirFunc<'a> {
    pub func: &'a Func,
    pub instrs: Vec<Air>,
}

/// A value allocated on the stack. Created by [`Air::SAlloc`]. `Var`
/// represents a location in memory.
///
/// It is up to the consumer of [`Air`] instructions to track variables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Var(usize);

/// Primary registers for binary operations and return values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reg {
    A,
    B,
}

/// A reference optionally `offset` from `var`.
#[derive(Debug, Clone, Copy)]
pub struct OffsetVar {
    pub var: Var,
    pub offset: ByteOffset,
}

impl OffsetVar {
    pub fn new(var: Var, offset: ByteOffset) -> Self {
        Self { var, offset }
    }

    pub fn zero(var: Var) -> Self {
        Self { var, offset: 0 }
    }
}

pub type ByteOffset = usize;
pub type Bytes = usize;

pub fn lower_func<'a>(ctx: &mut AirCtx<'a>, func: &'a Func) -> AirFunc<'a> {
    ctx.start(func);
    for stmt in func.block.stmts.iter() {
        match stmt {
            Stmt::Semi(stmt) => match stmt {
                SemiStmt::Let(let_) => air_let_stmt(ctx, let_),
                SemiStmt::Assign(assign) => air_assign_stmt(ctx, assign),
                SemiStmt::Call(Call { sig, .. }) => ctx.call(sig),
                SemiStmt::Bin(bin) => air_bin_semi(ctx, bin),
                SemiStmt::Ret(ret) => match &ret.expr {
                    Some(expr) => {
                        air_return(ctx, func.sig.ty, &expr);
                    }
                    None => ctx.ins(Air::Ret),
                },
            },
            Stmt::Open(_) => unreachable!(),
        }
    }

    if let Some(end) = &func.block.end {
        air_return(ctx, func.sig.ty, end);
    } else {
        ctx.ins(Air::Ret);
    }

    AirFunc {
        func,
        instrs: ctx.finish(),
    }
}

fn air_bin_semi(ctx: &mut AirCtx, bin: &BinOp) {
    match bin.kind {
        BinOpKind::Field => {}
        BinOpKind::Add | BinOpKind::Sub | BinOpKind::Mul => {
            air_bin_semi_expr(ctx, &bin.lhs);
            air_bin_semi_expr(ctx, &bin.rhs);
        }
    }
}

fn air_bin_semi_expr(ctx: &mut AirCtx, expr: &BinOpExpr) {
    match &expr {
        BinOpExpr::Bin(bin) => air_bin_semi(ctx, bin),
        BinOpExpr::Call(Call { sig, .. }) => ctx.call(sig),
        BinOpExpr::Lit(_) | BinOpExpr::Ident(_) => {}
    }
}

fn air_let_stmt(ctx: &mut AirCtx, stmt: &Let) {
    match stmt.lhs {
        LetTarget::Ident(ident) => {
            let dst = ctx.new_var_registered(ident.id, ctx.ty(ident.id));
            let ty = ctx.ty(ident.id);
            assign_let_expr(ctx, OffsetVar::zero(dst), ty, &stmt.rhs);
        }
    }
}

fn assign_let_expr(ctx: &mut AirCtx, dst: OffsetVar, ty: FullTy, expr: &LetExpr) {
    match &expr {
        LetExpr::Lit(lit) => {
            assign_lit(ctx, lit, dst, ty);
        }
        LetExpr::Bin(bin) => {
            assign_bin_op(ctx, dst, ty, bin);
        }
        LetExpr::Struct(def) => {
            define_struct(ctx, def, dst);
        }
        LetExpr::Call(Call { sig, .. }) => {
            ctx.ins(Air::Call(*sig));
            match sig.ty {
                FullTy::Ty(ty) => {
                    let kind = ty.expect_int();
                    ctx.ins(Air::PushIReg {
                        dst,
                        kind,
                        src: Reg::A,
                    });
                }
                FullTy::Struct(s) => {
                    let bytes = ctx.structs.layout(s).size;
                    ctx.ins_set(&[
                        Air::Addr(Reg::B, dst),
                        Air::MemCpy {
                            dst: Reg::B,
                            src: {
                                // the destination is supplied by the callee
                                const _: () = assert!(matches!(RET_REG, Reg::A));
                                Reg::A
                            },
                            bytes,
                        },
                    ]);
                }
            }
        }
        LetExpr::Ident(ident) => {
            let other = OffsetVar::zero(ctx.expect_var(ident.id));

            match ty {
                FullTy::Ty(ty) => match ty {
                    Ty::Int(kind) => {
                        ctx.ins(Air::PushIVar {
                            dst,
                            kind,
                            src: other,
                        });
                    }
                    _ => unreachable!(),
                },
                FullTy::Struct(id) => {
                    let bytes = ctx.structs.layout(id).size;
                    ctx.ins_set(&[
                        Air::Addr(Reg::B, dst),
                        Air::Addr(Reg::A, other),
                        Air::MemCpy {
                            dst: Reg::B,
                            src: Reg::A,
                            bytes,
                        },
                    ]);
                }
            }
        }
        LetExpr::Enum(_) => {
            todo!()
        }
    }
}

fn define_struct(ctx: &mut AirCtx, def: &StructDef, dst: OffsetVar) {
    for field in def.fields.iter() {
        let strukt = ctx.structs.strukt(def.id);
        let field_offset = strukt.field_offset(ctx, field.name.id);
        let ty = strukt.field_ty(field.name.id);

        assign_let_expr(
            ctx,
            OffsetVar::new(dst.var, dst.offset + field_offset as usize),
            ty,
            &field.expr,
        );
    }
}

fn air_assign_stmt(ctx: &mut AirCtx, stmt: &Assign) {
    let (var, ty) = match &stmt.lhs {
        AssignTarget::Ident(ident) => (OffsetVar::zero(ctx.expect_var(ident.id)), ctx.ty(ident.id)),
        AssignTarget::Field(bin) => aquire_bin_field_offset(ctx, &bin),
    };

    match stmt.kind {
        AssignKind::Equals => {
            assign_let_expr(ctx, var, ty, &stmt.rhs);
        }
        AssignKind::Add => {
            let tmp = OffsetVar::zero(ctx.anon_var(ty));
            let kind = ty.expect_ty().expect_int();
            ctx.ins(Air::PushIVar {
                dst: tmp,
                kind,
                src: var,
            });
            assign_let_expr(ctx, var, ty, &stmt.rhs);
            add!(ctx, kind, var, tmp, var);
        }
    }
}

#[track_caller]
fn assign_lit(ctx: &mut AirCtx, lit: &Lit, var: OffsetVar, ty: FullTy) {
    match ty {
        FullTy::Ty(ty) => match ty {
            Ty::Int(kind) => match ctx.expect_lit(lit.kind) {
                LitKind::Int(int) => {
                    ctx.ins(Air::PushIConst(var, kind, int));
                }
                other => panic!("cannot assign int to `{other:?}`"),
            },
            _ => unreachable!(),
        },
        FullTy::Struct(_) => {
            panic!("cannot assign lit to struct");
        }
    }
}

fn air_return(ctx: &mut AirCtx, ty: FullTy, end: &OpenStmt) {
    match end {
        OpenStmt::Lit(lit) => match ctx.expect_lit(lit.kind) {
            LitKind::Int(int) => {
                ctx.ret_iconst(int);
            }
            _ => unreachable!(),
        },
        OpenStmt::Ident(ident) => {
            let var = OffsetVar::zero(ctx.expect_var(ident.id));
            let out_ty = ctx.expect_var_ty(var.var);
            assert_eq!(out_ty, ty);

            match out_ty {
                FullTy::Ty(ty) => {
                    ctx.ret_ivar(var, ty.expect_int());
                }
                FullTy::Struct(_) => ctx.ret_ptr(var),
            }
        }
        OpenStmt::Bin(bin) => {
            if bin.kind.is_primitive() {
                match ty {
                    FullTy::Ty(ty) => {
                        let dst = OffsetVar::zero(ctx.anon_var(FullTy::Ty(ty)));
                        assign_prim_bin_op(ctx, dst, ty.expect_int(), bin);
                        ctx.ret_ivar(dst, ty.expect_int());
                    }
                    FullTy::Struct(_) => unreachable!(),
                }
            } else {
                match bin.kind {
                    BinOpKind::Field => {
                        let (field_var, field_ty) = aquire_bin_field_offset(ctx, bin);
                        assert_eq!(ty, field_ty);
                        match field_ty {
                            FullTy::Ty(kind) => {
                                ctx.ret_ivar(field_var, kind.expect_int());
                            }
                            FullTy::Struct(_) => {
                                ctx.ret_ptr(field_var);
                            }
                        }
                    }
                    k => unreachable!("{k:?}"),
                }
            }
        }
        OpenStmt::Call(Call { sig, .. }) => {
            assert_eq!(sig.ty, ty);
            ctx.ins(Air::Call(*sig));
            ctx.ins(Air::Ret);
        }
        OpenStmt::Struct(def) => {
            let dst = OffsetVar::zero(ctx.anon_var(FullTy::Struct(def.id)));
            define_struct(ctx, def, dst);
            ctx.ret_ptr(dst);
        }
    }
}
