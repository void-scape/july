use crate::ir::lit::{Lit, LitKind};
use crate::ir::sig::{Param, Sig};
use crate::ir::strukt::StructDef;
use crate::ir::ty::store::TyId;
use crate::ir::ty::{IWidth, IntTy, Sign, Ty};
use crate::ir::*;
use bin::*;
use ctx::*;

mod bin;
pub mod ctx;

/// Analyzed Intermediate Representation.
///
/// `Air` is a collection of low level instructions that are intended to be easily executable as
/// byte-code and lowerable in a backend.
#[derive(Debug, Clone)]
pub enum Air<'a> {
    Ret,
    Call(&'a Sig, Args),

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

/// Collection of [`Air`] instructions for a [`crate::ir::Func`].
#[derive(Debug)]
pub struct AirFunc<'a> {
    pub func: &'a Func,
    pub instrs: Vec<Air<'a>>,
}

#[derive(Debug, Default, Clone)]
pub struct Args {
    pub vars: Vec<OffsetVar>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntKind {
    I8,
    I16,
    I32,
    I64,

    U8,
    U16,
    U32,
    U64,
}

impl IntKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
        }
    }
}

impl IntTy {
    pub fn kind(&self) -> IntKind {
        match self.sign {
            Sign::I => match self.width {
                IWidth::W8 => IntKind::I8,
                IWidth::W16 => IntKind::I16,
                IWidth::W32 => IntKind::I32,
                IWidth::W64 => IntKind::I64,
            },
            Sign::U => match self.width {
                IWidth::W8 => IntKind::U8,
                IWidth::W16 => IntKind::U16,
                IWidth::W32 => IntKind::U32,
                IWidth::W64 => IntKind::U64,
            },
        }
    }
}

pub fn lower_func<'a, 'b>(ctx: &'b mut AirCtx<'a>, func: &'a Func) -> AirFunc<'a> {
    ctx.start(func);
    init_params(ctx, func);
    for stmt in func.block.stmts.iter() {
        match stmt {
            Stmt::Semi(stmt) => match stmt {
                SemiStmt::Let(let_) => air_let_stmt(ctx, let_),
                SemiStmt::Assign(assign) => air_assign_stmt(ctx, assign),
                SemiStmt::Call(call @ Call { sig, .. }) => {
                    let args = generate_args(ctx, call);
                    ctx.call(sig, args)
                }
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

fn init_params(ctx: &mut AirCtx, func: &Func) {
    for Param { ident, ty, .. } in func.sig.params.iter() {
        if ctx.get_var(ident.id).is_none() {
            ctx.new_var_registered_no_salloc(ident.id, *ty);
        }
    }
}

fn air_bin_semi<'a>(ctx: &mut AirCtx<'a>, bin: &'a BinOp) {
    match bin.kind {
        BinOpKind::Field => {}
        BinOpKind::Add | BinOpKind::Sub | BinOpKind::Mul => {
            air_bin_semi_expr(ctx, &bin.lhs);
            air_bin_semi_expr(ctx, &bin.rhs);
        }
    }
}

fn air_bin_semi_expr<'a>(ctx: &mut AirCtx<'a>, expr: &'a Expr) {
    match &expr {
        Expr::Bin(bin) => air_bin_semi(ctx, bin),
        Expr::Call(call @ Call { sig, .. }) => {
            let args = generate_args(ctx, call);
            ctx.call(sig, args)
        }
        Expr::Lit(_) | Expr::Ident(_) => {}
        _ => todo!(),
    }
}

fn air_let_stmt<'a>(ctx: &mut AirCtx<'a>, stmt: &'a Let) {
    match stmt.lhs {
        LetTarget::Ident(ident) => {
            let ty = ctx.var_ty(ident.id);
            let dst = ctx.new_var_registered(ident.id, ty);
            assign_expr(ctx, OffsetVar::zero(dst), ty, &stmt.rhs);
        }
    }
}

fn assign_expr<'a>(ctx: &mut AirCtx<'a>, dst: OffsetVar, ty: TyId, expr: &'a Expr) {
    match &expr {
        Expr::Lit(lit) => {
            assign_lit(ctx, lit, dst, ty);
        }
        Expr::Bin(bin) => {
            assign_bin_op(ctx, dst, ty, bin);
        }
        Expr::Struct(def) => {
            define_struct(ctx, def, dst);
        }
        Expr::Call(call) => {
            let args = generate_args(ctx, call);
            ctx.ins(Air::Call(&call.sig, args));
            match ctx.tys.ty(call.sig.ty) {
                Ty::Int(ty) => {
                    ctx.ins(Air::PushIReg {
                        dst,
                        kind: ty.kind(),
                        src: Reg::A,
                    });
                }
                Ty::Struct(s) => {
                    let bytes = ctx.tys.struct_layout(*s).size;
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
                Ty::Unit => todo!(),
            }
        }
        Expr::Ident(ident) => {
            let other = OffsetVar::zero(ctx.expect_var(ident.id));

            match ctx.tys.ty(ty) {
                Ty::Int(ty) => {
                    ctx.ins(Air::PushIVar {
                        dst,
                        kind: ty.kind(),
                        src: other,
                    });
                }
                Ty::Struct(id) => {
                    let bytes = ctx.tys.struct_layout(*id).size;
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
                Ty::Unit => todo!(),
            }
        }
        Expr::Enum(_) => {
            todo!()
        }
    }
}

fn generate_args<'a>(ctx: &mut AirCtx<'a>, call: &'a Call) -> Args {
    assert_eq!(call.args.args.len(), call.sig.params.len());
    if call.args.is_empty() {
        return Args::default();
    }

    let mut args = Args {
        vars: Vec::with_capacity(call.sig.params.len()),
    };

    for (expr, param) in call.args.args.iter().zip(call.sig.params.iter()) {
        //let arg = OffsetVar::zero(ctx.new_var_registered(param.ident.id, param.ty));

        let the_fn_param = match ctx.get_var_with_hash(param.ident.id, call.sig.hash()) {
            Some(var) => {
                ctx.ins(Air::SAlloc(var, ctx.tys.ty(param.ty).size(ctx)));
                OffsetVar::zero(var)
            }
            None => OffsetVar::zero(ctx.new_var_registered_with_hash(
                param.ident.id,
                call.sig.hash(),
                param.ty,
            )),
        };

        assign_expr(ctx, the_fn_param, param.ty, expr);

        //let bytes = ctx.tys.ty(param.ty).size(ctx);
        //ctx.ins_set(&[
        //    Air::Addr(Reg::B, the_fn_param),
        //    Air::Addr(Reg::A, arg),
        //    Air::MemCpy {
        //        dst: Reg::B,
        //        src: Reg::A,
        //        bytes,
        //    },
        //]);

        args.vars.push(the_fn_param);
    }

    args
}

fn define_struct<'a>(ctx: &mut AirCtx<'a>, def: &'a StructDef, dst: OffsetVar) {
    for field in def.fields.iter() {
        let strukt = ctx.tys.strukt(def.id);
        let field_offset = strukt.field_offset(ctx, field.name.id);
        let ty = strukt.field_ty(field.name.id);

        assign_expr(
            ctx,
            OffsetVar::new(dst.var, dst.offset + field_offset as usize),
            ty,
            &field.expr,
        );
    }
}

fn air_assign_stmt<'a>(ctx: &mut AirCtx<'a>, stmt: &'a Assign) {
    let (var, ty) = match &stmt.lhs {
        AssignTarget::Ident(ident) => (
            OffsetVar::zero(ctx.expect_var(ident.id)),
            ctx.var_ty(ident.id),
        ),
        AssignTarget::Field(bin) => aquire_bin_field_offset(ctx, &bin),
    };

    match stmt.kind {
        AssignKind::Equals => {
            assign_expr(ctx, var, ty, &stmt.rhs);
        }
        AssignKind::Add => {
            let tmp = OffsetVar::zero(ctx.anon_var(ty));
            let kind = ctx.tys.ty(ty).expect_int().kind();
            ctx.ins(Air::PushIVar {
                dst: tmp,
                kind,
                src: var,
            });
            assign_expr(ctx, var, ty, &stmt.rhs);
            add!(ctx, kind, var, tmp, var);
        }
    }
}

#[track_caller]
fn assign_lit(ctx: &mut AirCtx, lit: &Lit, var: OffsetVar, ty: TyId) {
    match ctx.tys.ty(ty) {
        Ty::Int(int_ty) => match ctx.expect_lit(lit.kind) {
            LitKind::Int(int) => {
                ctx.ins(Air::PushIConst(var, int_ty.kind(), int));
            }
            other => panic!("cannot assign int to `{other:?}`"),
        },
        Ty::Unit => panic!("cannot assign lit to unit"),
        Ty::Struct(_) => panic!("cannot assign lit to struct"),
    }
}

fn air_return<'a>(ctx: &mut AirCtx<'a>, ty: TyId, end: &'a OpenStmt) {
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

            match ctx.tys.ty(out_ty) {
                Ty::Int(int) => {
                    ctx.ret_ivar(var, int.kind());
                }
                Ty::Struct(_) => ctx.ret_ptr(var),
                Ty::Unit => todo!(),
            }
        }
        OpenStmt::Bin(bin) => {
            if bin.kind.is_primitive() {
                match ctx.tys.ty(ty) {
                    Ty::Int(int) => {
                        let kind = int.kind();
                        let dst = OffsetVar::zero(ctx.anon_var(ty));
                        assign_prim_bin_op(ctx, dst, kind, bin);
                        ctx.ret_ivar(dst, kind);
                    }
                    Ty::Struct(_) | Ty::Unit => unreachable!(),
                }
            } else {
                match bin.kind {
                    BinOpKind::Field => {
                        let (field_var, field_ty) = aquire_bin_field_offset(ctx, bin);
                        assert_eq!(ty, field_ty);
                        match ctx.tys.ty(field_ty) {
                            Ty::Int(int) => {
                                ctx.ret_ivar(field_var, int.kind());
                            }
                            Ty::Struct(_) => {
                                ctx.ret_ptr(field_var);
                            }
                            Ty::Unit => unreachable!(),
                        }
                    }
                    k => unreachable!("{k:?}"),
                }
            }
        }
        OpenStmt::Call(call) => {
            assert_eq!(call.sig.ty, ty);
            let args = generate_args(ctx, call);
            ctx.ins(Air::Call(&call.sig, args));
            ctx.ins(Air::Ret);
        }
        OpenStmt::Struct(def) => {
            let dst = OffsetVar::zero(ctx.anon_var(ctx.tys.struct_ty_id(def.id)));
            define_struct(ctx, def, dst);
            ctx.ret_ptr(dst);
        }
    }
}
