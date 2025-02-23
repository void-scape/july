use super::ctx::AirCtx;
use super::{IntKind, OffsetVar};
use crate::air::ctx::RET_REG;
use crate::air::{generate_args, Air, Reg};
use crate::ir::lit::LitKind;
use crate::ir::ty::store::TyId;
use crate::ir::ty::Ty;
use crate::ir::*;

/// Evaluate `bin` and assign to `dst`.
///
/// Panics
///     `bin` does not evaluate to `ty`
pub fn assign_bin_op<'a>(ctx: &mut AirCtx<'a>, dst: OffsetVar, ty: TyId, bin: &'a BinOp) {
    let ty = ctx.tys.ty(ty);

    if bin.kind.is_primitive() {
        let kind = ty.expect_int().kind();
        assign_prim_bin_op(ctx, dst, kind, bin);
    } else {
        match bin.kind {
            BinOpKind::Field => {
                let (field_var, field_ty) = aquire_bin_field_offset(ctx, bin);
                match ctx.tys.ty(field_ty) {
                    Ty::Int(int_ty) => {
                        assert_eq!(ty.expect_int().kind(), int_ty.kind());
                        ctx.ins(Air::PushIVar {
                            dst,
                            kind: int_ty.kind(),
                            src: field_var,
                        });
                    }
                    Ty::Struct(id) => {
                        let bytes = ctx.tys.struct_layout(*id).size;
                        ctx.ins_set(&[
                            Air::Addr(Reg::B, dst),
                            Air::Addr(Reg::A, field_var),
                            Air::MemCpy {
                                dst: Reg::B,
                                src: Reg::A,
                                bytes,
                            },
                        ]);
                    }
                    Ty::Unit => panic!("field type cannot be unit"),
                }
            }
            _ => unreachable!(),
        }
    }
}

/// Evalute `bin` and assign to `dst`.
///
/// Panics
///     `bin` is not primitive
///     `bin` does not evaluate to `kind`
#[track_caller]
pub fn assign_prim_bin_op<'a>(ctx: &mut AirCtx<'a>, dst: OffsetVar, kind: IntKind, bin: &'a BinOp) {
    assert!(bin.kind.is_primitive());
    match bin.kind {
        BinOpKind::Add => Add(kind).visit(ctx, kind, dst, bin),
        BinOpKind::Mul => Mul(kind).visit(ctx, kind, dst, bin),
        BinOpKind::Sub => Sub(kind).visit(ctx, kind, dst, bin),
        _ => unreachable!(),
    }
}

/// Descend `bin`s field accesses and return the destination offset and type.
///
/// Panics
///     `bin` is not a field access
///     field access is invalid
#[track_caller]
pub fn aquire_bin_field_offset(ctx: &AirCtx, bin: &BinOp) -> (OffsetVar, TyId) {
    assert_eq!(bin.kind, BinOpKind::Field);

    let mut accesses = Vec::new();
    descend_bin_op_field(ctx, bin, &mut accesses);

    let var = ctx.expect_var(accesses.first().unwrap().id);
    let id = ctx.tys.ty(ctx.expect_var_ty(var)).expect_struct();
    let mut strukt = ctx.tys.strukt(id);

    let mut offset = 0;
    for (i, access) in accesses.iter().skip(1).enumerate() {
        let ty = strukt.field_ty(access.id);
        if i == accesses.len() - 2 {
            let field_offset = strukt.field_offset(ctx, access.id);
            return (OffsetVar::new(var, (offset + field_offset) as usize), ty);
        }

        match ctx.tys.ty(ty) {
            Ty::Struct(id) => {
                let field_offset = strukt.field_offset(ctx, access.id);
                offset += field_offset;
                strukt = ctx.tys.strukt(*id);
            }
            Ty::Int(t) => panic!("{t:?} has not fields"),
            Ty::Unit => panic!("structs cannot contain unit fields"),
        }
    }
    unreachable!()
}

macro_rules! impl_op {
    ($name:ident, $strukt:ident) => {
        #[macro_export]
        macro_rules! $name {
            ($ctx:ident, $kind:ident, $dst:ident, $lhs:expr, $rhs:expr) => {{
                use crate::air::bin::BinOpLeaf;
                crate::air::bin::$strukt($kind).visit_leaf($ctx, $dst, $lhs, $rhs);
            }};
        }
        #[allow(unused)]
        pub use $name;
    };
}

impl_op!(add, Add);
impl_op!(sub, Sub);
impl_op!(mul, Mul);

crate::impl_prim_bin_op_visitor!(Add, AddAB, +);
crate::impl_prim_bin_op_visitor!(Sub, SubAB, -);
crate::impl_prim_bin_op_visitor!(Mul, MulAB, *);

trait BinOpVisitor
where
    Self: AllBinOpLeaves,
{
    fn visit<'a>(&self, ctx: &mut AirCtx<'a>, kind: IntKind, dst: OffsetVar, bin: &'a BinOp) {
        assert!(bin.kind.is_primitive());
        match &bin.lhs {
            Expr::Lit(lit) => match ctx.expect_lit(lit.kind) {
                LitKind::Int(lhs_lit) => match &bin.rhs {
                    Expr::Lit(lit) => match ctx.expect_lit(lit.kind) {
                        LitKind::Int(rhs_lit) => {
                            self.visit_leaf(ctx, dst, lhs_lit, rhs_lit);
                        }
                        _ => unreachable!(),
                    },
                    Expr::Bin(inner_bin) => {
                        if inner_bin.kind.is_field() {
                            let (field_var, field_ty) = aquire_bin_field_offset(ctx, inner_bin);
                            let field_ty = ctx.tys.ty(field_ty);
                            let kind = field_ty.expect_int().kind();
                            match bin.kind {
                                BinOpKind::Add => add!(ctx, kind, dst, lhs_lit, field_var),
                                BinOpKind::Sub => sub!(ctx, kind, dst, lhs_lit, field_var),
                                BinOpKind::Mul => mul!(ctx, kind, dst, lhs_lit, field_var),
                                BinOpKind::Field => unreachable!(),
                            }
                        } else {
                            assign_prim_bin_op(ctx, dst, kind, inner_bin);
                            self.visit_leaf(ctx, dst, lhs_lit, dst);
                        }
                    }
                    Expr::Call(call @ Call { sig, .. }) => {
                        let sig_ty = ctx.tys.ty(sig.ty);
                        assert!(sig_ty.is_int() && sig_ty.expect_int().kind() == kind);
                        let args = generate_args(ctx, call);
                        ctx.ins(Air::Call(sig, args));
                        self.visit_leaf(ctx, dst, lhs_lit, Reg::A);
                    }
                    Expr::Ident(ident) => {
                        let rhs = ctx.expect_var(ident.id);
                        self.visit_leaf(ctx, dst, lhs_lit, OffsetVar::zero(rhs));
                    }
                    Expr::Struct(_) | Expr::Enum(_) => unimplemented!(),
                },
                _ => unreachable!(),
            },
            Expr::Ident(ident) => {
                let var = OffsetVar::zero(ctx.expect_var(ident.id));
                match &bin.rhs {
                    Expr::Lit(lit) => match ctx.expect_lit(lit.kind) {
                        LitKind::Int(lit) => {
                            self.visit_leaf(ctx, dst, var, lit);
                        }
                        _ => unreachable!(),
                    },
                    Expr::Bin(inner_bin) => {
                        if inner_bin.kind.is_field() {
                            let (field_var, field_ty) = aquire_bin_field_offset(ctx, inner_bin);
                            let field_ty = ctx.tys.ty(field_ty);
                            let kind = field_ty.expect_int().kind();
                            match bin.kind {
                                BinOpKind::Add => add!(ctx, kind, dst, var, field_var),
                                BinOpKind::Sub => sub!(ctx, kind, dst, var, field_var),
                                BinOpKind::Mul => mul!(ctx, kind, dst, var, field_var),
                                BinOpKind::Field => unreachable!(),
                            }
                        } else {
                            assign_prim_bin_op(ctx, dst, kind, inner_bin);
                            self.visit_leaf(ctx, dst, var, dst);
                        }
                    }
                    Expr::Ident(ident) => {
                        self.visit_leaf(ctx, dst, var, OffsetVar::zero(ctx.expect_var(ident.id)));
                    }
                    Expr::Call(call @ Call { sig, .. }) => {
                        let sig_ty = ctx.tys.ty(sig.ty);
                        assert!(sig_ty.is_int() && sig_ty.expect_int().kind() == kind);
                        let args = generate_args(ctx, call);
                        ctx.ins(Air::Call(sig, args));
                        self.visit_leaf(ctx, dst, var, Reg::A);
                    }
                    Expr::Struct(_) | Expr::Enum(_) => unimplemented!(),
                }
            }
            Expr::Call(call @ Call { sig, .. }) => {
                let sig_ty = ctx.tys.ty(sig.ty);
                assert!(sig_ty.is_int() && sig_ty.expect_int().kind() == kind);
                match &bin.rhs {
                    Expr::Lit(lit) => match ctx.expect_lit(lit.kind) {
                        LitKind::Int(lit) => {
                            let args = generate_args(ctx, call);
                            ctx.ins(Air::Call(sig, args));
                            self.visit_leaf(ctx, dst, Reg::A, lit);
                        }
                        _ => unreachable!(),
                    },
                    Expr::Bin(inner_bin) => {
                        if inner_bin.kind.is_field() {
                            let (field_var, field_ty) = aquire_bin_field_offset(ctx, inner_bin);
                            let args = generate_args(ctx, call);
                            ctx.ins(Air::Call(sig, args));
                            let field_ty = ctx.tys.ty(field_ty);
                            const _: () = assert!(matches!(RET_REG, Reg::A));
                            let kind = field_ty.expect_int().kind();
                            match bin.kind {
                                BinOpKind::Add => add!(ctx, kind, dst, Reg::A, field_var),
                                BinOpKind::Sub => sub!(ctx, kind, dst, Reg::A, field_var),
                                BinOpKind::Mul => mul!(ctx, kind, dst, Reg::A, field_var),
                                BinOpKind::Field => unreachable!(),
                            }
                        } else {
                            assign_prim_bin_op(ctx, dst, kind, inner_bin);
                            let args = generate_args(ctx, call);
                            ctx.ins(Air::Call(sig, args));
                            const _: () = assert!(matches!(RET_REG, Reg::A));
                            self.visit_leaf(ctx, dst, Reg::A, dst);
                        }
                    }
                    Expr::Ident(ident) => {
                        let args = generate_args(ctx, call);
                        ctx.ins(Air::Call(sig, args));
                        self.visit_leaf(
                            ctx,
                            dst,
                            Reg::A,
                            OffsetVar::zero(ctx.expect_var(ident.id)),
                        );
                    }
                    Expr::Call(other) => {
                        let sig_ty = sig.ty;
                        let tmp = OffsetVar::zero(ctx.anon_var(sig_ty));

                        let sig_ty = ctx.tys.ty(sig_ty);
                        assert!(sig_ty.is_int() && sig_ty.expect_int().kind() == kind);
                        const _: () = assert!(matches!(RET_REG, Reg::A));
                        let kind = sig_ty.expect_int().kind();

                        let args = generate_args(ctx, call);
                        let other_args = generate_args(ctx, other);
                        ctx.ins_set(&[
                            Air::Call(sig, args),
                            Air::PushIReg {
                                dst: tmp,
                                kind,
                                src: Reg::A,
                            },
                            Air::Call(&other.sig, other_args),
                            Air::MovIVar(Reg::B, tmp, kind),
                        ]);
                        self.visit_leaf(ctx, dst, Reg::B, Reg::A);
                    }
                    Expr::Struct(_) | Expr::Enum(_) => unimplemented!(),
                }
            }
            Expr::Struct(_) | Expr::Enum(_) => unimplemented!(),
            Expr::Bin(_) => unreachable!(),
        }
    }
}

pub trait BinOpLeaf<L, R> {
    fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: L, rhs: R);
}

trait AllBinOpLeaves:
    BinOpLeaf<i64, i64>
    + BinOpLeaf<i64, OffsetVar>
    + BinOpLeaf<OffsetVar, i64>
    + BinOpLeaf<OffsetVar, OffsetVar>
    + BinOpLeaf<Reg, OffsetVar>
    + BinOpLeaf<OffsetVar, Reg>
    + BinOpLeaf<Reg, i64>
    + BinOpLeaf<i64, Reg>
    + BinOpLeaf<Reg, Reg>
{
}

#[macro_export]
macro_rules! impl_prim_bin_op_visitor {
    ($name:ident, $instr:ident, $op:tt) => {
        pub struct $name(pub IntKind);

        impl BinOpVisitor for $name {}

        impl AllBinOpLeaves for $name {}

        impl BinOpLeaf<i64, i64> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: i64, rhs: i64) {
                ctx.ins(Air::PushIConst(dst, self.0, lhs $op rhs));
            }
        }

        impl BinOpLeaf<i64, OffsetVar> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: i64, rhs: OffsetVar) {
                ctx.ins_set(&[
                    Air::MovIVar(Reg::A, rhs, self.0),
                    Air::MovIConst(Reg::B, lhs),
                    Air::$instr,
                    Air::PushIReg {
                        dst,
                        kind: self.0,
                        src: Reg::A,
                    },
                ]);
            }
        }

        impl BinOpLeaf<OffsetVar, i64> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: OffsetVar, rhs: i64) {
                ctx.ins_set(&[
                    Air::MovIVar(Reg::A, lhs, self.0),
                    Air::MovIConst(Reg::B, rhs),
                    Air::$instr,
                    Air::PushIReg {
                        dst,
                        kind: self.0,
                        src: Reg::A,
                    },
                ]);
            }
        }

        impl BinOpLeaf<OffsetVar, OffsetVar> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: OffsetVar, rhs: OffsetVar) {
                ctx.ins_set(&[
                    Air::MovIVar(Reg::A, rhs, self.0),
                    Air::MovIVar(Reg::B, lhs, self.0),
                    Air::$instr,
                    Air::PushIReg {
                        dst,
                        kind: self.0,
                        src: Reg::A,
                    },
                ]);
            }
        }

        impl BinOpLeaf<Reg, i64> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: Reg, rhs: i64) {
                if lhs == Reg::A {
                    ctx.ins_set(&[
                        Air::SwapReg,
                        Air::MovIConst(Reg::A, rhs),
                        Air::$instr,
                        Air::PushIReg {
                            dst,
                            kind: self.0,
                            src: Reg::A,
                        },
                    ]);
                } else {
                    ctx.ins_set(&[
                        Air::MovIConst(Reg::A, rhs),
                        Air::$instr,
                        Air::PushIReg {
                            dst,
                            kind: self.0,
                            src: Reg::A,
                        },
                    ]);
                }
            }
        }

        impl BinOpLeaf<i64, Reg> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: i64, rhs: Reg) {
                if rhs == Reg::B {
                    ctx.ins_set(&[
                        Air::SwapReg,
                        Air::MovIConst(Reg::B, lhs),
                        Air::$instr,
                        Air::PushIReg {
                            dst,
                            kind: self.0,
                            src: Reg::A,
                        },
                    ]);
                } else {
                    ctx.ins_set(&[
                        Air::MovIConst(Reg::B, lhs),
                        Air::$instr,
                        Air::PushIReg {
                            dst,
                            kind: self.0,
                            src: Reg::A,
                        },
                    ]);
                }
            }
        }

        impl BinOpLeaf<Reg, OffsetVar> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: Reg, rhs: OffsetVar) {
                if lhs == Reg::A {
                    ctx.ins_set(&[
                        Air::SwapReg,
                        Air::MovIVar(Reg::A, rhs, self.0),
                        Air::$instr,
                        Air::PushIReg {
                            dst,
                            kind: self.0,
                            src: Reg::A,
                        },
                    ]);
                } else {
                    ctx.ins_set(&[
                        Air::MovIVar(Reg::A, rhs, self.0),
                        Air::$instr,
                        Air::PushIReg {
                            dst,
                            kind: self.0,
                            src: Reg::A,
                        },
                    ]);
                }
            }
        }

        impl BinOpLeaf<OffsetVar, Reg> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: OffsetVar, rhs: Reg) {
                if rhs == Reg::B {
                    ctx.ins_set(&[
                        Air::SwapReg,
                        Air::MovIVar(Reg::B, lhs, self.0),
                        Air::$instr,
                        Air::PushIReg {
                            dst,
                            kind: self.0,
                            src: Reg::A,
                        },
                    ]);
                } else {
                    ctx.ins_set(&[
                        Air::MovIVar(Reg::B, lhs, self.0),
                        Air::$instr,
                        Air::PushIReg {
                            dst,
                            kind: self.0,
                            src: Reg::A,
                        },
                    ]);
                }
            }
        }

        impl BinOpLeaf<Reg, Reg> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: Reg, rhs: Reg) {
                assert!(lhs != rhs);
                if rhs == Reg::B {
                    ctx.ins(Air::SwapReg);
                }

                ctx.ins_set(&[
                    Air::$instr,
                    Air::PushIReg {
                        dst,
                        kind: self.0,
                        src: Reg::A,
                    },
                ]);
            }
        }
    }
}
