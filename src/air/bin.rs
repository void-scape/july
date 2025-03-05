use super::ctx::AirCtx;
use super::{eval_expr, IntKind, OffsetVar};
use crate::air::{generate_args, Air, ConstData, Reg};
use crate::ir::lit::LitKind;
use crate::ir::sig::Sig;
use crate::ir::ty::store::TyId;
use crate::ir::ty::Ty;
use crate::ir::*;

/// Evaluate `bin` and assign to `dst`.
///
/// Panics
///     `ty` is not integral
///     `bin` does not evaluate to `ty`
pub fn assign_bin_op<'a>(ctx: &mut AirCtx<'a>, dst: OffsetVar, ty: TyId, bin: &'a BinOp) {
    let ty = ctx.tys.ty(ty);
    let kind = ty.expect_int().kind();

    match bin.kind {
        BinOpKind::Add => Add(kind).visit(ctx, dst, bin),
        BinOpKind::Mul => Mul(kind).visit(ctx, dst, bin),
        BinOpKind::Sub => Sub(kind).visit(ctx, dst, bin),
        BinOpKind::Eq => {
            let mut eq = Eq {
                input: IntKind::BOOL,
                output: IntKind::BOOL,
            };

            let (lhs, lhs_ty) = eq.prepare_expr(ctx, bin.lhs);
            let (rhs, rhs_ty) = eq.prepare_expr(ctx, bin.rhs);
            match (lhs_ty, rhs_ty) {
                (Some(lhs), Some(rhs)) => {
                    assert_eq!(lhs, rhs);
                    eq.input = ctx.tys.ty(lhs).expect_int().kind();
                }
                (Some(ty), None) | (None, Some(ty)) => {
                    eq.input = ctx.tys.ty(ty).expect_int().kind();
                }
                (None, None) => {
                    // both are int literals
                    eq.input = IntKind::I64;
                }
            }
            eq.visit_with(ctx, dst, lhs, rhs);
        }
    }
}

/// Evaluate `bin` for its side effects then throw away.
pub fn eval_bin_op<'a>(ctx: &mut AirCtx<'a>, sig: &'a Sig<'a>, bin: &'a BinOp) {
    eval_expr(ctx, sig, &bin.lhs);
    eval_expr(ctx, sig, &bin.rhs);
}

pub fn aquire_accessor_field(ctx: &AirCtx, access: &Access) -> (OffsetVar, TyId) {
    let var = match access.lhs {
        Expr::Ident(ident) => ctx.expect_var(ident.id),
        lhs => unimplemented!("accessor: {lhs:?}"),
    };

    let mut deref = false;
    let ty = ctx.tys.ty(ctx.expect_var_ty(var));
    let (ty, derefs) = ty.peel_refs();
    if derefs == 1 {
        deref = true;
    } else if derefs > 1 {
        unimplemented!()
    }

    let id = ty.expect_struct();
    let mut strukt = ctx.tys.strukt(id);

    let mut offset = 0;
    for (i, acc) in access.accessors.iter().enumerate() {
        let ty = strukt.field_ty(acc.id);
        if i == access.accessors.len() - 1 {
            let field_offset = strukt.field_offset(ctx, acc.id);
            return (
                OffsetVar::new_deref(var, (offset + field_offset) as usize, deref),
                ty,
            );
        }

        match ctx.tys.ty(ty) {
            Ty::Struct(id) => {
                let field_offset = strukt.field_offset(ctx, acc.id);
                offset += field_offset;
                strukt = ctx.tys.strukt(id);
            }
            Ty::Ref(_) => todo!(),
            Ty::Str => todo!(),
            Ty::Int(t) => panic!("{t:?} has not fields"),
            Ty::Bool => panic!("bool has no fields"),
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

pub struct Add(pub IntKind);
crate::impl_prim_bin_op_visitor!(Add, AddAB, +, 0, 0);

pub struct Sub(pub IntKind);
crate::impl_prim_bin_op_visitor!(Sub, SubAB, -, 0, 0);

pub struct Mul(pub IntKind);
crate::impl_prim_bin_op_visitor!(Mul, MulAB, *, 0, 0);

pub struct Eq {
    pub input: IntKind,
    pub output: IntKind,
}
crate::impl_prim_bin_op_visitor!(Eq, EqAB, ==, input, output);

enum BinOpArg {
    Var(OffsetVar),
    Lit(i64),
}

trait BinOpVisitor
where
    Self: AllBinOpLeaves,
{
    fn visit<'a>(&mut self, ctx: &mut AirCtx<'a>, dst: OffsetVar, bin: &'a BinOp<'a>) {
        let (lhs, lhs_ty) = self.prepare_expr(ctx, bin.lhs);
        let (rhs, rhs_ty) = self.prepare_expr(ctx, bin.rhs);
        match (lhs_ty, rhs_ty) {
            (Some(lhs), Some(rhs)) => assert_eq!(lhs, rhs),
            _ => {}
        }
        self.visit_with(ctx, dst, lhs, rhs);
    }

    fn visit_with<'a>(
        &mut self,
        ctx: &mut AirCtx<'a>,
        dst: OffsetVar,
        lhs: BinOpArg,
        rhs: BinOpArg,
    ) {
        match (lhs, rhs) {
            (BinOpArg::Var(var), BinOpArg::Lit(lit)) => {
                self.visit_leaf(ctx, dst, var, lit);
            }
            (BinOpArg::Lit(lit), BinOpArg::Var(var)) => {
                self.visit_leaf(ctx, dst, var, lit);
            }
            (BinOpArg::Lit(rhs), BinOpArg::Lit(lhs)) => {
                self.visit_leaf(ctx, dst, lhs, rhs);
            }
            (BinOpArg::Var(rhs), BinOpArg::Var(lhs)) => {
                self.visit_leaf(ctx, dst, lhs, rhs);
            }
        }
    }

    fn prepare_expr<'a>(&self, ctx: &mut AirCtx<'a>, expr: &'a Expr) -> (BinOpArg, Option<TyId>) {
        match expr {
            Expr::Lit(lit) => match lit.kind {
                LitKind::Int(lhs_lit) => (BinOpArg::Lit(*lhs_lit), None),
                _ => unreachable!(),
            },
            Expr::Ident(ident) => {
                let var = OffsetVar::zero(ctx.expect_var(ident.id));
                (BinOpArg::Var(var), Some(ctx.var_ty(ident.id)))
            }
            Expr::Call(call @ Call { sig, .. }) => {
                let args = generate_args(ctx, sig, call);
                ctx.ins(Air::Call(sig, args));
                let result = OffsetVar::zero(ctx.anon_var(sig.ty));
                ctx.ins(Air::PushIReg {
                    dst: result,
                    kind: ctx.tys.ty(sig.ty).expect_int().kind(),
                    src: Reg::A,
                });
                (BinOpArg::Var(result), Some(sig.ty))
            }
            Expr::Bool(_, val) => (BinOpArg::Lit(*val as i64), Some(TyId::BOOL)),
            Expr::Ref(ref_) => match ref_.inner {
                Expr::Ident(ident) => {
                    let var = OffsetVar::zero(ctx.expect_var(ident.id));
                    ctx.ins(Air::Addr(Reg::A, var));
                    let var_ty = ctx.var_ty(ident.id);
                    let result = OffsetVar::zero(ctx.anon_var(var_ty));
                    ctx.ins(Air::PushIReg {
                        dst: result,
                        kind: IntKind::PTR,
                        src: Reg::A,
                    });

                    let inner = ctx.tys.ty(var_ty);
                    (
                        BinOpArg::Var(result),
                        Some(ctx.tys.get_ty_id(&Ty::Ref(&inner)).unwrap()),
                    )
                }
                _ => todo!(),
            },
            Expr::Bin(_bin) => {
                todo!()
                //let dst = OffsetVar::zero(ctx.anon_var(ty));
                //self.visit(ctx, kind, dst, bin);
                //(BinOpArg::Var(dst),)
            }
            Expr::Access(access) => {
                let (var, accessor_ty) = aquire_accessor_field(ctx, access);
                (BinOpArg::Var(var), Some(accessor_ty))
            }
            Expr::Struct(_) | Expr::Enum(_) | Expr::If(_) => unimplemented!(),
            Expr::Str(_, _) | Expr::Block(_) => todo!(),
            Expr::Loop(_) => todo!(),
        }
    }
}

pub trait BinOpLeaf<L, R> {
    fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: L, rhs: R);
}

pub trait BinOpIOKind {
    fn input(&self) -> IntKind;
    fn output(&self) -> IntKind;
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
    ($name:ident, $instr:ident, $op:tt, $input:tt, $output:tt) => {
        impl BinOpVisitor for $name {}

        impl AllBinOpLeaves for $name {}

        impl BinOpIOKind for $name {
            fn input(&self) -> IntKind {
                self.$input
            }

            fn output(&self) -> IntKind {
                self.$output
            }
        }

        impl BinOpLeaf<i64, i64> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: i64, rhs: i64) {
                ctx.ins(Air::PushIConst(dst, self.output(), ConstData::Int((lhs $op rhs) as i64)));
            }
        }

        impl BinOpLeaf<i64, OffsetVar> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: i64, rhs: OffsetVar) {
                ctx.ins_set([
                    Air::MovIVar(Reg::A, rhs, self.input()),
                    Air::MovIConst(Reg::B, lhs),
                    Air::$instr,
                    Air::PushIReg {
                        dst,
                        kind: self.output(),
                        src: Reg::A,
                    },
                ]);
            }
        }

        impl BinOpLeaf<OffsetVar, i64> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: OffsetVar, rhs: i64) {
                ctx.ins_set([
                    Air::MovIVar(Reg::A, lhs, self.input()),
                    Air::MovIConst(Reg::B, rhs),
                    Air::$instr,
                    Air::PushIReg {
                        dst,
                        kind: self.output(),
                        src: Reg::A,
                    },
                ]);
            }
        }

        impl BinOpLeaf<OffsetVar, OffsetVar> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: OffsetVar, rhs: OffsetVar) {
                ctx.ins_set([
                    Air::MovIVar(Reg::A, rhs, self.input()),
                    Air::MovIVar(Reg::B, lhs, self.input()),
                    Air::$instr,
                    Air::PushIReg {
                        dst,
                        kind: self.output(),
                        src: Reg::A,
                    },
                ]);
            }
        }

        impl BinOpLeaf<Reg, i64> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: Reg, rhs: i64) {
                if lhs == Reg::A {
                    ctx.ins_set([
                        Air::SwapReg,
                        Air::MovIConst(Reg::A, rhs),
                        Air::$instr,
                        Air::PushIReg {
                            dst,
                            kind: self.output(),
                            src: Reg::A,
                        },
                    ]);
                } else {
                    ctx.ins_set([
                        Air::MovIConst(Reg::A, rhs),
                        Air::$instr,
                        Air::PushIReg {
                            dst,
                            kind: self.output(),
                            src: Reg::A,
                        },
                    ]);
                }
            }
        }

        impl BinOpLeaf<i64, Reg> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: i64, rhs: Reg) {
                if rhs == Reg::B {
                    ctx.ins_set([
                        Air::SwapReg,
                        Air::MovIConst(Reg::B, lhs),
                        Air::$instr,
                        Air::PushIReg {
                            dst,
                            kind: self.output(),
                            src: Reg::A,
                        },
                    ]);
                } else {
                    ctx.ins_set([
                        Air::MovIConst(Reg::B, lhs),
                        Air::$instr,
                        Air::PushIReg {
                            dst,
                            kind: self.output(),
                            src: Reg::A,
                        },
                    ]);
                }
            }
        }

        impl BinOpLeaf<Reg, OffsetVar> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: Reg, rhs: OffsetVar) {
                if lhs == Reg::A {
                    ctx.ins(Air::SwapReg);
                }

                ctx.ins_set([
                    Air::MovIVar(Reg::A, rhs, self.input()),
                    Air::$instr,
                    Air::PushIReg {
                        dst,
                        kind: self.output(),
                        src: Reg::A,
                    },
                ]);
            }
        }

        impl BinOpLeaf<OffsetVar, Reg> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: OffsetVar, rhs: Reg) {
                if rhs == Reg::B {
                    ctx.ins_set([
                        Air::SwapReg,
                        Air::MovIVar(Reg::B, lhs, self.input()),
                        Air::$instr,
                        Air::PushIReg {
                            dst,
                            kind: self.output(),
                            src: Reg::A,
                        },
                    ]);
                } else {
                    ctx.ins_set([
                        Air::MovIVar(Reg::B, lhs, self.input()),
                        Air::$instr,
                        Air::PushIReg {
                            dst,
                            kind: self.output(),
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

                ctx.ins_set([
                    Air::$instr,
                    Air::PushIReg {
                        dst,
                        kind: self.output(),
                        src: Reg::A,
                    },
                ]);
            }
        }
    }
}
