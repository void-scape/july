use super::ctx::AirCtx;
use super::{eval_expr, OffsetVar};
use crate::air::{generate_args, Air, Bits, ConstData, Reg};
use crate::ir::lit::LitKind;
use crate::ir::ty::store::TyId;
use crate::ir::ty::{FloatTy, IntTy, Sign, Ty, Width};
use crate::ir::*;

/// Evaluate `bin` and assign to `dst`.
///
/// Panics
///     `bin` does not evaluate to `ty`
pub fn assign_bin_op<'a>(ctx: &mut AirCtx<'a>, dst: OffsetVar, ty: TyId, bin: &'a BinOp) {
    let out = ty;
    let ty = ctx.tys.ty(ty);
    match ty {
        Ty::Int(ty) => {
            let width = ty.width();
            match bin.kind {
                BinOpKind::Add => visit(Add(width), ctx, out, dst, bin),
                BinOpKind::Sub => visit(Sub(width), ctx, out, dst, bin),
                BinOpKind::Mul => visit(Mul(width), ctx, out, dst, bin),
                BinOpKind::Div => visit(Div(width), ctx, out, dst, bin),

                BinOpKind::Xor => visit(Xor(width), ctx, out, dst, bin),

                BinOpKind::Eq | BinOpKind::Ne => panic!("invalid op"),
            }
        }
        Ty::Float(ty) => {
            let width = ty.width();
            match bin.kind {
                BinOpKind::Add => visit(FloatAdd(width), ctx, out, dst, bin),
                BinOpKind::Sub => visit(FloatSub(width), ctx, out, dst, bin),
                BinOpKind::Mul => visit(FloatMul(width), ctx, out, dst, bin),
                BinOpKind::Div => visit(FloatDiv(width), ctx, out, dst, bin),
                BinOpKind::Eq => visit(
                    Eq {
                        input: width,
                        output: Width::BOOL,
                    },
                    ctx,
                    out,
                    dst,
                    bin,
                ),
                BinOpKind::Ne => {
                    visit(
                        Ne {
                            input: width,
                            output: Width::BOOL,
                        },
                        ctx,
                        out,
                        dst,
                        bin,
                    );
                }

                BinOpKind::Xor => panic!("invalid op"),
            }
        }
        Ty::Bool => {
            let ty = match (bin.lhs.infer(ctx), bin.rhs.infer(ctx)) {
                (InferTy::Ty(lhs), InferTy::Ty(_rhs)) => {
                    //assert_eq!(lhs, rhs);
                    lhs
                }
                (InferTy::Ty(ty), _) | (_, InferTy::Ty(ty)) => ty,
                (InferTy::Int, other) => {
                    assert_eq!(InferTy::Int, other);
                    ctx.tys.builtin(Ty::Int(IntTy::new_64(Sign::I)))
                }
                (InferTy::Float, other) => {
                    assert_eq!(InferTy::Float, other);
                    ctx.tys.builtin(Ty::Float(FloatTy::F64))
                }
            };

            let lhs = prepare_expr(ctx, ty, bin.lhs);
            let rhs = prepare_expr(ctx, ty, bin.rhs);

            let width = match ctx.tys.ty(ty) {
                Ty::Int(int) => int.width(),
                Ty::Float(float) => float.width(),
                Ty::Bool => Width::BOOL,
                Ty::Ref(&Ty::Str) => unreachable!(),
                Ty::Ref(_) => {
                    // TODO: NULL
                    Width::PTR
                }
                ty => unreachable!("{ty:#?}"),
            };

            match bin.kind {
                BinOpKind::Add
                | BinOpKind::Mul
                | BinOpKind::Sub
                | BinOpKind::Div
                | BinOpKind::Xor => {
                    panic!("invalid op")
                }
                BinOpKind::Eq => {
                    Eq {
                        input: width,
                        output: Width::BOOL,
                    }
                    .visit_with(ctx, dst, lhs, rhs);
                }
                BinOpKind::Ne => {
                    Ne {
                        input: width,
                        output: Width::BOOL,
                    }
                    .visit_with(ctx, dst, lhs, rhs);
                }
            }
        }
        ty => panic!("invalid type: {ty:#?}"),
    }
}

/// Evaluate `bin` for its side effects then throw away.
pub fn eval_bin_op<'a>(ctx: &mut AirCtx<'a>, bin: &'a BinOp) {
    eval_expr(ctx, &bin.lhs);
    eval_expr(ctx, &bin.rhs);
}

pub fn aquire_accessor_field(ctx: &AirCtx, access: &Access) -> (OffsetVar, TyId) {
    let var = match access.lhs {
        Expr::Ident(ident) => ctx.expect_var(ident.id),
        lhs => unimplemented!("accessor: {lhs:?}"),
    };

    let ty = ctx.tys.ty(ctx.expect_var_ty(var));
    let (ty, derefs) = ty.peel_refs();
    assert_eq!(derefs, 0);

    let id = ty.expect_struct();
    let mut strukt = ctx.tys.strukt(id);

    let mut offset = 0;
    for (i, acc) in access.accessors.iter().enumerate() {
        let ty = strukt.field_ty(acc.id);
        if i == access.accessors.len() - 1 {
            let field_offset = strukt.field_offset(ctx, acc.id);
            return (OffsetVar::new(var, (offset + field_offset) as usize), ty);
        }

        match ctx.tys.ty(ty) {
            Ty::Struct(id) => {
                let field_offset = strukt.field_offset(ctx, acc.id);
                offset += field_offset;
                strukt = ctx.tys.strukt(id);
            }
            Ty::Array(_, _)
            | Ty::Ref(_)
            | Ty::Str
            | Ty::Int(_)
            | Ty::Float(_)
            | Ty::Bool
            | Ty::Unit => {
                panic!("cannot access field on {ty:?}")
            }
        }
    }
    unreachable!()
}

macro_rules! impl_op {
    ($name:ident, $strukt:ident) => {
        #[macro_export]
        macro_rules! $name {
            ($ctx:ident, $width:expr, $dst:expr, $lhs:expr, $rhs:expr) => {{
                use crate::air::bin::BinOpLeaf;
                crate::air::bin::$strukt($width).visit_leaf($ctx, $dst, $rhs, $lhs);
            }};
        }
        #[allow(unused)]
        pub use $name;
    };
}

impl_op!(add, Add);
impl_op!(sub, Sub);
impl_op!(mul, Mul);

impl_op!(fadd, FloatAdd);
impl_op!(fsub, FloatSub);
impl_op!(fmul, FloatMul);

#[macro_export]
macro_rules! eq {
    ($ctx:ident, $width:expr, $dst:expr, $lhs:expr, $rhs:expr) => {{
        use crate::air::bin::BinOpLeaf;
        crate::air::bin::Eq {
            output: Width::W8,
            input: $width,
        }
        .visit_leaf($ctx, $dst, $rhs, $lhs);
    }};
}
#[allow(unused)]
pub use eq;

macro_rules! int_op {
    ($ty:ident, $instr:ident) => {
        pub struct $ty(pub Width);
        crate::impl_agnostic_bin_op_visitor!($ty, $instr, 0, 0);
        crate::impl_prim_bin_op_visitor!($ty, $instr);
        crate::impl_int_algebra!($ty);
    };
}

int_op!(Add, AddAB);
int_op!(Sub, SubAB);
int_op!(Mul, MulAB);
int_op!(Div, DivAB);

int_op!(Xor, XorAB);

pub struct Eq {
    pub input: Width,
    pub output: Width,
}
crate::impl_agnostic_bin_op_visitor!(Eq, EqAB, input, output);
crate::impl_prim_bin_op_visitor!(Eq, EqAB);
crate::impl_float_bin_op_visitor!(Eq, FEqAB);
crate::impl_cmp!(Eq);

pub struct Ne {
    pub input: Width,
    pub output: Width,
}
crate::impl_agnostic_bin_op_visitor!(Ne, NEqAB, input, output);
crate::impl_prim_bin_op_visitor!(Ne, NEqAB);
crate::impl_float_bin_op_visitor!(Ne, NFEqAB);
crate::impl_cmp!(Ne);

macro_rules! float_op {
    ($ty:ident, $instr:ident) => {
        pub struct $ty(pub Width);
        crate::impl_agnostic_bin_op_visitor!($ty, $instr, 0, 0);
        crate::impl_float_bin_op_visitor!($ty, $instr);
        crate::impl_float_algebra!($ty);
    };
}

float_op!(FloatAdd, FAddAB);
float_op!(FloatSub, FSubAB);
float_op!(FloatMul, FMulAB);
float_op!(FloatDiv, FDivAB);

pub enum BinOpArg {
    Var(OffsetVar),
    Int(u64),
    Float(f64),
}

fn visit<'a>(
    op: impl BinOpVisitWith,
    ctx: &mut AirCtx<'a>,
    ty: TyId,
    dst: OffsetVar,
    bin: &'a BinOp<'a>,
) {
    let lhs = prepare_expr(ctx, ty, bin.lhs);
    let rhs = prepare_expr(ctx, ty, bin.rhs);
    op.visit_with(ctx, dst, lhs, rhs);
}

fn prepare_expr<'a>(ctx: &mut AirCtx<'a>, ty: TyId, expr: &'a Expr) -> BinOpArg {
    match expr {
        Expr::Lit(lit) => match lit.kind {
            LitKind::Int(lit) => {
                assert!(ctx.tys.ty(ty).is_int());
                BinOpArg::Int(*lit)
            }
            LitKind::Float(float) => {
                assert!(ctx.tys.ty(ty).is_float());
                BinOpArg::Float(*float)
            }
        },
        Expr::Ident(ident) => {
            let var = OffsetVar::zero(ctx.expect_var(ident.id));
            //assert_eq!(ctx.var_ty(ident.id), ty);
            BinOpArg::Var(var)
        }
        Expr::Call(call @ Call { sig, .. }) => {
            assert_eq!(sig.ty, ty);
            let args = generate_args(ctx, call);
            ctx.ins(Air::Call(sig, args));
            let result = OffsetVar::zero(ctx.anon_var(sig.ty));
            let width = match ctx.tys.ty(sig.ty) {
                Ty::Bool => Width::BOOL,
                Ty::Int(ty) => ty.width(),
                Ty::Float(ty) => ty.width(),
                _ => unreachable!(),
            };

            ctx.ins(Air::PushIReg {
                dst: result,
                width,
                src: Reg::A,
            });
            BinOpArg::Var(result)
        }
        Expr::Bool(bool) => {
            assert_eq!(TyId::BOOL, ty);
            BinOpArg::Int(bool.val as u64)
        }
        Expr::Unary(unary) => match unary.kind {
            UOpKind::Ref => match unary.inner {
                Expr::Ident(ident) => {
                    let var = OffsetVar::zero(ctx.expect_var(ident.id));
                    ctx.ins(Air::Addr(Reg::A, var));
                    let var_ty = ctx.var_ty(ident);
                    let result = OffsetVar::zero(ctx.anon_var(var_ty));
                    ctx.ins(Air::PushIReg {
                        dst: result,
                        width: Width::PTR,
                        src: Reg::A,
                    });

                    let inner = ctx.tys.ty(var_ty);
                    assert_eq!(ctx.tys.get_ty_id(&Ty::Ref(&inner)).unwrap(), ty);
                    BinOpArg::Var(result)
                }
                _ => todo!(),
            },
            _ => todo!(),
        },
        Expr::Access(access) => {
            let (var, accessor_ty) = aquire_accessor_field(ctx, access);
            assert_eq!(accessor_ty, ty);
            BinOpArg::Var(var)
        }
        Expr::Bin(bin) => {
            let var = OffsetVar::zero(ctx.anon_var(ty));
            assign_bin_op(ctx, var, ty, bin);
            BinOpArg::Var(var)
        }
        Expr::If(_) | Expr::IndexOf(_) | Expr::Block(_) | Expr::Array(_) => todo!(),
        Expr::Struct(_)
        | Expr::Enum(_)
        | Expr::Str(_)
        | Expr::Loop(_)
        | Expr::Range(_)
        | Expr::For(_)
        | Expr::Break(_)
        | Expr::Continue(_) => unreachable!(),
    }
}

pub trait BinOpVisitWith {
    fn visit_with<'a>(&self, ctx: &mut AirCtx<'a>, dst: OffsetVar, lhs: BinOpArg, rhs: BinOpArg);
}

pub trait FloatAlgebra: AgnosticBinOpLeaves + AllFloatBinOpLeaves {}

#[macro_export]
macro_rules! impl_float_algebra {
    ($ty:ident) => {
        impl FloatAlgebra for $ty {}
        impl BinOpVisitWith for $ty {
            fn visit_with<'a>(
                &self,
                ctx: &mut AirCtx<'a>,
                dst: OffsetVar,
                lhs: BinOpArg,
                rhs: BinOpArg,
            ) {
                visit_float(self, ctx, dst, rhs, lhs);
            }
        }
    };
}

fn visit_float<'a>(
    op: &impl FloatAlgebra,
    ctx: &mut AirCtx<'a>,
    dst: OffsetVar,
    lhs: BinOpArg,
    rhs: BinOpArg,
) {
    match (rhs, lhs) {
        (BinOpArg::Var(var), BinOpArg::Float(lit)) => {
            op.visit_leaf(ctx, dst, lit, var);
        }
        (BinOpArg::Float(lit), BinOpArg::Var(var)) => {
            op.visit_leaf(ctx, dst, var, lit);
        }
        (BinOpArg::Float(rhs), BinOpArg::Float(lhs)) => {
            op.visit_leaf(ctx, dst, lhs, rhs);
        }
        (BinOpArg::Var(rhs), BinOpArg::Var(lhs)) => {
            op.visit_leaf(ctx, dst, lhs, rhs);
        }

        _ => panic!("invalid op"),
    }
}

pub trait IntAlgebra: AgnosticBinOpLeaves + AllPrimBinOpLeaves {}

#[macro_export]
macro_rules! impl_int_algebra {
    ($ty:ident) => {
        impl IntAlgebra for $ty {}
        impl BinOpVisitWith for $ty {
            fn visit_with<'a>(
                &self,
                ctx: &mut AirCtx<'a>,
                dst: OffsetVar,
                lhs: BinOpArg,
                rhs: BinOpArg,
            ) {
                visit_int(self, ctx, dst, rhs, lhs);
            }
        }
    };
}

#[macro_export]
macro_rules! impl_cmp {
    ($ty:ident) => {
        impl IntAlgebra for $ty {}
        impl FloatAlgebra for $ty {}

        impl BinOpVisitWith for $ty {
            fn visit_with<'a>(
                &self,
                ctx: &mut AirCtx<'a>,
                dst: OffsetVar,
                lhs: BinOpArg,
                rhs: BinOpArg,
            ) {
                match (&lhs, &rhs) {
                    (BinOpArg::Float(_), _) | (_, BinOpArg::Float(_)) => {
                        visit_float(self, ctx, dst, rhs, lhs);
                    }
                    _ => {
                        visit_int(self, ctx, dst, rhs, lhs);
                    }
                }
            }
        }
    };
}

fn visit_int<'a>(
    op: &impl IntAlgebra,
    ctx: &mut AirCtx<'a>,
    dst: OffsetVar,
    lhs: BinOpArg,
    rhs: BinOpArg,
) {
    match (rhs, lhs) {
        (BinOpArg::Var(var), BinOpArg::Int(lit)) => {
            op.visit_leaf(ctx, dst, var, lit);
        }
        (BinOpArg::Int(lit), BinOpArg::Var(var)) => {
            op.visit_leaf(ctx, dst, var, lit);
        }
        (BinOpArg::Int(rhs), BinOpArg::Int(lhs)) => {
            op.visit_leaf(ctx, dst, lhs, rhs);
        }
        (BinOpArg::Var(rhs), BinOpArg::Var(lhs)) => {
            op.visit_leaf(ctx, dst, lhs, rhs);
        }

        _ => panic!("invalid op"),
    }
}

pub trait BinOpLeaf<L, R> {
    fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: L, rhs: R);
}

pub trait BinOpIOKind {
    fn input(&self) -> Width;
    fn output(&self) -> Width;
}

pub trait AgnosticBinOpLeaves:
    BinOpLeaf<OffsetVar, OffsetVar>
    + BinOpLeaf<Reg, OffsetVar>
    + BinOpLeaf<OffsetVar, Reg>
    + BinOpLeaf<Reg, Reg>
{
}

pub trait AllPrimBinOpLeaves:
    BinOpLeaf<u64, u64>
    + BinOpLeaf<u64, OffsetVar>
    + BinOpLeaf<OffsetVar, u64>
    + BinOpLeaf<Reg, u64>
    + BinOpLeaf<u64, Reg>
{
}

pub trait AllFloatBinOpLeaves:
    BinOpLeaf<f64, f64>
    + BinOpLeaf<f64, OffsetVar>
    + BinOpLeaf<OffsetVar, f64>
    + BinOpLeaf<Reg, f64>
    + BinOpLeaf<f64, Reg>
{
}

#[macro_export]
macro_rules! impl_agnostic_bin_op_visitor {
    ($name:ident, $instr:ident, $input:tt, $output:tt) => {
        impl AgnosticBinOpLeaves for $name {}

        impl BinOpIOKind for $name {
            fn input(&self) -> Width {
                self.$input
            }

            fn output(&self) -> Width {
                self.$output
            }
        }

        impl BinOpLeaf<OffsetVar, OffsetVar> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: OffsetVar, rhs: OffsetVar) {
                ctx.ins_set([
                    Air::MovIVar(Reg::A, rhs, self.input()),
                    Air::MovIVar(Reg::B, lhs, self.input()),
                    Air::$instr(self.input()),
                    Air::PushIReg {
                        dst,
                        width: self.output(),
                        src: Reg::A,
                    },
                ]);
            }
        }

        impl BinOpLeaf<Reg, OffsetVar> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: Reg, rhs: OffsetVar) {
                if lhs == Reg::A {
                    ctx.ins(Air::SwapReg);
                }

                ctx.ins_set([
                    Air::MovIVar(Reg::A, rhs, self.input()),
                    Air::$instr(self.input()),
                    Air::PushIReg {
                        dst,
                        width: self.output(),
                        src: Reg::A,
                    },
                ]);
            }
        }

        impl BinOpLeaf<OffsetVar, Reg> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: OffsetVar, rhs: Reg) {
                if rhs == Reg::B {
                    ctx.ins(Air::SwapReg);
                }

                ctx.ins_set([
                    Air::MovIVar(Reg::B, lhs, self.input()),
                    Air::$instr(self.input()),
                    Air::PushIReg {
                        dst,
                        width: self.output(),
                        src: Reg::A,
                    },
                ]);
            }
        }

        impl BinOpLeaf<Reg, Reg> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: Reg, rhs: Reg) {
                assert!(lhs != rhs);
                if rhs == Reg::B {
                    ctx.ins(Air::SwapReg);
                }

                ctx.ins_set([
                    Air::$instr(self.input()),
                    Air::PushIReg {
                        dst,
                        width: self.output(),
                        src: Reg::A,
                    },
                ]);
            }
        }
    };
}

#[macro_export]
macro_rules! impl_prim_bin_op_visitor {
    ($name:ident, $instr:ident) => {
        impl AllPrimBinOpLeaves for $name {}

        impl BinOpLeaf<u64, u64> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: u64, rhs: u64) {
                ctx.ins_set([
                    Air::MovIConst(
                        Reg::A,
                        ConstData::Bits(Bits::from_width(rhs as u64, self.input())),
                    ),
                    Air::MovIConst(
                        Reg::B,
                        ConstData::Bits(Bits::from_width(lhs as u64, self.input())),
                    ),
                    Air::$instr(self.input()),
                    Air::PushIReg {
                        dst,
                        width: self.output(),
                        src: Reg::A,
                    },
                ]);
            }
        }

        impl BinOpLeaf<u64, OffsetVar> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: u64, rhs: OffsetVar) {
                ctx.ins_set([
                    Air::MovIVar(Reg::A, rhs, self.input()),
                    Air::MovIConst(
                        Reg::B,
                        ConstData::Bits(Bits::from_width(lhs as u64, self.input())),
                    ),
                    Air::$instr(self.input()),
                    Air::PushIReg {
                        dst,
                        width: self.output(),
                        src: Reg::A,
                    },
                ]);
            }
        }

        impl BinOpLeaf<OffsetVar, u64> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: OffsetVar, rhs: u64) {
                ctx.ins_set([
                    Air::MovIVar(Reg::A, lhs, self.input()),
                    Air::MovIConst(
                        Reg::B,
                        ConstData::Bits(Bits::from_width(rhs as u64, self.input())),
                    ),
                    Air::$instr(self.input()),
                    Air::PushIReg {
                        dst,
                        width: self.output(),
                        src: Reg::A,
                    },
                ]);
            }
        }

        impl BinOpLeaf<Reg, u64> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: Reg, rhs: u64) {
                if lhs == Reg::A {
                    ctx.ins(Air::SwapReg);
                }

                ctx.ins_set([
                    Air::MovIConst(
                        Reg::A,
                        ConstData::Bits(Bits::from_width(rhs as u64, self.input())),
                    ),
                    Air::$instr(self.input()),
                    Air::PushIReg {
                        dst,
                        width: self.output(),
                        src: Reg::A,
                    },
                ]);
            }
        }

        impl BinOpLeaf<u64, Reg> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: u64, rhs: Reg) {
                if rhs == Reg::B {
                    ctx.ins(Air::SwapReg);
                }

                ctx.ins_set([
                    Air::MovIConst(
                        Reg::B,
                        ConstData::Bits(Bits::from_width(lhs as u64, self.input())),
                    ),
                    Air::$instr(self.input()),
                    Air::PushIReg {
                        dst,
                        width: self.output(),
                        src: Reg::A,
                    },
                ]);
            }
        }
    };
}

#[macro_export]
macro_rules! impl_float_bin_op_visitor {
    ($name:ident, $instr:ident) => {
        impl AllFloatBinOpLeaves for $name {}

        impl BinOpLeaf<f64, f64> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: f64, rhs: f64) {
                ctx.ins_set([
                    Air::MovIConst(
                        Reg::A,
                        ConstData::Bits(Bits::from_width_float(rhs, self.input())),
                    ),
                    Air::MovIConst(
                        Reg::B,
                        ConstData::Bits(Bits::from_width_float(lhs, self.input())),
                    ),
                    Air::$instr(self.input()),
                    Air::PushIReg {
                        dst,
                        width: self.output(),
                        src: Reg::A,
                    },
                ]);
            }
        }

        impl BinOpLeaf<f64, OffsetVar> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: f64, rhs: OffsetVar) {
                ctx.ins(Air::MovIConst(
                    Reg::B,
                    ConstData::Bits(Bits::from_width_float(lhs, self.input())),
                ));

                ctx.ins_set([
                    Air::MovIVar(Reg::A, rhs, self.input()),
                    Air::$instr(self.input()),
                    Air::PushIReg {
                        dst,
                        width: self.output(),
                        src: Reg::A,
                    },
                ]);
            }
        }

        impl BinOpLeaf<OffsetVar, f64> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: OffsetVar, rhs: f64) {
                ctx.ins(Air::MovIConst(
                    Reg::A,
                    ConstData::Bits(Bits::from_width_float(rhs, self.input())),
                ));

                ctx.ins_set([
                    Air::MovIVar(Reg::B, lhs, self.input()),
                    Air::$instr(self.input()),
                    Air::PushIReg {
                        dst,
                        width: self.output(),
                        src: Reg::A,
                    },
                ]);
            }
        }

        impl BinOpLeaf<Reg, f64> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: Reg, rhs: f64) {
                if lhs == Reg::A {
                    ctx.ins(Air::SwapReg);
                }

                ctx.ins(Air::MovIConst(
                    Reg::A,
                    ConstData::Bits(Bits::from_width_float(rhs, self.input())),
                ));

                ctx.ins_set([
                    Air::$instr(self.input()),
                    Air::PushIReg {
                        dst,
                        width: self.output(),
                        src: Reg::A,
                    },
                ]);
            }
        }

        impl BinOpLeaf<f64, Reg> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: f64, rhs: Reg) {
                if rhs == Reg::B {
                    ctx.ins(Air::SwapReg);
                }

                ctx.ins(Air::MovIConst(
                    Reg::B,
                    ConstData::Bits(Bits::from_width_float(lhs, self.input())),
                ));

                ctx.ins_set([
                    Air::$instr(self.input()),
                    Air::PushIReg {
                        dst,
                        width: self.output(),
                        src: Reg::A,
                    },
                ]);
            }
        }
    };
}
