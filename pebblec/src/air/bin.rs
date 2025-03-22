use super::ctx::AirCtx;
use super::{OffsetVar, eval_expr};
use crate::air::{Air, Bits, ConstData, Reg, extract_var_from_expr};
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
            let sign = ty.sign();
            match bin.kind {
                BinOpKind::Add => visit(Add::new(width, sign), ctx, out, dst, bin),
                BinOpKind::Sub => visit(Sub::new(width, sign), ctx, out, dst, bin),
                BinOpKind::Mul => visit(Mul::new(width, sign), ctx, out, dst, bin),
                BinOpKind::Div => visit(Div::new(width, sign), ctx, out, dst, bin),

                BinOpKind::Shl => visit(Shl::new(width), ctx, out, dst, bin),
                BinOpKind::Shr => visit(Shr::new(width), ctx, out, dst, bin),
                BinOpKind::Band => visit(Band::new(width), ctx, out, dst, bin),
                BinOpKind::Xor => visit(Xor::new(width), ctx, out, dst, bin),
                BinOpKind::Bor => visit(Bor::new(width), ctx, out, dst, bin),

                BinOpKind::Eq
                | BinOpKind::Ne
                | BinOpKind::Gt
                | BinOpKind::Lt
                | BinOpKind::Ge
                | BinOpKind::Le
                | BinOpKind::And
                | BinOpKind::Or => {
                    panic!("invalid op")
                }
            }
        }
        Ty::Float(ty) => {
            let width = ty.width();
            let sign = Sign::I;
            match bin.kind {
                BinOpKind::Add => visit(FloatAdd::new(width), ctx, out, dst, bin),
                BinOpKind::Sub => visit(FloatSub::new(width), ctx, out, dst, bin),
                BinOpKind::Mul => visit(FloatMul::new(width), ctx, out, dst, bin),
                BinOpKind::Div => visit(FloatDiv::new(width), ctx, out, dst, bin),

                BinOpKind::Eq => visit(Eq::new(width, sign), ctx, out, dst, bin),
                BinOpKind::Ne => {
                    visit(Ne::new(width, sign), ctx, out, dst, bin);
                }
                BinOpKind::Gt => visit(Gt::new(width, sign), ctx, out, dst, bin),
                BinOpKind::Lt => {
                    visit(Lt::new(width, sign), ctx, out, dst, bin);
                }
                BinOpKind::Ge => visit(Ge::new(width, sign), ctx, out, dst, bin),
                BinOpKind::Le => {
                    visit(Le::new(width, sign), ctx, out, dst, bin);
                }

                BinOpKind::And
                | BinOpKind::Or
                | BinOpKind::Shl
                | BinOpKind::Shr
                | BinOpKind::Band
                | BinOpKind::Xor
                | BinOpKind::Bor => panic!("invalid op"),
            }
        }
        Ty::Bool => {
            let ty = match (bin.lhs.infer(ctx), bin.rhs.infer(ctx)) {
                (InferTy::Ty(lhs), InferTy::Ty(rhs)) => {
                    assert_eq!(lhs, rhs);
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

            let (width, sign) = match ctx.tys.ty(ty) {
                Ty::Int(int) => (int.width(), int.sign()),
                Ty::Float(float) => (float.width(), Sign::I),
                Ty::Bool => (Width::BOOL, Sign::U),
                Ty::Ref(&Ty::Str) => unreachable!(),
                Ty::Ref(_) => {
                    // TODO: NULL
                    (Width::PTR, Sign::U)
                }
                ty => unreachable!("{ty:#?}"),
            };

            match bin.kind {
                BinOpKind::Add
                | BinOpKind::Mul
                | BinOpKind::Sub
                | BinOpKind::Div
                | BinOpKind::Shl
                | BinOpKind::Shr
                | BinOpKind::Band
                | BinOpKind::Bor
                | BinOpKind::Xor => {
                    panic!("invalid op")
                }
                BinOpKind::Eq => {
                    Eq::new(width, sign).visit_with(ctx, dst, lhs, rhs);
                }
                BinOpKind::Ne => {
                    Ne::new(width, sign).visit_with(ctx, dst, lhs, rhs);
                }
                BinOpKind::Gt => {
                    Gt::new(width, sign).visit_with(ctx, dst, lhs, rhs);
                }
                BinOpKind::Lt => {
                    Lt::new(width, sign).visit_with(ctx, dst, lhs, rhs);
                }
                BinOpKind::Ge => {
                    Ge::new(width, sign).visit_with(ctx, dst, lhs, rhs);
                }
                BinOpKind::Le => {
                    Le::new(width, sign).visit_with(ctx, dst, lhs, rhs);
                }
                BinOpKind::And | BinOpKind::Or => {
                    panic!("invalid operations")
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

pub fn aquire_accessor_field<'a>(ctx: &mut AirCtx<'a>, access: &'a Access) -> (OffsetVar, TyId) {
    let ty = access.lhs.infer_abs(ctx).unwrap();
    let var = extract_var_from_expr(ctx, ty, access.lhs);

    let id = ctx.tys.ty(ty).expect_struct();
    let mut strukt = ctx.tys.strukt(id);

    let mut offset = 0;
    for (i, acc) in access.accessors.iter().rev().enumerate() {
        let ty = strukt.field_ty(acc.id);
        if i == access.accessors.len() - 1 {
            let field_offset = strukt.field_offset(ctx, acc.id);
            return (var.add((offset + field_offset) as usize), ty);
        }

        match ctx.tys.ty(ty) {
            Ty::Struct(id) => {
                let field_offset = strukt.field_offset(ctx, acc.id);
                offset += field_offset;
                strukt = ctx.tys.strukt(id);
            }
            Ty::Array(_, _)
            | Ty::Slice(_)
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
    (Int, $name:ident, $strukt:ident) => {
        #[macro_export]
        macro_rules! $name {
            ($ctx:ident, $width:expr, $sign:expr, $dst:expr, $lhs:expr, $rhs:expr) => {{
                use crate::air::bin::BinOpLeaf;
                crate::air::bin::$strukt {
                    input: $width,
                    output: $width,
                    sign: $sign,
                }
                .visit_leaf($ctx, $dst, $rhs, $lhs);
            }};
        }
        #[allow(unused)]
        pub use $name;
    };

    (Float, $name:ident, $strukt:ident) => {
        #[macro_export]
        macro_rules! $name {
            ($ctx:ident, $width:expr, $dst:expr, $lhs:expr, $rhs:expr) => {{
                use crate::air::bin::BinOpLeaf;
                crate::air::bin::$strukt {
                    input: $width,
                    output: $width,
                }
                .visit_leaf($ctx, $dst, $rhs, $lhs);
            }};
        }
        #[allow(unused)]
        pub use $name;
    };
}

impl_op!(Int, add, Add);
impl_op!(Int, sub, Sub);
impl_op!(Int, mul, Mul);

impl_op!(Float, fadd, FloatAdd);
impl_op!(Float, fsub, FloatSub);
impl_op!(Float, fmul, FloatMul);

#[macro_export]
macro_rules! eq {
    ($ctx:ident, $width:expr, $sign:expr, $dst:expr, $lhs:expr, $rhs:expr) => {{
        use crate::air::bin::BinOpLeaf;
        crate::air::bin::Eq {
            output: Width::W8,
            input: $width,
            sign: $sign,
        }
        .visit_leaf($ctx, $dst, $rhs, $lhs);
    }};
}
#[allow(unused)]
pub use eq;
use pebblec_parse::BinOpKind;

macro_rules! int_op {
    ($ty:ident, $instr:ident) => {
        pub struct $ty {
            pub input: Width,
            pub output: Width,
            pub sign: Sign,
        }
        impl $ty {
            pub fn new(width: Width, sign: Sign) -> Self {
                Self {
                    input: width,
                    output: width,
                    sign,
                }
            }
        }
        crate::impl_agnostic_bin_op_visitor!($ty, $instr, input, sign);
        crate::impl_prim_bin_op_visitor!($ty, $instr, input, sign);
        crate::impl_int_algebra!($ty);
    };
}

macro_rules! bit_op {
    ($ty:ident, $instr:ident) => {
        pub struct $ty {
            pub input: Width,
            pub output: Width,
        }
        impl $ty {
            pub fn new(width: Width) -> Self {
                Self {
                    input: width,
                    output: width,
                }
            }
        }
        crate::impl_agnostic_bin_op_visitor!($ty, $instr, input);
        crate::impl_prim_bin_op_visitor!($ty, $instr, input);
        crate::impl_int_algebra!($ty);
    };
}

macro_rules! cmp_op {
    ($ty:ident, $instr:ident, $finstr:ident) => {
        pub struct $ty {
            pub input: Width,
            pub output: Width,
            pub sign: Sign,
        }
        impl $ty {
            pub fn new(input: Width, sign: Sign) -> Self {
                Self {
                    input,
                    output: Width::BOOL,
                    sign,
                }
            }
        }
        crate::impl_agnostic_bin_op_visitor!($ty, $instr, input, sign);
        crate::impl_prim_bin_op_visitor!($ty, $instr, input, sign);
        crate::impl_float_bin_op_visitor!($ty, $finstr);
        crate::impl_cmp!($ty);
    };
}

macro_rules! float_op {
    ($ty:ident, $instr:ident) => {
        pub struct $ty {
            pub input: Width,
            pub output: Width,
        }
        impl $ty {
            pub fn new(width: Width) -> Self {
                Self {
                    input: width,
                    output: width,
                }
            }
        }
        crate::impl_agnostic_bin_op_visitor!($ty, $instr, input);
        crate::impl_float_bin_op_visitor!($ty, $instr);
        crate::impl_float_algebra!($ty);
    };
}

int_op!(Add, AddAB);
int_op!(Sub, SubAB);
int_op!(Mul, MulAB);
int_op!(Div, DivAB);

bit_op!(Shl, ShlAB);
bit_op!(Shr, ShrAB);
bit_op!(Band, BandAB);
bit_op!(Xor, XorAB);
bit_op!(Bor, BorAB);

cmp_op!(Eq, EqAB, FEqAB);
cmp_op!(Ne, NEqAB, NFEqAB);
cmp_op!(Lt, LtAB, FLtAB);
cmp_op!(Gt, GtAB, FGtAB);
cmp_op!(Le, LeAB, FLeAB);
cmp_op!(Ge, GeAB, FGeAB);

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
        _ => BinOpArg::Var(extract_var_from_expr(ctx, ty, expr)),
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
macro_rules! instr {
    ($instr:ident, $self:ident,) => {
        Air::$instr
    };

    ($instr:ident, $self:ident, $($args:ident),+) => {
        Air::$instr($($self.$args),*)
    };
}

#[macro_export]
macro_rules! impl_agnostic_bin_op_visitor {
    ($name:ident, $instr:ident, $($args:ident),*) => {
        impl AgnosticBinOpLeaves for $name {}

        impl BinOpLeaf<OffsetVar, OffsetVar> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: OffsetVar, rhs: OffsetVar) {
                ctx.ins_set([
                    Air::MovIVar(Reg::A, rhs, self.input),
                    Air::MovIVar(Reg::B, lhs, self.input),
                    crate::instr!($instr, self, $($args),*),
                    Air::PushIReg {
                        dst,
                        width: self.output,
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
                    Air::MovIVar(Reg::A, rhs, self.input),
                    crate::instr!($instr, self, $($args),*),
                    Air::PushIReg {
                        dst,
                        width: self.output,
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
                    Air::MovIVar(Reg::B, lhs, self.input),
                    crate::instr!($instr, self, $($args),*),
                    Air::PushIReg {
                        dst,
                        width: self.output,
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
                    crate::instr!($instr, self, $($args),*),
                    Air::PushIReg {
                        dst,
                        width: self.output,
                        src: Reg::A,
                    },
                ]);
            }
        }
    };
}

#[macro_export]
macro_rules! impl_prim_bin_op_visitor {
    ($name:ident, $instr:ident, $($args:ident),*) => {
        impl AllPrimBinOpLeaves for $name {}

        impl BinOpLeaf<u64, u64> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: u64, rhs: u64) {
                ctx.ins_set([
                    Air::MovIConst(
                        Reg::A,
                        ConstData::Bits(Bits::from_width(rhs as u64, self.input)),
                    ),
                    Air::MovIConst(
                        Reg::B,
                        ConstData::Bits(Bits::from_width(lhs as u64, self.input)),
                    ),
                    crate::instr!($instr, self, $($args),*),
                    Air::PushIReg {
                        dst,
                        width: self.output,
                        src: Reg::A,
                    },
                ]);
            }
        }

        impl BinOpLeaf<u64, OffsetVar> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: u64, rhs: OffsetVar) {
                ctx.ins_set([
                    Air::MovIVar(Reg::A, rhs, self.input),
                    Air::MovIConst(
                        Reg::B,
                        ConstData::Bits(Bits::from_width(lhs as u64, self.input)),
                    ),
                    crate::instr!($instr, self, $($args),*),
                    Air::PushIReg {
                        dst,
                        width: self.output,
                        src: Reg::A,
                    },
                ]);
            }
        }

        impl BinOpLeaf<OffsetVar, u64> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: OffsetVar, rhs: u64) {
                ctx.ins_set([
                    Air::MovIVar(Reg::A, lhs, self.input),
                    Air::MovIConst(
                        Reg::B,
                        ConstData::Bits(Bits::from_width(rhs as u64, self.input)),
                    ),
                    crate::instr!($instr, self, $($args),*),
                    Air::PushIReg {
                        dst,
                        width: self.output,
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
                        ConstData::Bits(Bits::from_width(rhs as u64, self.input)),
                    ),
                    crate::instr!($instr, self, $($args),*),
                    Air::PushIReg {
                        dst,
                        width: self.output,
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
                        ConstData::Bits(Bits::from_width(lhs as u64, self.input)),
                    ),
                    crate::instr!($instr, self, $($args),*),
                    Air::PushIReg {
                        dst,
                        width: self.output,
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
                        ConstData::Bits(Bits::from_width_float(rhs, self.input)),
                    ),
                    Air::MovIConst(
                        Reg::B,
                        ConstData::Bits(Bits::from_width_float(lhs, self.input)),
                    ),
                    Air::$instr(self.input),
                    Air::PushIReg {
                        dst,
                        width: self.output,
                        src: Reg::A,
                    },
                ]);
            }
        }

        impl BinOpLeaf<f64, OffsetVar> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: f64, rhs: OffsetVar) {
                ctx.ins(Air::MovIConst(
                    Reg::B,
                    ConstData::Bits(Bits::from_width_float(lhs, self.input)),
                ));

                ctx.ins_set([
                    Air::MovIVar(Reg::A, rhs, self.input),
                    Air::$instr(self.input),
                    Air::PushIReg {
                        dst,
                        width: self.output,
                        src: Reg::A,
                    },
                ]);
            }
        }

        impl BinOpLeaf<OffsetVar, f64> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: OffsetVar, rhs: f64) {
                ctx.ins(Air::MovIConst(
                    Reg::A,
                    ConstData::Bits(Bits::from_width_float(rhs, self.input)),
                ));

                ctx.ins_set([
                    Air::MovIVar(Reg::B, lhs, self.input),
                    Air::$instr(self.input),
                    Air::PushIReg {
                        dst,
                        width: self.output,
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
                    ConstData::Bits(Bits::from_width_float(rhs, self.input)),
                ));

                ctx.ins_set([
                    Air::$instr(self.input),
                    Air::PushIReg {
                        dst,
                        width: self.output,
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
                    ConstData::Bits(Bits::from_width_float(lhs, self.input)),
                ));

                ctx.ins_set([
                    Air::$instr(self.input),
                    Air::PushIReg {
                        dst,
                        width: self.output,
                        src: Reg::A,
                    },
                ]);
            }
        }
    };
}
