use super::OffsetVar;
use super::ctx::AirCtx;
use crate::air::{Air, Bits, ConstData, Reg, assign_expr, extract_var_from_expr};
use crate::ir::lit::LitKind;
use crate::ir::ty::{Sign, Ty, TyKind, Width};
use crate::ir::*;
use pebblec_parse::BinOpKind;

/// Evaluate `bin` and assign to `dst`.
///
/// Panics
///     `bin` does not evaluate to `ty`
pub fn assign_bin_op(ctx: &mut AirCtx, dst: OffsetVar, ty: Ty, bin: &BinOp) {
    let out = ty;
    match ty.0 {
        TyKind::Int(ty) => {
            let width = ty.width();
            let sign = ty.sign();
            match bin.kind {
                BinOpKind::Mul => visit(Mul::new(width, sign), ctx, out, dst, bin),
                BinOpKind::Div => visit(Div::new(width, sign), ctx, out, dst, bin),
                BinOpKind::Rem => visit(Rem::new(width, sign), ctx, out, dst, bin),

                BinOpKind::Add => visit(Add::new(width, sign), ctx, out, dst, bin),
                BinOpKind::Sub => visit(Sub::new(width, sign), ctx, out, dst, bin),

                BinOpKind::Shl => visit(Shl::new(width, sign), ctx, out, dst, bin),
                BinOpKind::Shr => visit(Shr::new(width, sign), ctx, out, dst, bin),

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
        TyKind::Float(ty) => {
            let width = ty.width();
            let sign = Sign::I;

            match bin.kind {
                BinOpKind::Mul => visit(FloatMul::new(width), ctx, out, dst, bin),
                BinOpKind::Div => visit(FloatDiv::new(width), ctx, out, dst, bin),
                BinOpKind::Rem => visit(FloatRem::new(width), ctx, out, dst, bin),

                BinOpKind::Add => visit(FloatAdd::new(width), ctx, out, dst, bin),
                BinOpKind::Sub => visit(FloatSub::new(width), ctx, out, dst, bin),

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
        TyKind::Bool => {
            match bin.kind {
                BinOpKind::Or => {
                    assert!(ty.is_bool());
                    assert_eq!(bin.lhs.infer(ctx), InferTy::Ty(Ty::BOOL));
                    assert_eq!(bin.rhs.infer(ctx), InferTy::Ty(Ty::BOOL));
                    assign_logical_or(ctx, bin.lhs, bin.rhs, dst);
                    return;
                }
                BinOpKind::And => {
                    assert!(ty.is_bool());
                    assert_eq!(bin.lhs.infer(ctx), InferTy::Ty(Ty::BOOL));
                    assert_eq!(bin.rhs.infer(ctx), InferTy::Ty(Ty::BOOL));
                    assign_logical_and(ctx, bin.lhs, bin.rhs, dst);
                    return;
                }
                _ => {}
            }

            let ty = match (bin.lhs.infer(ctx), bin.rhs.infer(ctx)) {
                (InferTy::Ty(lhs), InferTy::Ty(rhs)) => {
                    assert_eq!(lhs, rhs);
                    lhs
                }
                (InferTy::Ty(ty), _) | (_, InferTy::Ty(ty)) => ty,
                (InferTy::Int, other) => {
                    assert_eq!(InferTy::Int, other);
                    Ty::ISIZE
                }
                (InferTy::Float, other) => {
                    assert_eq!(InferTy::Float, other);
                    Ty::FSIZE
                }
            };

            let lhs = prepare_expr(ctx, ty, bin.lhs);
            let rhs = prepare_expr(ctx, ty, bin.rhs);

            let (width, sign) = match ty.0 {
                TyKind::Int(int) => (int.width(), int.sign()),
                TyKind::Float(float) => (float.width(), Sign::I),
                TyKind::Bool => (Width::BOOL, Sign::U),
                TyKind::Ref(TyKind::Str) => unreachable!(),
                TyKind::Ref(_) => {
                    // TODO: NULL
                    (Width::PTR, Sign::U)
                }
                ty => unreachable!("{ty:#?}"),
            };

            match bin.kind {
                BinOpKind::Add
                | BinOpKind::Sub
                | BinOpKind::Mul
                | BinOpKind::Div
                | BinOpKind::Rem
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
                _ => unreachable!(),
            }
        }
        ty => panic!("invalid type: {ty:#?}"),
    }
}

pub fn aquire_accessor_field(ctx: &mut AirCtx, access: &Access) -> (OffsetVar, Ty) {
    let ty = access.lhs.infer_abs(ctx).unwrap();
    let var = extract_var_from_expr(ctx, ty, access.lhs);

    let id = ty.expect_struct();
    let mut strukt = ctx.tys.strukt(id);

    let mut offset = 0;
    for (i, acc) in access.accessors.iter().rev().enumerate() {
        let ty = strukt.field_ty(acc.sym);
        if i == access.accessors.len() - 1 {
            let field_offset = strukt.field_offset(&ctx.tys, acc.sym);
            return (var.add((offset + field_offset) as usize), ty);
        }

        match ty.0 {
            TyKind::Struct(id) => {
                let field_offset = strukt.field_offset(&ctx.tys, acc.sym);
                offset += field_offset;
                strukt = ctx.tys.strukt(*id);
            }
            TyKind::Array(_, _)
            | TyKind::Slice(_)
            | TyKind::Ref(_)
            | TyKind::Str
            | TyKind::Int(_)
            | TyKind::Float(_)
            | TyKind::Bool
            | TyKind::Unit => {
                panic!("cannot access field on {ty:?}")
            }
        }
    }
    unreachable!()
}

fn assign_logical_and(ctx: &mut AirCtx, lhs: &Expr, rhs: &Expr, dst: OffsetVar) {
    let mut ordered = Vec::new();
    flatten_logical_exprs(BinOpKind::And, lhs, rhs, &mut ordered);
    assert!(ordered.len() >= 2);

    ctx.push_pop_sp(|ctx| {
        let exit = ctx.new_block();
        let creg = Reg::A;

        let blocks = (0..ordered.len())
            .map(|_| ctx.new_block())
            .collect::<Vec<_>>();

        for (i, (expr, next_block)) in ordered.iter().zip(blocks.into_iter()).enumerate() {
            if i != ordered.len() - 1 {
                assign_expr(ctx, dst, Ty::BOOL, expr);
                ctx.ins_set([
                    Air::MovIVar(creg, dst, Width::W8),
                    Air::IfElse {
                        condition: creg,
                        then: next_block,
                        otherwise: exit,
                    },
                ]);
                ctx.set_active_block(next_block);
            } else {
                assign_expr(ctx, dst, Ty::BOOL, expr);
                ctx.ins(Air::Jmp(exit));
            }
        }

        ctx.set_active_block(exit);
    });
}

fn assign_logical_or(ctx: &mut AirCtx, lhs: &Expr, rhs: &Expr, dst: OffsetVar) {
    let mut ordered = Vec::new();
    flatten_logical_exprs(BinOpKind::Or, lhs, rhs, &mut ordered);
    assert!(ordered.len() >= 2);

    ctx.push_pop_sp(|ctx| {
        let exit = ctx.new_block();
        let creg = Reg::A;

        let blocks = (0..ordered.len())
            .map(|_| ctx.new_block())
            .collect::<Vec<_>>();

        for (i, (expr, next_block)) in ordered.iter().zip(blocks.into_iter()).enumerate() {
            if i != ordered.len() - 1 {
                assign_expr(ctx, dst, Ty::BOOL, expr);
                ctx.ins_set([
                    Air::MovIVar(creg, dst, Width::W8),
                    Air::IfElse {
                        condition: creg,
                        then: exit,
                        otherwise: next_block,
                    },
                ]);
                ctx.set_active_block(next_block);
            } else {
                assign_expr(ctx, dst, Ty::BOOL, expr);
                ctx.ins(Air::Jmp(exit));
            }
        }

        ctx.set_active_block(exit);
    });
}

fn flatten_logical_exprs<'a>(
    op: BinOpKind,
    lhs: &'a Expr<'a>,
    rhs: &'a Expr<'a>,
    output: &mut Vec<&'a Expr<'a>>,
) {
    match lhs {
        Expr::Bin(bin) => {
            if bin.kind == op {
                flatten_logical_exprs(op, bin.lhs, bin.rhs, output);
            } else {
                output.push(lhs);
            }
        }
        _ => output.push(lhs),
    }

    output.push(rhs);
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

    (Bits, $name:ident, $strukt:ident) => {
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

impl_op!(Int, mul, Mul);
impl_op!(Int, div, Div);
impl_op!(Int, rem, Rem);

impl_op!(Int, add, Add);
impl_op!(Int, sub, Sub);

impl_op!(Bits, xor, Xor);
impl_op!(Bits, and, Band);
impl_op!(Bits, or, Bor);

impl_op!(Int, shl, Shl);
impl_op!(Int, shr, Shr);

impl_op!(Float, fadd, FloatAdd);
impl_op!(Float, fsub, FloatSub);
impl_op!(Float, fmul, FloatMul);
impl_op!(Float, fdiv, FloatDiv);
impl_op!(Float, frem, FloatRem);

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

#[macro_export]
macro_rules! ge {
    ($ctx:ident, $width:expr, $sign:expr, $dst:expr, $lhs:expr, $rhs:expr) => {{
        use crate::air::bin::BinOpLeaf;
        crate::air::bin::Ge {
            output: Width::W8,
            input: $width,
            sign: $sign,
        }
        .visit_leaf($ctx, $dst, $rhs, $lhs);
    }};
}
#[allow(unused)]
pub use ge;

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

int_op!(Mul, MulAB);
int_op!(Div, DivAB);
int_op!(Rem, RemAB);

int_op!(Add, AddAB);
int_op!(Sub, SubAB);

bit_op!(Band, BandAB);
bit_op!(Xor, XorAB);
bit_op!(Bor, BorAB);

int_op!(Shl, ShlAB);
int_op!(Shr, ShrAB);

cmp_op!(Eq, EqAB, FEqAB);
cmp_op!(Ne, NEqAB, NFEqAB);
cmp_op!(Lt, LtAB, FLtAB);
cmp_op!(Gt, GtAB, FGtAB);
cmp_op!(Le, LeAB, FLeAB);
cmp_op!(Ge, GeAB, FGeAB);

float_op!(FloatMul, FMulAB);
float_op!(FloatDiv, FDivAB);
float_op!(FloatRem, FRemAB);

float_op!(FloatAdd, FAddAB);
float_op!(FloatSub, FSubAB);

pub enum BinOpArg {
    Var(OffsetVar),
    Int(u64),
    Float(f64),
}

fn visit(op: impl BinOpVisitWith, ctx: &mut AirCtx, ty: Ty, dst: OffsetVar, bin: &BinOp) {
    let lhs = prepare_expr(ctx, ty, bin.lhs);
    let rhs = prepare_expr(ctx, ty, bin.rhs);
    op.visit_with(ctx, dst, lhs, rhs);
}

fn prepare_expr(ctx: &mut AirCtx, ty: Ty, expr: &Expr) -> BinOpArg {
    match expr {
        Expr::Lit(lit) => match lit.kind {
            LitKind::Int(lit) => {
                assert!(ty.is_int());
                BinOpArg::Int(*lit)
            }
            LitKind::Float(float) => {
                assert!(ty.is_float());
                BinOpArg::Float(*float)
            }
        },
        _ => BinOpArg::Var(extract_var_from_expr(ctx, ty, expr)),
    }
}

pub trait BinOpVisitWith {
    fn visit_with(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: BinOpArg, rhs: BinOpArg);
}

pub trait FloatAlgebra: AgnosticBinOpLeaves + AllFloatBinOpLeaves {}

#[macro_export]
macro_rules! impl_float_algebra {
    ($ty:ident) => {
        impl FloatAlgebra for $ty {}
        impl BinOpVisitWith for $ty {
            fn visit_with(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: BinOpArg, rhs: BinOpArg) {
                visit_float(self, ctx, dst, rhs, lhs);
            }
        }
    };
}

fn visit_float(
    op: &impl FloatAlgebra,
    ctx: &mut AirCtx,
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
            fn visit_with(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: BinOpArg, rhs: BinOpArg) {
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
            fn visit_with(&self, ctx: &mut AirCtx, dst: OffsetVar, lhs: BinOpArg, rhs: BinOpArg) {
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

fn visit_int(op: &impl IntAlgebra, ctx: &mut AirCtx, dst: OffsetVar, lhs: BinOpArg, rhs: BinOpArg) {
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
