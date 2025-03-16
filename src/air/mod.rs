use crate::ir::lit::{Lit, LitKind};
use crate::ir::sig::{Param, Sig};
use crate::ir::strukt::StructDef;
use crate::ir::ty::store::TyId;
use crate::ir::ty::{FloatTy, IntTy, Sign, Ty, Width};
use crate::ir::*;
use crate::parse::rules::prelude::Attr;
use bin::*;
use ctx::*;
use data::BssEntry;
use std::collections::HashMap;
use std::ops::Range;

mod bin;
pub mod ctx;
mod data;

/// Analyzed Intermediate Representation.
///
/// `Air` is a collection of low level instructions that are intended to be easily executable as
/// byte-code and lowerable in a backend.
///
/// TODO: break out IntKind in favor of byte slices with sign extension?
#[derive(Debug)]
pub enum Air<'a> {
    Ret,

    Call(&'a Sig<'a>, Args),

    /// Swap the A and B registers.
    SwapReg,
    MovIVar(Reg, OffsetVar, Width),
    MovIConst(Reg, ConstData),

    SAlloc(Var, Bytes),

    /// Load address of `Var` into `Reg`.
    Addr(Reg, OffsetVar),

    /// Non overlapping copy.
    MemCpy {
        dst: Reg,
        src: Reg,
        bytes: usize,
    },

    IfElse {
        /// The evalauted boolean condition.
        condition: Reg,
        then: BlockId,
        otherwise: BlockId,
    },
    Jmp(BlockId),

    ReadSP(OffsetVar),
    WriteSP(OffsetVar),

    PushIConst(OffsetVar, ConstData),
    PushIReg {
        dst: OffsetVar,
        width: Width,
        src: Reg,
    },
    PushIVar {
        dst: OffsetVar,
        width: Width,
        src: OffsetVar,
    },

    Read {
        dst: Reg,
        addr: Reg,
        width: Width,
    },
    Write {
        addr: Reg,
        data: Reg,
        width: Width,
    },

    /// Binary operations use [`Reg::A`] and [`Reg::B`], then store result in [`Reg::A`].
    ///
    /// TODO: need sign and overflow settings
    AddAB(Width),
    SubAB(Width),
    MulAB(Width),
    DivAB(Width),

    XorAB(Width),

    EqAB(Width),
    NEqAB(Width),

    FAddAB(Width),
    FSubAB(Width),
    FMulAB(Width),
    FDivAB(Width),

    FEqAB(Width),
    NFEqAB(Width),

    /// Bitwise operations read from, evaluate, then load back into [`Reg::A`].
    XOR(Mask),

    /// Cast [`Reg::A`] from an f32 to a u32.
    Fu32,

    FSqrt(FloatTy),

    /// Exit with code stored in [`Reg::A`].
    Exit,
    /// The address of `fmt` should be loaded into [`Reg::A`].
    PrintCStr,
}

pub type Addr = u64;
pub type Mask = u64;

#[derive(Debug, Clone, Copy)]
pub enum Bits {
    B8(u8),
    B16(u16),
    B32(u32),
    B64(u64),
}

#[allow(unused)]
impl Bits {
    pub const TRUE: Self = Self::B8(1);
    pub const FALSE: Self = Self::B8(0);

    pub fn from_u8(bits: u8) -> Self {
        Self::B8(bits)
    }

    pub fn from_u16(bits: u16) -> Self {
        Self::B16(bits)
    }

    pub fn from_u32(bits: u32) -> Self {
        Self::B32(bits)
    }

    pub fn from_u64(bits: u64) -> Self {
        Self::B64(bits)
    }

    pub fn from_i8(bits: i8) -> Self {
        Self::B8(unsafe { std::mem::transmute(bits) })
    }

    pub fn from_i16(bits: i16) -> Self {
        Self::B16(unsafe { std::mem::transmute(bits) })
    }

    pub fn from_i32(bits: i32) -> Self {
        Self::B32(unsafe { std::mem::transmute(bits) })
    }

    pub fn from_i64(bits: i64) -> Self {
        Self::B64(unsafe { std::mem::transmute(bits) })
    }

    pub fn from_f32(bits: f32) -> Self {
        Self::B32(bits.to_bits())
    }

    pub fn from_f64(bits: f64) -> Self {
        Self::B64(bits.to_bits())
    }

    pub fn from_width(bits: u64, width: Width) -> Self {
        match width {
            Width::W8 => Self::B8(bits as u8),
            Width::W16 => Self::B16(bits as u16),
            Width::W32 => Self::B32(bits as u32),
            Width::W64 => Self::B64(bits as u64),
        }
    }

    pub fn from_width_float(bits: f64, width: Width) -> Self {
        match width {
            Width::W32 => Bits::from_u32((bits as f32).to_bits()),
            Width::W64 => Bits::from_f64(bits),
            _ => unreachable!(),
        }
    }

    pub fn to_u64(self) -> u64 {
        match self {
            Self::B8(b) => b as u64,
            Self::B16(b) => b as u64,
            Self::B32(b) => b as u64,
            Self::B64(b) => b,
        }
    }
}

#[derive(Debug)]
pub enum ConstData {
    Bits(Bits),
    Ptr(BssEntry),
}

/// Collection of [`Air`] instructions for a [`crate::ir::Func`].
#[derive(Debug)]
pub struct AirFunc<'a> {
    pub func: &'a Func<'a>,
    instrs: Vec<Air<'a>>,
    blocks: HashMap<BlockId, Range<usize>>,
}

impl<'a> AirFunc<'a> {
    pub fn new(func: &'a Func<'a>, blocks: Vec<Vec<Air<'a>>>) -> Self {
        let mut instrs = Vec::new();
        let mut ranges = HashMap::with_capacity(blocks.len());
        for (hash, block_instrs) in blocks.into_iter().enumerate() {
            let start = instrs.len();
            instrs.extend(block_instrs);
            let end = instrs.len();
            ranges.insert(BlockId(hash), start..end);
        }

        Self {
            blocks: ranges,
            func,
            instrs,
        }
    }

    pub fn start(&self) -> &[Air<'a>] {
        self.block(BlockId(0))
    }

    pub fn start_block(&self) -> BlockId {
        BlockId(0)
    }

    #[track_caller]
    pub fn block(&self, block: BlockId) -> &[Air<'a>] {
        self.blocks
            .get(&block)
            .map(|range| &self.instrs[range.start..range.end])
            .expect("invalid block")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(usize);

#[derive(Debug)]
pub struct AirFuncBuilder<'a> {
    pub func: &'a Func<'a>,
    instrs: Vec<Vec<Air<'a>>>,
    active: BlockId,
    loop_ctx: Option<LoopCtx>,
}

impl<'a> AirFuncBuilder<'a> {
    pub fn new(func: &'a Func<'a>) -> Self {
        Self {
            instrs: vec![Vec::new()],
            active: BlockId(0),
            loop_ctx: None,
            func,
        }
    }

    pub fn new_block(&mut self) -> BlockId {
        let id = BlockId(self.instrs.len());
        self.instrs.push(Vec::new());
        id
    }

    #[track_caller]
    pub fn insert_active(&mut self, instr: Air<'a>) {
        self.instrs[self.active.0].push(instr);
    }

    #[track_caller]
    pub fn insert_active_set(&mut self, instrs: impl IntoIterator<Item = Air<'a>>) {
        self.instrs[self.active.0].extend(instrs);
    }

    #[track_caller]
    pub fn build(&mut self) -> AirFunc<'a> {
        assert!(!self.instrs.is_empty());
        AirFunc::new(self.func, std::mem::take(&mut self.instrs))
    }
}

#[derive(Debug, Default, Clone)]
pub struct Args {
    pub vars: Vec<(TyId, Var)>,
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
    pub fn new(var: Var, offset: impl Sized) -> Self {
        Self {
            var,
            offset: offset.size(),
        }
    }

    pub fn zero(var: Var) -> Self {
        Self::new(var, 0)
    }

    pub fn add(&self, offset: impl Sized) -> Self {
        Self {
            var: self.var,
            offset: self.offset + offset.size(),
        }
    }
}

pub trait Sized {
    fn size(&self) -> usize;
}

impl Sized for usize {
    fn size(&self) -> usize {
        *self
    }
}

impl Sized for Width {
    fn size(&self) -> usize {
        self.bytes()
    }
}

impl Sized for IntKind {
    fn size(&self) -> usize {
        self.size()
    }
}

impl Sized for IntTy {
    fn size(&self) -> usize {
        self.kind().size()
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

#[allow(unused)]
impl IntKind {
    // TODO: unify all these types sizes into one place, this is spread out everywhere
    pub const BOOL: Self = IntKind::U8;
    pub const PTR: Self = IntKind::I64;
    pub const USIZE: Self = IntKind::U64;

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

    pub fn size(&self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 => 4,
            Self::I64 | Self::U64 => 8,
        }
    }
}

impl IntTy {
    pub fn kind(&self) -> IntKind {
        match self.sign() {
            Sign::I => match self.width() {
                Width::W8 => IntKind::I8,
                Width::W16 => IntKind::I16,
                Width::W32 => IntKind::I32,
                Width::W64 => IntKind::I64,
            },
            Sign::U => match self.width() {
                Width::W8 => IntKind::U8,
                Width::W16 => IntKind::U16,
                Width::W32 => IntKind::U32,
                Width::W64 => IntKind::U64,
            },
        }
    }
}

pub fn lower_const<'a>(ctx: &mut AirCtx<'a>, konst: &Const) -> Vec<Air<'a>> {
    ctx.start_const();
    let dst = OffsetVar::zero(ctx.new_var_registered(&konst.name, konst.ty));
    match konst.expr {
        Expr::Lit(lit) => match lit.kind {
            LitKind::Int(int) => {
                let width = match ctx.tys.ty(konst.ty) {
                    Ty::Int(ty) => ty.width(),
                    _ => unreachable!(),
                };

                ctx.ins(Air::PushIConst(
                    dst,
                    ConstData::Bits(Bits::from_width(*int as u64, width)),
                ));
            }
            LitKind::Float(float) => {
                let ty = match ctx.tys.ty(konst.ty) {
                    Ty::Float(ty) => ty,
                    _ => unreachable!(),
                };

                ctx.ins(Air::PushIConst(
                    dst,
                    ConstData::Bits(Bits::from_width_float(*float, ty.width())),
                ));
            }
        },
        _ => unimplemented!(),
    }
    ctx.finish_const()
}

pub fn lower_func<'a, 'b>(ctx: &'b mut AirCtx<'a>, func: &'a Func) -> AirFunc<'a> {
    if func.has_attr(Attr::Intrinsic) {
        return lower_intrinsic(ctx, func);
    }

    ctx.in_var_scope(|ctx| {
        ctx.start_func(func);
        init_params(ctx, func);

        if ctx.tys.is_unit(func.sig.ty) {
            air_block(ctx, &func.block);
            ctx.ins(Air::Ret);
        } else {
            let dst = OffsetVar::zero(ctx.anon_var(func.sig.ty));
            assign_air_block(ctx, dst, func.sig.ty, &func.block);
            ctx.ret_var(dst, func.sig.ty);
        }
    });
    ctx.finish_func()
}

pub fn lower_intrinsic<'a, 'b>(ctx: &'b mut AirCtx<'a>, func: &'a Func) -> AirFunc<'a> {
    ctx.in_var_scope(|ctx| {
        init_params(ctx, func);
        match ctx.expect_ident(func.sig.ident) {
            "exit" => exit(ctx, func),
            "printf" => printf(ctx, func),
            "cs" => c_str(ctx, func),
            "print_cs" => print_c_str(ctx, func),
            "fu32" => fu32(ctx, func),
            "sqrt_f32" => sqrt_f32(ctx, func),
            i => unimplemented!("intrinsic: {i}"),
        }
    })
}

pub fn exit<'a, 'b>(ctx: &'b mut AirCtx<'a>, func: &'a Func) -> AirFunc<'a> {
    ctx.start_func(func);
    let Param { ident, .. } = func.sig.params.iter().next().unwrap();

    let var = ctx.expect_var(ident.id);
    ctx.ins_set([
        Air::MovIVar(Reg::A, OffsetVar::zero(var), Width::W32),
        Air::Exit,
    ]);
    ctx.finish_func()
}

pub fn c_str<'a, 'b>(ctx: &'b mut AirCtx<'a>, func: &'a Func) -> AirFunc<'a> {
    ctx.start_func(func);
    let Param { ident, .. } = func.sig.params.iter().next().unwrap();

    let var = ctx.expect_var(ident.id);
    ctx.ins_set([
        Air::Addr(Reg::A, OffsetVar::new(var, IntKind::USIZE)),
        Air::Ret,
    ]);
    ctx.finish_func()
}

pub fn print_c_str<'a, 'b>(ctx: &'b mut AirCtx<'a>, func: &'a Func) -> AirFunc<'a> {
    ctx.start_func(func);
    let Param { ident, .. } = func.sig.params.iter().next().unwrap();
    let var = ctx.expect_var(ident.id);
    ctx.ins_set([
        Air::MovIVar(Reg::A, OffsetVar::zero(var), Width::PTR),
        Air::PrintCStr,
        Air::Ret,
    ]);
    ctx.finish_func()
}

pub fn printf<'a, 'b>(ctx: &'b mut AirCtx<'a>, func: &'a Func) -> AirFunc<'a> {
    assert_eq!(func.sig.params.len(), 1);
    ctx.start_func(func);
    ctx.ins_set([Air::Ret]);
    ctx.finish_func()
}

pub fn fu32<'a, 'b>(ctx: &'b mut AirCtx<'a>, func: &'a Func) -> AirFunc<'a> {
    ctx.start_func(func);
    let Param { ident, ty, .. } = func.sig.params.iter().next().unwrap();
    ctx.new_var_registered_no_salloc(ident, *ty);

    let var = ctx.expect_var(ident.id);
    ctx.ins_set([
        Air::MovIVar(Reg::A, OffsetVar::zero(var), Width::W32),
        Air::Fu32,
        Air::Ret,
    ]);
    ctx.finish_func()
}

pub fn sqrt_f32<'a, 'b>(ctx: &'b mut AirCtx<'a>, func: &'a Func) -> AirFunc<'a> {
    ctx.start_func(func);
    let Param { ident, .. } = func.sig.params.iter().next().unwrap();

    let var = ctx.expect_var(ident.id);
    ctx.ins_set([
        Air::MovIVar(Reg::A, OffsetVar::zero(var), Width::W32),
        Air::FSqrt(FloatTy::F32),
        Air::Ret,
    ]);
    ctx.finish_func()
}

fn init_params(ctx: &mut AirCtx, func: &Func) {
    for Param { ident, ty, .. } in func.sig.params.iter() {
        ctx.func_arg_var(*ident, *ty);
    }
}

fn air_block<'a>(ctx: &mut AirCtx<'a>, block: &'a Block<'a>) {
    block_stmts(ctx, block.stmts);
    match block.end {
        Some(end) => {
            // TODO: ensure that this is always unit
            eval_expr(ctx, end);
        }
        None => {}
    }
}

#[track_caller]
fn assign_air_block<'a>(ctx: &mut AirCtx<'a>, dst: OffsetVar, ty: TyId, block: &'a Block<'a>) {
    block_stmts(ctx, block.stmts);
    if let Some(end) = &block.end {
        assign_expr(ctx, dst, ty, end);
    } else {
        // TODO: need analysis of return statements
        //println!("{block:?}");
        //assert!(ty == ctx.tys.unit());
    }
}

fn block_stmts<'a>(ctx: &mut AirCtx<'a>, stmts: &'a [Stmt<'a>]) {
    for stmt in stmts.iter() {
        match stmt {
            Stmt::Semi(stmt) => match stmt {
                SemiStmt::Let(let_) => air_let_stmt(ctx, let_),
                SemiStmt::Assign(assign) => air_assign_stmt(ctx, assign),
                SemiStmt::Ret(ret) => match &ret.expr {
                    Some(expr) => {
                        air_return(ctx, ctx.active_sig().ty, &expr);
                    }
                    None => ctx.ins(Air::Ret),
                },
                SemiStmt::Expr(expr) => {
                    eval_expr(ctx, expr);
                }
            },
            Stmt::Open(expr) => eval_expr(ctx, expr),
        }
    }
}

fn air_let_stmt<'a>(ctx: &mut AirCtx<'a>, stmt: &'a Let) {
    match &stmt.lhs {
        LetTarget::Ident(ident) => {
            let ty = ctx.var_ty(ident);
            let dst = ctx.new_var_registered(ident, ty);
            assign_expr(ctx, OffsetVar::zero(dst), ty, &stmt.rhs);
        }
    }
}

fn load_addr_index_of<'a>(ctx: &mut AirCtx<'a>, index: &'a IndexOf, dst: Reg) -> TyId {
    let arr_ty = index.array.infer_abs(ctx).unwrap();
    let ty = match ctx.tys.ty(arr_ty) {
        Ty::Array(_, inner) => ctx.tys.get_ty_id(inner).unwrap(),
        _ => unreachable!(),
    };

    let index_ty = ctx.tys.builtin(Ty::Int(IntTy::USIZE));
    let index_var = OffsetVar::zero(ctx.anon_var(index_ty));
    assign_expr(ctx, index_var, index_ty, index.index);
    ctx.ins_set([
        Air::MovIVar(Reg::A, index_var, Width::SIZE),
        Air::MovIConst(
            Reg::B,
            ConstData::Bits(Bits::from_u64(ctx.tys.ty(ty).size(ctx) as u64)),
        ),
        Air::MulAB(Width::SIZE),
        Air::PushIReg {
            dst: index_var,
            width: Width::SIZE,
            src: Reg::A,
        },
    ]);

    let addr_reg = Reg::A;
    let ptr_offset_reg = Reg::B;
    match index.array {
        Expr::Ident(ident) => {
            let var = OffsetVar::zero(ctx.expect_var(ident.id));
            ctx.ins(Air::Addr(addr_reg, var));
        }
        _ => todo!(),
    }

    ctx.ins_set([
        Air::MovIVar(ptr_offset_reg, index_var, Width::PTR),
        // the address of the array is in `A` and the offset is in `B`
        Air::AddAB(Width::PTR),
    ]);

    if dst == Reg::B {
        ctx.ins(Air::SwapReg);
    } else {
        assert_eq!(dst, Reg::A);
    }

    match ctx.tys.ty(ty) {
        //Ty::Int(int) => {}
        //Ty::Bool => {}
        //Ty::Float(_float) => {}
        Ty::Ref(&Ty::Str) => todo!(),
        Ty::Bool | Ty::Ref(_) => {}
        _ => {} //ty => todo!("load addr of ty: {ty:#?}"),
    }

    ty
}

fn assign_expr<'a>(ctx: &mut AirCtx<'a>, dst: OffsetVar, ty: TyId, expr: &'a Expr) {
    match &expr {
        Expr::IndexOf(index) => {
            load_addr_index_of(ctx, index, Reg::A);
            ctx.ins_set([
                Air::Read {
                    dst: Reg::A,
                    addr: Reg::A,
                    width: Width::PTR,
                },
                Air::PushIReg {
                    dst,
                    width: Width::PTR,
                    src: Reg::A,
                },
            ]);
        }
        Expr::Array(arr) => match ctx.tys.ty(ty) {
            Ty::Array(len, inner) => {
                assert_eq!(len, arr.exprs.len());
                let inner_id = ctx.tys.get_ty_id(inner).unwrap();
                for (i, expr) in arr.exprs.iter().enumerate() {
                    assign_expr(ctx, dst.add(i * inner.size(ctx)), inner_id, expr);
                }
            }
            _ => unreachable!(),
        },
        Expr::Bool(bool) => {
            assert_eq!(ty, TyId::BOOL);
            ctx.ins(Air::PushIConst(
                dst,
                ConstData::Bits(if bool.val { Bits::TRUE } else { Bits::FALSE }),
            ));
        }
        Expr::Str(str) => {
            assert_eq!(ty, TyId::STR_LIT);
            let (entry, len) = ctx.str_lit(str.val);
            ctx.ins_set([
                Air::PushIConst(dst, ConstData::Bits(Bits::from_u64(len as u64))),
                Air::PushIConst(dst.add(IntKind::U64), ConstData::Ptr(entry)),
            ]);
        }
        Expr::Lit(lit) => {
            assign_lit(ctx, lit, dst, ty);
        }
        Expr::Bin(bin) => {
            assign_bin_op(ctx, dst, ty, bin);
        }
        Expr::Struct(def) => {
            assert_eq!(ty, ctx.tys.struct_ty_id(def.id));
            define_struct(ctx, def, dst);
        }
        Expr::Call(call) => {
            assert_eq!(ty, call.sig.ty);

            ctx.push_pop_sp(|ctx| {
                let args = generate_args(ctx, call);
                ctx.ins(Air::Call(&call.sig, args));
            });
            match ctx.tys.ty(call.sig.ty) {
                Ty::Int(ty) => {
                    ctx.ins(Air::PushIReg {
                        dst,
                        width: ty.width(),
                        src: Reg::A,
                    });
                }
                Ty::Float(ty) => {
                    ctx.ins(Air::PushIReg {
                        dst,
                        width: ty.width(),
                        src: Reg::A,
                    });
                }
                Ty::Ref(&Ty::Str) => {
                    todo!();
                    //ctx.ins_set([
                    //    Air::PushIReg {
                    //        dst,
                    //        width: Width::PTR,
                    //        src: Reg::A,
                    //    },
                    //    Air::PushIReg {
                    //        dst: dst.add(Width::PTR),
                    //        width: Width::PTR,
                    //        src: Reg::A,
                    //    },
                    //]);
                }
                Ty::Ref(_) => {
                    ctx.ins(Air::PushIReg {
                        dst,
                        width: Width::PTR,
                        src: Reg::A,
                    });
                }
                Ty::Array(len, inner) => {
                    let bytes = inner.size(ctx) * len;
                    ctx.ins_set([
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
                Ty::Struct(s) => {
                    let bytes = ctx.tys.struct_layout(s).size;
                    ctx.ins_set([
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
                Ty::Bool => {
                    ctx.ins(Air::PushIReg {
                        dst,
                        width: Width::BOOL,
                        src: Reg::A,
                    });
                }
                Ty::Str => {
                    panic!("cannot assign to str");
                }
                Ty::Unit => todo!(),
            }
        }
        Expr::Ident(ident) => {
            let other = OffsetVar::zero(ctx.expect_var(ident.id));
            assign_var_other(ctx, dst, other, ty);
        }
        Expr::Access(access) => {
            let (other, field_ty) = aquire_accessor_field(ctx, access);
            assert_eq!(field_ty, ty);
            assign_var_other(ctx, dst, other, field_ty);
        }
        Expr::If(if_) => {
            assign_if(ctx, dst, ty, if_);
        }
        Expr::Unary(unary) => match unary.kind {
            UOpKind::Ref => {
                assert!(ctx.tys.ty(ty).is_ref());
                match unary.inner {
                    Expr::Ident(ident) => {
                        let var = ctx.expect_var(ident.id);
                        ctx.ins_set([
                            Air::Addr(Reg::A, OffsetVar::zero(var)),
                            Air::PushIReg {
                                dst,
                                width: Width::PTR,
                                src: Reg::A,
                            },
                        ]);
                    }
                    Expr::Lit(lit) => match lit.kind {
                        LitKind::Int(int) => {
                            let Some(Ok(inner_ty)) = ctx.tys.ty(ty).ref_inner_ty(ctx) else {
                                unreachable!()
                            };

                            let var = OffsetVar::zero(ctx.anon_var(inner_ty));
                            let width = ctx.tys.ty(inner_ty).expect_int().width();
                            ctx.ins_set([
                                Air::PushIConst(
                                    var,
                                    ConstData::Bits(Bits::from_width(*int as u64, width)),
                                ),
                                Air::Addr(Reg::A, var),
                                Air::PushIReg {
                                    dst,
                                    width: Width::PTR,
                                    src: Reg::A,
                                },
                            ])
                        }
                        LitKind::Float(_) => panic!("invalid op"),
                    },
                    _ => todo!(),
                }
            }
            UOpKind::Deref => {
                match unary.inner.infer(ctx) {
                    InferTy::Int | InferTy::Float => panic!(),
                    InferTy::Ty(ty) => assert!(ctx.tys.ty(ty).is_ref()),
                }

                // load the address into `addr_reg`
                let ptr_var = extract_var_from_expr(ctx, unary.inner);
                let addr_reg = Reg::A;
                ctx.ins(Air::MovIVar(addr_reg, ptr_var, Width::PTR));

                let inner_ty = ctx.tys.ty(ty);
                match inner_ty {
                    Ty::Bool => ctx.ins_set([
                        Air::Read {
                            dst: Reg::A,
                            addr: addr_reg,
                            width: Width::BOOL,
                        },
                        Air::PushIReg {
                            dst,
                            width: Width::BOOL,
                            src: Reg::A,
                        },
                    ]),
                    _ => todo!(),
                }
            }
            UOpKind::Not => {
                let (width, mask) = match ctx.tys.ty(ty) {
                    Ty::Bool => (Width::BOOL, 1),
                    Ty::Int(ty) => (ty.width(), u64::MAX),
                    _ => unreachable!(),
                };
                let result = OffsetVar::zero(ctx.anon_var(ty));
                assign_expr(ctx, result, ty, unary.inner);
                ctx.ins_set([
                    Air::MovIVar(Reg::A, result, width),
                    Air::XOR(mask),
                    Air::PushIReg {
                        dst,
                        width,
                        src: Reg::A,
                    },
                ]);
            }
        },
        Expr::Enum(_) | Expr::Block(_) => todo!(),
        Expr::Continue(_) | Expr::Break(_) => unreachable!(),
        Expr::Range(_) | Expr::For(_) | Expr::Loop(_) => panic!("invalid assignment"),
    }
}

fn extract_var_from_expr(ctx: &mut AirCtx, expr: &Expr) -> OffsetVar {
    match expr {
        Expr::Ident(ident) => OffsetVar::zero(ctx.expect_var(ident.id)),
        _ => todo!(),
    }
}

fn assign_expr_deref<'a>(ctx: &mut AirCtx<'a>, addr: OffsetVar, ty: TyId, expr: &'a Expr) {
    match &expr {
        Expr::Bool(bool) => {
            assert_eq!(ty, TyId::BOOL);
            ctx.ins_set([
                Air::MovIVar(Reg::A, addr, Width::PTR),
                Air::MovIConst(
                    Reg::B,
                    ConstData::Bits(if bool.val { Bits::TRUE } else { Bits::FALSE }),
                ),
                Air::Write {
                    addr: Reg::A,
                    data: Reg::B,
                    width: Width::BOOL,
                },
            ]);
        }
        _ => todo!(),
        //Expr::IndexOf(index) => {
        //    load_addr_index_of(ctx, index, Reg::A);
        //    ctx.ins_set([
        //        Air::Read {
        //            dst: Reg::A,
        //            addr: Reg::A,
        //            width: Width::PTR,
        //        },
        //        Air::PushIReg {
        //            dst,
        //            width: Width::PTR,
        //            src: Reg::A,
        //        },
        //    ]);
        //}
        //Expr::Array(arr) => match ctx.tys.ty(ty) {
        //    Ty::Array(len, inner) => {
        //        assert_eq!(len, arr.exprs.len());
        //        let inner_id = ctx.tys.get_ty_id(inner).unwrap();
        //        for (i, expr) in arr.exprs.iter().enumerate() {
        //            assign_expr(ctx, dst.add(i * inner.size(ctx)), inner_id, expr);
        //        }
        //    }
        //    _ => unreachable!(),
        //},
        //Expr::Str(str) => {
        //    assert_eq!(ty, TyId::STR_LIT);
        //    let (entry, len) = ctx.str_lit(str.val);
        //    ctx.ins_set([
        //        Air::PushIConst(dst, ConstData::Bits(Bits::from_u64(len as u64))),
        //        Air::PushIConst(dst.add(IntKind::U64), ConstData::Ptr(entry)),
        //    ]);
        //}
        //Expr::Lit(lit) => {
        //    assign_lit(ctx, lit, dst, ty);
        //}
        //Expr::Bin(bin) => {
        //    assign_bin_op(ctx, dst, ty, bin);
        //}
        //Expr::Struct(def) => {
        //    assert_eq!(ty, ctx.tys.struct_ty_id(def.id));
        //    define_struct(ctx, def, dst);
        //}
        //Expr::Call(call) => {
        //    assert_eq!(ty, call.sig.ty);
        //
        //    let args = generate_args(ctx, call);
        //    ctx.ins(Air::Call(&call.sig, args));
        //    match ctx.tys.ty(call.sig.ty) {
        //        Ty::Int(ty) => {
        //            ctx.ins(Air::PushIReg {
        //                dst,
        //                width: ty.width(),
        //                src: Reg::A,
        //            });
        //        }
        //        Ty::Float(ty) => {
        //            ctx.ins(Air::PushIReg {
        //                dst,
        //                width: ty.width(),
        //                src: Reg::A,
        //            });
        //        }
        //        Ty::Ref(&Ty::Str) => {
        //            todo!();
        //            //ctx.ins_set([
        //            //    Air::PushIReg {
        //            //        dst,
        //            //        width: Width::PTR,
        //            //        src: Reg::A,
        //            //    },
        //            //    Air::PushIReg {
        //            //        dst: dst.add(Width::PTR),
        //            //        width: Width::PTR,
        //            //        src: Reg::A,
        //            //    },
        //            //]);
        //        }
        //        Ty::Ref(_) => {
        //            ctx.ins(Air::PushIReg {
        //                dst,
        //                width: Width::PTR,
        //                src: Reg::A,
        //            });
        //        }
        //        Ty::Array(len, inner) => {
        //            let bytes = inner.size(ctx) * len;
        //            ctx.ins_set([
        //                Air::Addr(Reg::B, dst),
        //                Air::MemCpy {
        //                    dst: Reg::B,
        //                    src: {
        //                        // the destination is supplied by the callee
        //                        const _: () = assert!(matches!(RET_REG, Reg::A));
        //                        Reg::A
        //                    },
        //                    bytes,
        //                },
        //            ]);
        //        }
        //        Ty::Struct(s) => {
        //            let bytes = ctx.tys.struct_layout(s).size;
        //            ctx.ins_set([
        //                Air::Addr(Reg::B, dst),
        //                Air::MemCpy {
        //                    dst: Reg::B,
        //                    src: {
        //                        // the destination is supplied by the callee
        //                        const _: () = assert!(matches!(RET_REG, Reg::A));
        //                        Reg::A
        //                    },
        //                    bytes,
        //                },
        //            ]);
        //        }
        //        Ty::Bool => {
        //            ctx.ins(Air::PushIReg {
        //                dst,
        //                width: Width::BOOL,
        //                src: Reg::A,
        //            });
        //        }
        //        Ty::Str => {
        //            panic!("cannot assign to str");
        //        }
        //        Ty::Unit => todo!(),
        //    }
        //}
        //Expr::Ident(ident) => {
        //    let other = OffsetVar::zero(ctx.expect_var(ident.id));
        //    assign_var_other(ctx, dst, other, ty);
        //}
        //Expr::Access(access) => {
        //    let (other, field_ty) = aquire_accessor_field(ctx, access);
        //    assert_eq!(field_ty, ty);
        //    assign_var_other(ctx, dst, other, field_ty);
        //}
        //Expr::If(_if_) => {
        //    todo!();
        //    //assign_if(ctx, dst, ty, if_);
        //}
        //Expr::Loop(_) => {
        //    // this means that a loop is the last statement, and in order to leave we must return, so
        //    // just ignore this for now
        //    //assert_eq!(ty, TyId::UNIT);
        //    eval_expr(ctx, expr);
        //}
        //Expr::Unary(unary) => match unary.kind {
        //    UOpKind::Ref => {
        //        assert!(ctx.tys.ty(ty).is_ref());
        //        match unary.inner {
        //            Expr::Ident(ident) => {
        //                let var = ctx.expect_var(ident.id);
        //                ctx.ins_set([
        //                    Air::Addr(Reg::A, OffsetVar::zero(var)),
        //                    Air::PushIReg {
        //                        dst,
        //                        width: Width::PTR,
        //                        src: Reg::A,
        //                    },
        //                ]);
        //            }
        //            Expr::Lit(lit) => match lit.kind {
        //                LitKind::Int(int) => {
        //                    let Some(Ok(inner_ty)) = ctx.tys.ty(ty).ref_inner_ty(ctx) else {
        //                        unreachable!()
        //                    };
        //
        //                    let var = OffsetVar::zero(ctx.anon_var(inner_ty));
        //                    let width = ctx.tys.ty(inner_ty).expect_int().width();
        //                    ctx.ins_set([
        //                        Air::PushIConst(
        //                            var,
        //                            ConstData::Bits(Bits::from_width(*int as u64, width)),
        //                        ),
        //                        Air::Addr(Reg::A, var),
        //                        Air::PushIReg {
        //                            dst,
        //                            width: Width::PTR,
        //                            src: Reg::A,
        //                        },
        //                    ])
        //                }
        //                LitKind::Float(_) => panic!("invalid op"),
        //            },
        //            _ => todo!(),
        //        }
        //    }
        //    UOpKind::Not => {
        //        let (width, mask) = match ctx.tys.ty(ty) {
        //            Ty::Bool => (Width::BOOL, 1),
        //            Ty::Int(ty) => (ty.width(), u64::MAX),
        //            _ => unreachable!(),
        //        };
        //        let result = OffsetVar::zero(ctx.anon_var(ty));
        //        assign_expr(ctx, result, ty, unary.inner);
        //        ctx.ins_set([
        //            Air::MovIVar(Reg::A, result, width),
        //            Air::XOR(mask),
        //            Air::PushIReg {
        //                dst,
        //                width,
        //                src: Reg::A,
        //            },
        //        ]);
        //    }
        //    _ => todo!(),
        //},
        //Expr::Enum(_) | Expr::Block(_) => todo!(),
    }
}

fn assign_var_other<'a>(ctx: &mut AirCtx<'a>, dst: OffsetVar, other: OffsetVar, ty: TyId) {
    match ctx.tys.ty(ty) {
        Ty::Int(ty) => {
            ctx.ins(Air::PushIVar {
                dst,
                width: ty.width(),
                src: other,
            });
        }
        Ty::Float(ty) => {
            ctx.ins_set([Air::PushIVar {
                dst,
                width: ty.width(),
                src: other,
            }]);
        }
        Ty::Ref(&Ty::Str) => {
            ctx.ins_set([
                Air::PushIVar {
                    dst,
                    width: Width::W64,
                    src: other,
                },
                Air::PushIVar {
                    dst: dst.add(Width::PTR),
                    width: Width::PTR,
                    src: other.add(Width::PTR),
                },
            ]);
        }
        Ty::Ref(_) => {
            ctx.ins(Air::PushIVar {
                dst,
                width: Width::PTR,
                src: other,
            });
        }
        Ty::Struct(id) => {
            let bytes = ctx.tys.struct_layout(id).size;
            ctx.ins_set([
                Air::Addr(Reg::B, dst),
                Air::Addr(Reg::A, other),
                Air::MemCpy {
                    dst: Reg::B,
                    src: Reg::A,
                    bytes,
                },
            ]);
        }
        Ty::Bool => {
            ctx.ins(Air::PushIVar {
                dst,
                width: Width::BOOL,
                src: other,
            });
        }
        Ty::Array(len, inner) => {
            ctx.ins_set([
                Air::Addr(Reg::B, other),
                Air::Addr(Reg::A, dst),
                Air::MemCpy {
                    dst: Reg::A,
                    src: Reg::B,
                    bytes: inner.size(ctx) * len,
                },
            ]);
        }
        Ty::Str => {
            panic!("cannot assign to str");
        }
        Ty::Unit => todo!(),
    }
}

fn eval_if<'a>(ctx: &mut AirCtx<'a>, if_: &If<'a>) {
    eval_or_assign_if(ctx, if_, None);
}

fn assign_if<'a>(ctx: &mut AirCtx<'a>, dst: OffsetVar, ty: TyId, if_: &If<'a>) {
    eval_or_assign_if(ctx, if_, Some((dst, ty)));
}

fn eval_or_assign_if<'a>(ctx: &mut AirCtx<'a>, if_: &If<'a>, dst: Option<(OffsetVar, TyId)>) {
    ctx.in_var_scope(|ctx| {
        let bool_ = ctx.tys.bool();
        let condition = OffsetVar::zero(ctx.anon_var(bool_));

        ctx.push_pop_sp(|ctx| {
            assign_expr(ctx, condition, bool_, if_.condition);
            ctx.ins(Air::MovIVar(Reg::A, condition, Width::BOOL));

            match (if_.block, if_.otherwise) {
                (Expr::Block(then), Some(Expr::Block(otherwise))) => {
                    let exit = ctx.new_block();
                    let then = ctx.in_scope(|ctx, _| {
                        if let Some((var, ty)) = dst {
                            assign_air_block(ctx, var, ty, then);
                        } else {
                            air_block(ctx, then);
                        }
                        ctx.ins(Air::Jmp(exit));
                    });
                    let otherwise = ctx.in_scope(|ctx, _| {
                        if let Some((var, ty)) = dst {
                            assign_air_block(ctx, var, ty, otherwise);
                        } else {
                            air_block(ctx, otherwise);
                        }
                        ctx.ins(Air::Jmp(exit));
                    });
                    ctx.ins(Air::IfElse {
                        condition: Reg::A,
                        then,
                        otherwise,
                    });
                    ctx.set_active_block(exit);
                }
                (Expr::Block(then), None) => {
                    let otherwise = ctx.new_block();
                    let then = ctx.in_scope(|ctx, _| {
                        assert!(dst.is_none());
                        air_block(ctx, then);
                        ctx.ins(Air::Jmp(otherwise));
                    });
                    ctx.ins(Air::IfElse {
                        condition: Reg::A,
                        then,
                        otherwise,
                    });
                    ctx.set_active_block(otherwise);
                }
                _ => unimplemented!(),
            }
        });
    });
}

fn eval_expr<'a>(ctx: &mut AirCtx<'a>, expr: &'a Expr) {
    match &expr {
        Expr::IndexOf(_) => todo!(),
        Expr::Access(_) | Expr::Str(_) | Expr::Lit(_) | Expr::Bool(_) | Expr::Ident(_) => {}
        Expr::Bin(bin) => {
            eval_bin_op(ctx, bin);
        }
        Expr::Struct(def) => {
            let dst = OffsetVar::zero(ctx.anon_var(ctx.tys.struct_ty_id(def.id)));
            define_struct(ctx, def, dst);
        }
        Expr::Call(call) => {
            ctx.push_pop_sp(|ctx| {
                let args = generate_args(ctx, call);
                ctx.ins(Air::Call(&call.sig, args));
            });
        }
        Expr::Enum(_) => {
            todo!()
        }
        Expr::If(if_) => eval_if(ctx, if_),
        Expr::Block(_) => todo!(),
        Expr::Loop(loop_) => {
            ctx.in_var_scope(|ctx| {
                let sp = OffsetVar::zero(ctx.anon_var(TyId::USIZE));
                ctx.ins(Air::ReadSP(sp));

                let exit = ctx.new_block();
                let loop_block = ctx.in_loop(exit, |ctx, loop_block| {
                    air_block(ctx, &loop_.block);
                    ctx.ins(Air::WriteSP(sp));
                    ctx.ins(Air::Jmp(loop_block));
                });
                ctx.ins(Air::Jmp(loop_block));
                ctx.set_active_block(exit);

                ctx.ins(Air::WriteSP(sp));
            });
        }
        Expr::For(for_) => {
            match for_.iterable {
                // TODO: check for when start is LARGER than end
                Expr::Range(range) => {
                    ctx.in_var_scope(|ctx| {
                        ctx.push_pop_sp(|ctx| {
                            let ty = ctx.var_ty(&for_.iter);
                            let anon_it = OffsetVar::zero(ctx.anon_var(ty));
                            assign_expr(ctx, anon_it, ty, range.start.unwrap());
                            let it = OffsetVar::zero(ctx.new_var_registered(&for_.iter, ty));
                            assign_var_other(ctx, it, anon_it, ty);

                            let width = ctx.tys.ty(ty).expect_int().width();

                            let end = OffsetVar::zero(ctx.anon_var(ty));
                            assign_expr(ctx, end, ty, range.end.unwrap());
                            if range.inclusive {
                                add!(ctx, width, end, end, 1);
                            }

                            let continue_ = OffsetVar::zero(ctx.anon_var(TyId::BOOL));

                            let exit = ctx.new_block();
                            let loop_block = ctx.in_loop(exit, |ctx, loop_block| {
                                let post_condition = ctx.in_scope(|ctx, _| {
                                    // reset iter to anon_it
                                    assign_var_other(ctx, it, anon_it, ty);

                                    // perform user code
                                    air_block(ctx, &for_.block);

                                    // add assign anon_it
                                    add!(ctx, width, anon_it, anon_it, 1);
                                    ctx.ins(Air::Jmp(loop_block));
                                });

                                // break if end is met
                                eq!(ctx, width, continue_, anon_it, end);
                                ctx.ins(Air::IfElse {
                                    condition: Reg::A,
                                    then: exit,
                                    otherwise: post_condition,
                                });
                            });
                            ctx.ins(Air::Jmp(loop_block));
                            ctx.set_active_block(exit);
                        });
                    });
                }
                Expr::Ident(ident) => {
                    ctx.in_var_scope(|ctx| {
                        ctx.push_pop_sp(|ctx| {
                            // load address of the array into `arr_ptr`
                            let arr_var = OffsetVar::zero(ctx.expect_var(ident.id));
                            let arr_ptr = OffsetVar::zero(ctx.anon_var(TyId::ANON_PTR));
                            ctx.ins_set([
                                Air::Addr(Reg::A, arr_var),
                                Air::PushIReg {
                                    dst: arr_ptr,
                                    width: Width::PTR,
                                    src: Reg::A,
                                },
                            ]);

                            let (len, iter_ty, elem_size) = match ctx.tys.ty(ctx.var_ty(ident)) {
                                Ty::Array(len, inner) => {
                                    let inner_ty_ref = ctx.tys.ty_id(&Ty::Ref(&inner));
                                    (len, inner_ty_ref, inner.size(ctx))
                                }
                                _ => panic!("invalid iterable type"),
                            };

                            let it = OffsetVar::zero(ctx.new_var_registered(&for_.iter, iter_ty));
                            assign_var_other(ctx, it, arr_ptr, iter_ty);

                            let it_count = OffsetVar::zero(ctx.anon_var(TyId::USIZE));
                            ctx.ins(Air::PushIConst(
                                it_count,
                                ConstData::Bits(Bits::from_u64(0)),
                            ));
                            let continue_ = OffsetVar::zero(ctx.anon_var(TyId::BOOL));

                            let exit = ctx.new_block();
                            let loop_block = ctx.in_loop(exit, |ctx, loop_block| {
                                let post_condition = ctx.in_scope(|ctx, _| {
                                    assign_var_other(ctx, it, arr_ptr, iter_ty);

                                    air_block(ctx, &for_.block);

                                    add!(ctx, Width::SIZE, it_count, it_count, 1);
                                    add!(ctx, Width::SIZE, arr_ptr, arr_ptr, elem_size as u64);
                                    ctx.ins(Air::Jmp(loop_block));
                                });

                                // break if end is met
                                eq!(ctx, Width::SIZE, continue_, it_count, len as u64);
                                ctx.ins(Air::IfElse {
                                    condition: Reg::A,
                                    then: exit,
                                    otherwise: post_condition,
                                });
                            });
                            ctx.ins(Air::Jmp(loop_block));
                            ctx.set_active_block(exit);
                        });
                    });
                }
                _ => unreachable!(),
            }
        }
        Expr::Break(_) => {
            let break_block = ctx.break_block().unwrap();
            ctx.ins(Air::Jmp(break_block));
        }
        Expr::Continue(_) => {
            let loop_start = ctx.loop_start().unwrap();
            ctx.ins(Air::Jmp(loop_start));
        }
        Expr::Array(_) => todo!(),
        Expr::Unary(_) => todo!(),
        Expr::Range(_) => panic!("range should not be evaluated!"),
    }
}

fn generate_args<'a>(ctx: &mut AirCtx<'a>, call: &'a Call) -> Args {
    if ctx.expect_ident(call.sig.ident) == "printf" {
        return printf_generate_args(ctx, call);
    }

    assert_eq!(call.args.len(), call.sig.params.len());
    if call.args.is_empty() {
        return Args::default();
    }

    let mut args = Args {
        vars: Vec::with_capacity(call.sig.params.len()),
    };

    for (expr, param) in call.args.iter().zip(call.sig.params.iter()) {
        let var = ctx.func_arg_var(param.ident, param.ty);
        ctx.ins(Air::SAlloc(var, ctx.tys.ty(param.ty).size(ctx)));
        let the_fn_param = OffsetVar::zero(var);
        assign_expr(ctx, the_fn_param, param.ty, expr);
        args.vars.push((param.ty, the_fn_param.var));
    }

    args
}

fn printf_generate_args<'a>(ctx: &mut AirCtx<'a>, call: &'a Call) -> Args {
    assert!(!call.args.is_empty());

    let mut args = Args {
        vars: Vec::with_capacity(call.args.len()),
    };

    for expr in call.args.iter() {
        let ty = match expr.infer(ctx) {
            InferTy::Int => ctx.tys.builtin(Ty::Int(IntTy::ISIZE)),
            InferTy::Float => ctx.tys.builtin(Ty::Float(FloatTy::F64)),
            InferTy::Ty(ty) => ty,
        };

        let the_fn_param = OffsetVar::zero(ctx.anon_var(ty));
        assign_expr(ctx, the_fn_param, ty, expr);
        args.vars.push((ty, the_fn_param.var));
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
    match &stmt.lhs {
        Expr::Ident(ident) => {
            assign_stmt_with_var_and_ty(
                ctx,
                stmt,
                OffsetVar::zero(ctx.expect_var(ident.id)),
                ctx.var_ty(ident),
            );
        }
        Expr::Access(access) => {
            let (var, ty) = aquire_accessor_field(ctx, access);
            assign_stmt_with_var_and_ty(ctx, stmt, var, ty);
        }
        Expr::IndexOf(index) => {
            let addr_reg = Reg::A;
            let inner = load_addr_index_of(ctx, index, addr_reg);
            let addr = OffsetVar::zero(ctx.anon_var(ctx.tys.builtin(Ty::Int(IntTy::PTR))));
            ctx.ins(Air::PushIReg {
                dst: addr,
                width: Width::PTR,
                src: addr_reg,
            });

            // TODO: run time bounds check
            match stmt.kind {
                AssignKind::Equals => match ctx.tys.ty(inner) {
                    Ty::Bool => {
                        assign_expr_deref(ctx, addr, inner, &stmt.rhs);
                    }
                    _ => todo!(),
                },
                _ => todo!(),
            }
        }
        assign => todo!("assign target: {assign:#?}"),
    };
}

fn assign_stmt_with_var_and_ty<'a>(
    ctx: &mut AirCtx<'a>,
    stmt: &'a Assign,
    var: OffsetVar,
    ty: TyId,
) {
    match stmt.kind {
        AssignKind::Equals => {
            assign_expr(ctx, var, ty, &stmt.rhs);
        }
        AssignKind::Add | AssignKind::Sub => {
            let other = OffsetVar::zero(ctx.anon_var(ty));
            assign_expr(ctx, other, ty, &stmt.rhs);
            match ctx.tys.ty(ty) {
                Ty::Int(int_ty) => {
                    let width = int_ty.width();
                    match stmt.kind {
                        AssignKind::Add => {
                            add!(ctx, width, var, var, other);
                        }
                        AssignKind::Sub => {
                            sub!(ctx, width, var, var, other);
                        }
                        AssignKind::Equals => unreachable!(),
                    }
                }
                Ty::Float(float_ty) => {
                    let width = float_ty.width();
                    match stmt.kind {
                        AssignKind::Add => {
                            fadd!(ctx, width, var, var, other);
                        }
                        AssignKind::Sub => {
                            fsub!(ctx, width, var, var, other);
                        }
                        AssignKind::Equals => unreachable!(),
                    }
                }
                _ => unreachable!(),
            }
        }
    }
}

#[track_caller]
fn assign_lit(ctx: &mut AirCtx, lit: &Lit, var: OffsetVar, ty: TyId) {
    match ctx.tys.ty(ty) {
        Ty::Int(int_ty) => match lit.kind {
            LitKind::Int(int) => {
                let width = int_ty.width();
                ctx.ins(Air::PushIConst(
                    var,
                    ConstData::Bits(Bits::from_width(*int as u64, width)),
                ));
            }
            LitKind::Float(_) => panic!("invalid op"),
        },
        Ty::Float(float_ty) => match lit.kind {
            LitKind::Float(float) => {
                ctx.ins(Air::PushIConst(
                    var,
                    ConstData::Bits(Bits::from_width_float(*float, float_ty.width())),
                ));
            }
            LitKind::Int(_) => panic!("invalid op"),
        },
        Ty::Ref(_) => match lit.kind {
            LitKind::Int(int) => {
                ctx.ins(Air::PushIConst(
                    var,
                    ConstData::Bits(Bits::from_u64(*int as u64)),
                ));
            }
            LitKind::Float(_) => panic!("invalid op"),
        },
        Ty::Array(_, _) | Ty::Bool | Ty::Unit | Ty::Str | Ty::Struct(_) => {
            panic!("cannot assign lit to {ty:?}")
        }
    }
}

// TODO: return to optimized return stratedgy
fn air_return<'a>(ctx: &mut AirCtx<'a>, ty: TyId, end: &'a Expr) {
    let dst = OffsetVar::zero(ctx.anon_var(ty));
    assign_expr(ctx, dst, ty, end);
    ctx.ret_var(dst, ty);

    //match end {
    //    Expr::Bool(_, val) => ctx.ret_iconst(*val as i64),
    //    Expr::Lit(lit) => match lit.kind {
    //        LitKind::Int(int) => {
    //            ctx.ret_iconst(*int);
    //        }
    //        _ => unreachable!(),
    //    },
    //    Expr::Ident(ident) => {
    //        let var = OffsetVar::zero(ctx.expect_var(ident.id));
    //        let out_ty = ctx.expect_var_ty(var.var);
    //        assert_eq!(out_ty, ty);
    //
    //        match ctx.tys.ty(out_ty) {
    //            Ty::Int(int) => {
    //                ctx.ret_ivar(var, int.kind());
    //            }
    //            Ty::Struct(_) => ctx.ret_ptr(var),
    //            Ty::Bool => ctx.ret_ivar(var, IntKind::I8),
    //            Ty::Unit => todo!(),
    //        }
    //    }
    //    Expr::Bin(bin) => {
    //        if bin.kind.is_primitive() {
    //            match ctx.tys.ty(ty) {
    //                Ty::Int(int) => {
    //                    let kind = int.kind();
    //                    let dst = OffsetVar::zero(ctx.anon_var(ty));
    //                    assign_prim_bin_op(ctx, dst, kind, bin);
    //                    ctx.ret_ivar(dst, kind);
    //                }
    //                Ty::Bool => {
    //                    panic!("invalid operation");
    //                    //let kind = IntKind::I8;
    //                    //let dst = OffsetVar::zero(ctx.anon_var(ty));
    //                    //assign_prim_bin_op(ctx, dst, kind, bin);
    //                    //ctx.ret_ivar(dst, kind);
    //                }
    //                Ty::Struct(_) | Ty::Unit => unreachable!(),
    //            }
    //        } else {
    //            match bin.kind {
    //                BinOpKind::Field => {
    //                    let (field_var, field_ty) = aquire_bin_field_offset(ctx, bin);
    //                    assert_eq!(ty, field_ty);
    //                    match ctx.tys.ty(field_ty) {
    //                        Ty::Int(int) => {
    //                            ctx.ret_ivar(field_var, int.kind());
    //                        }
    //                        Ty::Struct(_) => {
    //                            ctx.ret_ptr(field_var);
    //                        }
    //                        Ty::Bool => ctx.ret_ivar(field_var, IntKind::I8),
    //                        Ty::Unit => unreachable!(),
    //                    }
    //                }
    //                k => unreachable!("{k:?}"),
    //            }
    //        }
    //    }
    //    Expr::Call(call) => {
    //        assert_eq!(call.sig.ty, ty);
    //        let args = generate_args(ctx, call);
    //        ctx.ins(Air::Call(&call. args));
    //        ctx.ins(Air::Ret);
    //    }
    //    Expr::Struct(def) => {
    //        let dst = OffsetVar::zero(ctx.anon_var(ctx.tys.struct_ty_id(def.id)));
    //        define_struct(ctx, def, dst);
    //        ctx.ret_ptr(dst);
    //    }
    //    Expr::If(if_) => {
    //        let bool_ = ctx.tys.bool();
    //        let condition = ctx.anon_var(bool_);
    //        assign_expr(ctx, OffsetVar::zero(condition), bool_, if_.condition);
    //
    //        todo!()
    //    }
    //    Expr::Block(_) => todo!(),
    //    Expr::Enum(_) => todo!(),
    //}
}
