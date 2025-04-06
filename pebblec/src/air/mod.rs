use self::data::Bss;
use crate::ir::ctx::CtxFmt;
use crate::ir::ident::IdentId;
use crate::ir::lit::{Lit, LitKind};
use crate::ir::sig::{Param, Sig};
use crate::ir::strukt::StructDef;
use crate::ir::ty::store::TyStore;
use crate::ir::ty::{FloatTy, IntTy, Sign, Ty, TyKind, Width};
use crate::ir::*;
use bin::*;
use ctx::*;
use data::BssEntry;
use indexmap::IndexMap;
use pebblec_arena::BlobArena;
use pebblec_parse::rules::prelude::Attr;
use pebblec_parse::{AssignKind, UOpKind};
use std::collections::HashMap;
use std::ops::Range;

mod bin;
pub mod ctx;
pub mod data;

/// Analyzed Intermediate Representation.
///
/// `Air` is a collection of low level instructions that are intended to be easily executable as
/// byte-code and lowerable in a backend.
///
/// TODO: break out IntKind in favor of byte slices with sign extension?
#[derive(Debug, PartialEq)]
pub enum Air<'a> {
    Ret,

    Call(&'a AirSig<'a>, Args),

    /// Swap the A and B registers.
    SwapReg,
    MovIVar(Reg, OffsetVar, Width),
    MovIConst(Reg, ConstData),

    SAlloc(Var, usize),

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
    /// Point `dst` to the address in `addr`.
    Deref {
        dst: OffsetVar,
        addr: Reg,
    },

    /// Binary operations use [`Reg::A`] and [`Reg::B`], then store result in [`Reg::A`].
    ///
    /// TODO: need sign and overflow settings
    MulAB(Width, Sign),
    DivAB(Width, Sign),
    RemAB(Width, Sign),

    AddAB(Width, Sign),
    SubAB(Width, Sign),

    ShlAB(Width, Sign),
    ShrAB(Width, Sign),

    BandAB(Width),
    XorAB(Width),
    BorAB(Width),

    EqAB(Width, Sign),
    NEqAB(Width, Sign),
    LtAB(Width, Sign),
    GtAB(Width, Sign),
    LeAB(Width, Sign),
    GeAB(Width, Sign),

    FMulAB(Width),
    FDivAB(Width),
    FRemAB(Width),

    FAddAB(Width),
    FSubAB(Width),

    FEqAB(Width),
    NFEqAB(Width),
    FLtAB(Width),
    FGtAB(Width),
    FLeAB(Width),
    FGeAB(Width),

    CastA {
        from: (Prim, Width),
        to: (Prim, Width),
    },

    FSqrt(FloatTy),

    /// Exit with code stored in [`Reg::A`].
    Exit,
    /// The address of `fmt` should be loaded into [`Reg::A`].
    PrintCStr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Prim {
    UInt,
    Int,
    Float,
    Bool,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OffsetVar {
    pub var: Var,
    pub offset: usize,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[derive(Debug, PartialEq)]
pub enum ConstData {
    Bits(Bits),
    Ptr(BssEntry),
}

/// Collection of [`Air`] instructions for a [`crate::ir::Func`].
#[derive(Debug, PartialEq)]
pub struct AirFunc<'a> {
    pub sig: &'a AirSig<'a>,
    instrs: Vec<Air<'a>>,
    blocks: IndexMap<BlockId, Range<usize>>,
}

impl<'a> AirFunc<'a> {
    pub fn new(sig: &'a AirSig<'a>, blocks: Vec<Vec<Air<'a>>>) -> Self {
        let mut instrs = Vec::new();
        let mut ranges = IndexMap::with_capacity(blocks.len());
        for (hash, block_instrs) in blocks.into_iter().enumerate() {
            let start = instrs.len();
            instrs.extend(block_instrs);
            let end = instrs.len();
            ranges.insert(BlockId(hash), start..end);
        }

        Self {
            blocks: ranges,
            sig,
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
pub struct AirSig<'a> {
    pub ident: &'a str,
    pub ty: Ty,
    pub params: &'a [Ty],
    pub linkage: AirLinkage<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AirLinkage<'a> {
    Local,
    External { link: &'a str },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(usize);

#[derive(Debug, PartialEq)]
pub struct AirFuncBuilder<'a, 'ctx> {
    pub func: &'ctx Func<'ctx>,
    instrs: Vec<Vec<Air<'a>>>,
    active: BlockId,
    loop_ctx: Option<LoopCtx>,
}

impl<'a, 'ctx> AirFuncBuilder<'a, 'ctx> {
    pub fn new(func: &'ctx Func<'ctx>) -> Self {
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
    pub fn build(
        &mut self,
        sigs: &IndexMap<IdentId, &'a AirSig<'a>>,
        impl_sigs: &IndexMap<(Ty, IdentId), &'a AirSig<'a>>,
    ) -> AirFunc<'a> {
        assert!(!self.instrs.is_empty());
        let sig = self.func.sig;
        let air_sig = match sig.method_self {
            Some(ty) => impl_sigs.get(&(ty, sig.ident)).unwrap(),
            None => sigs.get(&sig.ident).unwrap(),
        };

        AirFunc::new(air_sig, std::mem::take(&mut self.instrs))
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Args {
    pub vars: Vec<(Ty, Var)>,
}

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
    pub const BOOL: Self = IntTy::BOOL.kind();
    pub const PTR: Self = IntTy::PTR.kind();
    pub const USIZE: Self = IntTy::USIZE.kind();

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
    pub const fn kind(&self) -> IntKind {
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

impl Expr<'_> {
    pub fn infer(&self, ctx: &mut AirCtx) -> InferTy {
        match self {
            Self::Lit(lit) => match lit.kind {
                LitKind::Int(_) => InferTy::Int,
                LitKind::Float(_) => InferTy::Float,
            },
            Self::Ident(ident) => {
                // instead of `var_ty` with the ident, we use this to avoid grabbing the most
                // recent type definition of ident, because it may be shadowed within the same line
                //
                // e.g.
                //
                // let var = 10;
                // let var = returns_different_type(var) + 2;
                //     ^^^                          ^^^
                //      |                           `var_ty` would grab the type defined here
                //      |                                                                 |
                //      |-----------------------------------------------------------------|
                InferTy::Ty(ctx.expect_var_ty(ctx.expect_var(ident.id)))
            }
            Self::Access(access) => InferTy::Ty(aquire_access_ty(ctx, access)),
            Self::Call(call) => InferTy::Ty(call.sig.ty),
            Self::Str(_) => InferTy::Ty(Ty::STR_LIT),
            Self::Bin(bin) => {
                let lhs = bin.lhs.infer(ctx);
                let rhs = bin.lhs.infer(ctx);
                // I know this is not right, but I am too lazy
                assert_eq!(lhs, rhs);
                if bin.kind.output_is_input() {
                    lhs
                } else {
                    InferTy::Ty(Ty::BOOL)
                }
            }
            Self::Bool(_) => InferTy::Ty(Ty::BOOL),
            Self::IndexOf(index) => match index.array.infer(ctx) {
                InferTy::Ty(arr_ty) => match arr_ty.0 {
                    TyKind::Array(_, inner) => InferTy::Ty(Ty(*inner)),
                    TyKind::Ref(TyKind::Slice(inner)) => InferTy::Ty(Ty(inner)),
                    other => panic!("invalid array type for index of: {:?}", other),
                },
                other => panic!("invalid array type for index of: {:?}", other),
            },
            Self::Cast(cast) => InferTy::Ty(cast.ty),
            Self::Unary(unary) => match unary.kind {
                UOpKind::Deref => match unary.inner.infer(ctx) {
                    InferTy::Ty(ty) => match ty.0 {
                        TyKind::Ref(inner) => InferTy::Ty(Ty(*inner)),
                        _ => unreachable!(),
                    },
                    InferTy::Int | InferTy::Float => unreachable!(),
                },
                UOpKind::Ref => match unary.inner.infer(ctx) {
                    InferTy::Ty(ty) => InferTy::Ty(ctx.tys.intern_kind(TyKind::Ref(ty.0))),
                    InferTy::Int | InferTy::Float => {
                        todo!("how do you describe a reference to these?")
                    }
                },
                UOpKind::Not | UOpKind::Neg => unary.inner.infer(ctx),
            },
            Self::MethodCall(call) => InferTy::Ty(call.expect_sig(ctx).ty),
            Self::Struct(def) => InferTy::Ty(def.ty),
            Self::Enum(_) => unimplemented!(),
            Self::Block(block) => block
                .end
                .map(|e| e.infer(ctx))
                .unwrap_or_else(|| InferTy::Ty(Ty::UNIT)),
            Self::If(if_) => if_.block.infer(ctx),
            Self::Loop(_) | Self::For(_) => InferTy::Ty(Ty::UNIT),
            Self::Break(_) | Self::Continue(_) => unreachable!(),
            // TODO: there is array and range left, but they are never called? Why not?
            ty => todo!("infer: {ty:?}"),
        }
    }

    pub fn infer_abs(&self, ctx: &mut AirCtx) -> Option<Ty> {
        match self.infer(ctx) {
            InferTy::Float | InferTy::Int => None,
            InferTy::Ty(ty) => Some(ty),
        }
    }
}

fn aquire_access_ty(ctx: &mut AirCtx, access: &Access) -> Ty {
    let ty = access.lhs.infer_abs(ctx).unwrap();
    let id = match ty.0 {
        TyKind::Struct(id) => id,
        TyKind::Array(_, _)
        | TyKind::Slice(_)
        | TyKind::Int(_)
        | TyKind::Unit
        | TyKind::Bool
        | TyKind::Ref(_)
        | TyKind::Str
        | TyKind::Float(_) => {
            unreachable!()
        }
    };
    let mut strukt = ctx.tys.strukt(*id);

    for (i, acc) in access.accessors.iter().rev().enumerate() {
        let ty = strukt.field_ty(acc.id);
        if i == access.accessors.len() - 1 {
            return ty;
        }

        match ty.0 {
            TyKind::Struct(id) => {
                strukt = ctx.tys.strukt(*id);
            }
            TyKind::Array(_, _)
            | TyKind::Slice(_)
            | TyKind::Int(_)
            | TyKind::Unit
            | TyKind::Bool
            | TyKind::Ref(_)
            | TyKind::Str
            | TyKind::Float(_) => {
                unreachable!()
            }
        }
    }
    unreachable!()
}

pub fn lower_const<'a, 'ctx>(ctx: &mut AirCtx<'a, 'ctx>, konst: &Const) -> Vec<Air<'a>> {
    ctx.start_const();
    let dst = OffsetVar::zero(ctx.new_var_registered(&konst.name, konst.ty));
    assign_expr(ctx, dst, konst.ty, &konst.expr);
    ctx.finish_const()
}

#[derive(Debug)]
pub struct ByteCode<'a> {
    pub bss: Bss,
    pub tys: TyStore,
    pub extern_sigs: HashMap<&'a str, &'a AirSig<'a>>,
    pub funcs: Vec<AirFunc<'a>>,
    pub consts: Vec<Air<'a>>,
    _storage: BlobArena,
}

pub fn lower<'a, 'ctx>(mut ir: Ir<'ctx>) -> ByteCode<'a> {
    let tys = std::mem::take(&mut ir.ctx.tys);
    let mut air_ctx = AirCtx::new(&ir.ctx, ir.key, tys);
    let consts = ir
        .const_eval_order
        .into_iter()
        .flat_map(|id| lower_const(&mut air_ctx, ir.ctx.get_const(id).unwrap()))
        .collect::<Vec<_>>();
    let funcs = ir
        .ctx
        .funcs
        .iter()
        .map(|func| lower_func(&mut air_ctx, func))
        .collect::<Vec<_>>();
    let extern_sigs = air_ctx
        .air_sigs
        .iter()
        .filter_map(|(ident, sig)| {
            matches!(sig.linkage, AirLinkage::External { .. })
                .then_some((air_ctx.storage.alloc_str(ir.ctx.expect_ident(*ident)), *sig))
        })
        .collect();
    let tys = std::mem::take(&mut air_ctx.tys);
    let (storage, bss) = air_ctx.into_inner();

    ByteCode {
        bss,
        tys,
        extern_sigs,
        funcs,
        consts,
        _storage: storage,
    }
}

pub fn lower_func<'a, 'ctx>(ctx: &mut AirCtx<'a, 'ctx>, func: &'ctx Func) -> AirFunc<'a> {
    if func.has_attr(Attr::Intrinsic) {
        return lower_intrinsic(ctx, func);
    }

    ctx.in_var_scope(|ctx| {
        ctx.start_func(func);
        init_params(ctx, func);

        if func.sig.ty.is_unit() {
            air_block(ctx, &func.block);
            if ctx.expect_ident(func.sig.ident) == "main" {
                ctx.ins(Air::MovIConst(Reg::A, ConstData::Bits(Bits::from_u64(0))));
            }
            ctx.ins(Air::Ret);
        } else {
            let dst = OffsetVar::zero(ctx.anon_var(func.sig.ty));
            assign_air_block(ctx, dst, func.sig.ty, &func.block);
            ctx.ret_var(dst, func.sig.ty);
        }
    });
    ctx.finish_func()
}

pub fn lower_intrinsic<'a, 'ctx>(ctx: &mut AirCtx<'a, 'ctx>, func: &'ctx Func) -> AirFunc<'a> {
    ctx.in_var_scope(|ctx| {
        init_params(ctx, func);
        match ctx.expect_ident(func.sig.ident) {
            "exit" => exit(ctx, func),
            "print" => print(ctx, func),
            "println" => println(ctx, func),
            "cs" => c_str(ctx, func),
            "print_cs" => print_c_str(ctx, func),
            "sqrt_f32" => sqrt_f32(ctx, func),
            i => unimplemented!("intrinsic: {i}"),
        }
    })
}

pub fn exit<'a, 'ctx>(ctx: &mut AirCtx<'a, 'ctx>, func: &'ctx Func) -> AirFunc<'a> {
    ctx.start_func(func);
    let Param::Named { ident, .. } = func.sig.params.iter().next().unwrap() else {
        unreachable!()
    };

    let var = ctx.expect_var(ident.id);
    ctx.ins_set([
        Air::MovIVar(Reg::A, OffsetVar::zero(var), Width::W32),
        Air::Exit,
    ]);
    ctx.finish_func()
}

pub fn c_str<'a, 'ctx>(ctx: &mut AirCtx<'a, 'ctx>, func: &'ctx Func) -> AirFunc<'a> {
    ctx.start_func(func);
    let Param::Named { ident, .. } = func.sig.params.iter().next().unwrap() else {
        unreachable!()
    };

    let var = ctx.expect_var(ident.id);
    ctx.ins_set([
        Air::Addr(Reg::A, OffsetVar::new(var, IntKind::USIZE)),
        Air::Ret,
    ]);
    ctx.finish_func()
}

pub fn print_c_str<'a, 'ctx>(ctx: &mut AirCtx<'a, 'ctx>, func: &'ctx Func) -> AirFunc<'a> {
    ctx.start_func(func);
    let Param::Named { ident, .. } = func.sig.params.iter().next().unwrap() else {
        unreachable!()
    };

    let var = ctx.expect_var(ident.id);
    ctx.ins_set([
        Air::MovIVar(Reg::A, OffsetVar::zero(var), Width::PTR),
        Air::PrintCStr,
        Air::Ret,
    ]);
    ctx.finish_func()
}

pub fn print<'a, 'ctx>(ctx: &mut AirCtx<'a, 'ctx>, func: &'ctx Func) -> AirFunc<'a> {
    assert_eq!(func.sig.params.len(), 1);
    ctx.start_func(func);
    ctx.ins_set([Air::Ret]);
    ctx.finish_func()
}

pub fn println<'a, 'ctx>(ctx: &mut AirCtx<'a, 'ctx>, func: &'ctx Func) -> AirFunc<'a> {
    assert_eq!(func.sig.params.len(), 1);
    ctx.start_func(func);
    ctx.ins_set([Air::Ret]);
    ctx.finish_func()
}

pub fn sqrt_f32<'a, 'ctx>(ctx: &mut AirCtx<'a, 'ctx>, func: &'ctx Func) -> AirFunc<'a> {
    ctx.start_func(func);
    let Param::Named { ident, .. } = func.sig.params.iter().next().unwrap() else {
        unreachable!()
    };

    let var = ctx.expect_var(ident.id);
    ctx.ins_set([
        Air::MovIVar(Reg::A, OffsetVar::zero(var), Width::W32),
        Air::FSqrt(FloatTy::F32),
        Air::Ret,
    ]);
    ctx.finish_func()
}

fn init_params(ctx: &mut AirCtx, func: &Func) {
    for param in func.sig.params.iter() {
        match &param {
            Param::Named { ident, ty, .. } => {
                ctx.func_arg_var(*ident, *ty);
            }
            Param::Slf(ident) => {
                let ref_ty = ctx
                    .tys
                    .intern_kind(TyKind::Ref(func.sig.method_self.unwrap().0));
                ctx.func_arg_var(*ident, ref_ty);
            }
        }
    }
}

fn air_block(ctx: &mut AirCtx, block: &Block) {
    block_stmts(ctx, block.stmts);
    match block.end {
        Some(end) => {
            // TODO: ensure that this is always unit
            eval_expr(ctx, end);
        }
        None => {}
    }
}

fn assign_air_block(ctx: &mut AirCtx, dst: OffsetVar, ty: Ty, block: &Block) {
    block_stmts(ctx, block.stmts);
    if let Some(end) = &block.end {
        assign_expr(ctx, dst, ty, end);
    } else {
        // TODO: need analysis of return statements
        //println!("{block:?}");
        //assert!(ty == ctx.tys.unit());
    }
}

fn block_stmts(ctx: &mut AirCtx, stmts: &[Stmt]) {
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

fn air_let_stmt(ctx: &mut AirCtx, stmt: &Let) {
    match &stmt.lhs {
        LetTarget::Ident(ident) => {
            let ty = ctx.var_ty(ident);
            let dst = ctx.anon_var(ty);
            assign_expr(ctx, OffsetVar::zero(dst), ty, &stmt.rhs);
            // defer registering so that something in stmt.rhs isn't shadowed
            ctx.register_var(ident, dst);
        }
    }
}

fn load_addr_index_of(ctx: &mut AirCtx, index: &IndexOf, dst: Reg) -> Ty {
    let arr_ty = index.array.infer_abs(ctx).unwrap();
    let ty = match arr_ty.0 {
        TyKind::Array(_, inner) => inner,
        // TODO: run time bounds checking
        TyKind::Ref(TyKind::Slice(inner)) => inner,
        _ => unreachable!(),
    };

    let index_ty = Ty::USIZE;
    let index_var = OffsetVar::zero(ctx.anon_var(index_ty));
    assign_expr(ctx, index_var, index_ty, index.index);
    ctx.ins_set([
        Air::MovIVar(Reg::A, index_var, Width::SIZE),
        Air::MovIConst(
            Reg::B,
            ConstData::Bits(Bits::from_u64(ty.size(&ctx.tys) as u64)),
        ),
        Air::MulAB(Width::SIZE, Sign::U),
        Air::PushIReg {
            dst: index_var,
            width: Width::SIZE,
            src: Reg::A,
        },
    ]);

    let addr_reg = Reg::A;
    let ptr_offset_reg = Reg::B;

    let var = extract_var_from_expr(ctx, arr_ty, index.array);
    match arr_ty.0 {
        TyKind::Array(_, _) => {
            ctx.ins(Air::Addr(addr_reg, var));
        }
        TyKind::Ref(TyKind::Slice(_)) => {
            ctx.ins(Air::MovIVar(addr_reg, var, Width::PTR));
        }
        _ => unreachable!(),
    }

    ctx.ins_set([
        Air::MovIVar(ptr_offset_reg, index_var, Width::PTR),
        // the address of the array is in `A` and the offset is in `B`
        Air::AddAB(Width::PTR, Sign::U),
    ]);

    if dst == Reg::B {
        ctx.ins(Air::SwapReg);
    } else {
        assert_eq!(dst, Reg::A);
    }

    Ty(ty)
}

fn assign_expr(ctx: &mut AirCtx, dst: OffsetVar, ty: Ty, expr: &Expr) {
    match &expr {
        Expr::IndexOf(index) => {
            load_addr_index_of(ctx, index, Reg::A);

            let bytes = ty.size(&ctx.tys);
            match bytes {
                1 | 2 | 4 | 8 => {
                    let width = match bytes {
                        1 => Width::W8,
                        2 => Width::W16,
                        4 => Width::W32,
                        8 => Width::W64,
                        _ => unreachable!(),
                    };
                    ctx.ins_set([
                        Air::Read {
                            dst: Reg::A,
                            addr: Reg::A,
                            width,
                        },
                        Air::PushIReg {
                            dst,
                            width,
                            src: Reg::A,
                        },
                    ]);
                }
                _ => ctx.ins_set([
                    Air::Addr(Reg::B, dst),
                    Air::MemCpy {
                        dst: Reg::B,
                        src: Reg::A,
                        bytes,
                    },
                ]),
            }
        }
        Expr::Array(arr) => match ty.0 {
            TyKind::Array(len, inner) => match arr {
                ArrDef::Elems { exprs, .. } => {
                    assert_eq!(*len, exprs.len());
                    for (i, expr) in exprs.iter().enumerate() {
                        assign_expr(ctx, dst.add(i * inner.size(&ctx.tys)), Ty(inner), expr);
                    }
                }
                ArrDef::Repeated { expr, .. } => {
                    // TODO: assert the length is equal to the type
                    //
                    //assert_eq!(*num, len);
                    let inner = Ty(*inner);
                    let the_expr = OffsetVar::zero(ctx.anon_var(inner));
                    assign_expr(ctx, the_expr, inner, expr);
                    for i in 0..*len {
                        assign_var_other(ctx, dst.add(i * inner.size(&ctx.tys)), the_expr, inner);
                    }
                }
            },
            _ => unreachable!(),
        },
        Expr::Bool(bool) => {
            assert_eq!(ty, Ty::BOOL);
            ctx.ins(Air::PushIConst(
                dst,
                ConstData::Bits(if bool.val { Bits::TRUE } else { Bits::FALSE }),
            ));
        }
        Expr::Str(str) => {
            assert_eq!(ty, Ty::STR_LIT);
            let (entry, len) = ctx.str_lit(str.val);
            ctx.ins_set([
                Air::PushIConst(dst, ConstData::Ptr(entry)),
                Air::PushIConst(
                    dst.add(IntKind::U64),
                    ConstData::Bits(Bits::from_u64(len as u64)),
                ),
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
                let args = generate_args(ctx, call.sig, call.args);
                ctx.call(call.sig, args);
            });
            if !ty.is_unit() {
                extract_return_from_a(ctx, dst, ty);
            }
        }
        Expr::MethodCall(call) => {
            let sig = call.expect_sig(ctx);
            assert_eq!(ty, sig.ty);

            ctx.push_pop_sp(|ctx| {
                let args = match call.receiver {
                    MethodPath::Field(expr) => generate_method_args(ctx, sig, expr, call.args),
                    MethodPath::Path(_, _) => generate_args(ctx, sig, call.args),
                };

                let ty = call.expect_ty(ctx);
                ctx.method_call(sig, ty, args);
            });
            if !ty.is_unit() {
                extract_return_from_a(ctx, dst, ty);
            }
        }
        Expr::Ident(ident) => {
            let other = OffsetVar::zero(ctx.expect_var(ident.id));
            assert!(
                ctx.expect_var_ty(other.var).equiv(*ty.0),
                "`{:?}` != `{:?}`",
                ctx.expect_var_ty(other.var),
                ty.0,
            );
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
        Expr::Cast(cast) => {
            let infer = cast.lhs.infer_abs(ctx).unwrap();
            assert!(infer.is_castable());

            let var = extract_var_from_expr(ctx, infer, cast.lhs);
            assert!(ty.is_castable());
            assert_eq!(ty, cast.ty);

            let (from, width) = match infer.0 {
                TyKind::Int(int) => match int.sign() {
                    Sign::I => (Prim::Int, int.width()),
                    Sign::U => (Prim::UInt, int.width()),
                },
                TyKind::Float(float) => (Prim::Float, float.width()),
                TyKind::Bool => (Prim::Bool, Width::BOOL),
                TyKind::Ref(_) => {
                    //assert!(matches!(ty, Ty::PTR));
                    (Prim::UInt, Width::PTR)
                }
                _ => unreachable!(),
            };

            let (to, to_width) = match ty.0 {
                TyKind::Int(int) => match int.sign() {
                    Sign::I => (Prim::Int, int.width()),
                    Sign::U => (Prim::UInt, int.width()),
                },
                TyKind::Float(float) => (Prim::Float, float.width()),
                TyKind::Bool => (Prim::Bool, Width::BOOL),
                TyKind::Ref(TyKind::Str) => unreachable!(),
                TyKind::Ref(_) => {
                    assert_eq!(width, Width::PTR);
                    assert_eq!(from, Prim::UInt);
                    (Prim::UInt, Width::PTR)
                }
                _ => unreachable!(),
            };

            ctx.ins_set([
                Air::MovIVar(Reg::A, var, width),
                Air::CastA {
                    from: (from, width),
                    to: (to, to_width),
                },
                Air::PushIReg {
                    dst,
                    width: to_width,
                    src: Reg::A,
                },
            ]);
        }
        Expr::Unary(unary) => match unary.kind {
            UOpKind::Ref => match ty.0 {
                TyKind::Ref(inner) => {
                    if inner.is_slice() {
                        let ty = ctx.tys.intern_kind(TyKind::Ref(inner));
                        take_arr_ref(ctx, ty, dst, unary.inner);
                    } else {
                        let var = extract_var_from_expr(ctx, Ty(inner), unary.inner);
                        ctx.ins_set([
                            Air::Addr(Reg::A, var),
                            Air::PushIReg {
                                dst,
                                width: Width::PTR,
                                src: Reg::A,
                            },
                        ]);
                    }
                }
                _ => unreachable!(),
            },
            UOpKind::Deref => {
                match unary.inner.infer(ctx) {
                    InferTy::Int | InferTy::Float => panic!(),
                    InferTy::Ty(ty) => assert!(ty.is_ref()),
                }

                let ref_ty = ctx.tys.intern_kind(TyKind::Ref(ty.0));
                let ptr_var = extract_var_from_expr(ctx, ref_ty, unary.inner);
                let addr_reg = Reg::A;
                ctx.ins_set([
                    Air::MovIVar(addr_reg, ptr_var, Width::PTR),
                    Air::Deref {
                        dst,
                        addr: addr_reg,
                    },
                ]);
            }
            UOpKind::Not => {
                let (width, mask) = match ty.0 {
                    TyKind::Bool => (Width::BOOL, 1),
                    TyKind::Int(ty) => (ty.width(), u64::MAX),
                    _ => unreachable!(),
                };
                let result = OffsetVar::zero(ctx.anon_var(ty));
                assign_expr(ctx, result, ty, unary.inner);
                ctx.ins_set([
                    Air::MovIVar(Reg::A, result, width),
                    Air::MovIConst(Reg::B, ConstData::Bits(Bits::from_u64(mask))),
                    Air::XorAB(width),
                    Air::PushIReg {
                        dst,
                        width,
                        src: Reg::A,
                    },
                ]);
            }
            UOpKind::Neg => {
                let result = OffsetVar::zero(ctx.anon_var(ty));
                assign_expr(ctx, result, ty, unary.inner);

                match ty.0 {
                    TyKind::Int(int_ty) => {
                        let width = int_ty.width();
                        let mask = u64::MAX;
                        ctx.ins_set([
                            Air::MovIVar(Reg::A, result, width),
                            Air::MovIConst(Reg::B, ConstData::Bits(Bits::from_u64(mask))),
                            Air::XorAB(width),
                            Air::MovIConst(Reg::B, ConstData::Bits(Bits::from_u64(1))),
                            Air::AddAB(width, Sign::I),
                            Air::PushIReg {
                                dst,
                                width,
                                src: Reg::A,
                            },
                        ]);
                    }
                    TyKind::Float(float_ty) => {
                        let width = float_ty.width();
                        let mask = if *float_ty == FloatTy::F32 {
                            0x80000000
                        } else {
                            0x8000000000000000
                        };

                        ctx.ins_set([
                            Air::MovIVar(Reg::A, result, width),
                            Air::MovIConst(Reg::B, ConstData::Bits(Bits::from_u64(mask))),
                            Air::XorAB(width),
                            Air::PushIReg {
                                dst,
                                width,
                                src: Reg::A,
                            },
                        ]);
                    }
                    _ => unreachable!(),
                }
            }
        },
        Expr::Enum(_) | Expr::Block(_) => todo!(),
        Expr::Continue(_)
        | Expr::Break(_)
        | Expr::Range(_)
        | Expr::For(_)
        | Expr::Loop(_)
        | Expr::While(_) => unreachable!(),
    }
}

impl MethodCall<'_> {
    pub fn expect_ty(&self, ctx: &mut AirCtx) -> Ty {
        match self.receiver {
            MethodPath::Field(expr) => expr.infer_abs(ctx).unwrap(),
            MethodPath::Path(_, ty) => ty,
        }
    }

    pub fn expect_sig<'a, 'ctx>(&self, ctx: &mut AirCtx<'a, 'ctx>) -> &'ctx Sig<'ctx> {
        let ty = self.expect_ty(ctx);
        ctx.get_method_sig(ty, self.call.id).unwrap()
    }
}

#[track_caller]
fn extract_return_from_a(ctx: &mut AirCtx, dst: OffsetVar, ty: Ty) {
    match ty.0 {
        TyKind::Int(ty) => {
            ctx.ins(Air::PushIReg {
                dst,
                width: ty.width(),
                src: Reg::A,
            });
        }
        TyKind::Float(ty) => {
            ctx.ins(Air::PushIReg {
                dst,
                width: ty.width(),
                src: Reg::A,
            });
        }
        TyKind::Ref(TyKind::Str) | TyKind::Slice(_) => {
            ctx.ins_set([
                Air::Deref { dst, addr: Reg::A },
                Air::MovIConst(
                    Reg::B,
                    ConstData::Bits(Bits::from_u64(Width::SIZE.size() as u64)),
                ),
                Air::AddAB(Width::SIZE, Sign::U),
                Air::Deref {
                    dst: dst.add(Width::SIZE),
                    addr: Reg::A,
                },
            ]);
        }
        TyKind::Ref(_) => {
            ctx.ins(Air::PushIReg {
                dst,
                width: Width::PTR,
                src: Reg::A,
            });
        }
        TyKind::Array(len, inner) => {
            let bytes = inner.size(&ctx.tys) * len;
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
        TyKind::Struct(s) => {
            let bytes = ctx.tys.struct_layout(*s).size;
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
        TyKind::Bool => {
            ctx.ins(Air::PushIReg {
                dst,
                width: Width::BOOL,
                src: Reg::A,
            });
        }
        _ => panic!("invalid return type: {:?}", ty.0),
    }
}

fn take_arr_ref(ctx: &mut AirCtx, ty: Ty, dst: OffsetVar, expr: &Expr) {
    match expr {
        Expr::Ident(ident) => {
            let var = ctx.expect_var(ident.id);
            let var_ty = ctx.expect_var_ty(var);

            match (var_ty.0, ty.0) {
                (TyKind::Array(len, lhs), TyKind::Ref(TyKind::Slice(rhs))) => {
                    assert_eq!(lhs, rhs);
                    ctx.ins_set([
                        Air::Addr(Reg::A, OffsetVar::zero(var)),
                        Air::PushIReg {
                            dst,
                            width: Width::PTR,
                            src: Reg::A,
                        },
                        Air::PushIConst(
                            dst.add(Width::PTR),
                            ConstData::Bits(Bits::from_u64(*len as u64)),
                        ),
                    ]);
                }
                _ => unreachable!(),
            }
        }
        _ => todo!(),
    }
}

fn extract_var_from_expr(ctx: &mut AirCtx, ty: Ty, expr: &Expr) -> OffsetVar {
    match expr {
        Expr::Ident(ident) => {
            let var = ctx.expect_var(ident.id);
            let var_ty = ctx.expect_var_ty(var);

            assert_eq!(
                var_ty,
                ty,
                "{} != {} @ {}",
                var_ty.to_string(ctx),
                ty.to_string(ctx),
                ctx.expect_ident(ident.id)
            );

            OffsetVar::zero(var)
        }
        Expr::Access(access) => aquire_accessor_field(ctx, access).0,
        Expr::IndexOf(index) => {
            let addr_reg = Reg::A;
            let ty = load_addr_index_of(ctx, index, addr_reg);
            let var = OffsetVar::zero(ctx.anon_var_no_salloc(ty));
            ctx.ins(Air::Deref {
                dst: var,
                addr: addr_reg,
            });

            var
        }
        Expr::Unary(unary) if unary.kind == UOpKind::Deref => {
            let addr_reg = Reg::A;
            let inner_ty = ctx.tys.intern_kind(TyKind::Ref(ty.0));
            let inner = extract_var_from_expr(ctx, inner_ty, unary.inner);
            let var = OffsetVar::zero(ctx.anon_var_no_salloc(ty));
            ctx.ins_set([
                Air::MovIVar(addr_reg, inner, Width::PTR),
                Air::Deref {
                    dst: var,
                    addr: addr_reg,
                },
            ]);

            var
        }
        Expr::Break(_) | Expr::Continue(_) | Expr::Loop(_) | Expr::For(_) => unreachable!(),
        _ => {
            let dst = OffsetVar::zero(ctx.anon_var(ty));
            assign_expr(ctx, dst, ty, expr);
            dst
        }
    }
}

fn assign_var_other(ctx: &mut AirCtx, dst: OffsetVar, other: OffsetVar, ty: Ty) {
    match ty.0 {
        TyKind::Int(ty) => {
            ctx.ins(Air::PushIVar {
                dst,
                width: ty.width(),
                src: other,
            });
        }
        TyKind::Float(ty) => {
            ctx.ins_set([Air::PushIVar {
                dst,
                width: ty.width(),
                src: other,
            }]);
        }
        TyKind::Ref(TyKind::Str) => {
            ctx.ins_set([
                Air::PushIVar {
                    dst,
                    width: Width::PTR,
                    src: other,
                },
                Air::PushIVar {
                    dst: dst.add(Width::SIZE),
                    width: Width::SIZE,
                    src: other.add(Width::SIZE),
                },
            ]);
        }
        TyKind::Ref(_) => {
            ctx.ins(Air::PushIVar {
                dst,
                width: Width::PTR,
                src: other,
            });
        }
        TyKind::Struct(id) => {
            let bytes = ctx.tys.struct_layout(*id).size;
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
        TyKind::Bool => {
            ctx.ins(Air::PushIVar {
                dst,
                width: Width::BOOL,
                src: other,
            });
        }
        TyKind::Array(len, inner) => {
            ctx.ins_set([
                Air::Addr(Reg::B, other),
                Air::Addr(Reg::A, dst),
                Air::MemCpy {
                    dst: Reg::A,
                    src: Reg::B,
                    bytes: inner.size(&ctx.tys) * len,
                },
            ]);
        }
        TyKind::Slice(_) => {
            ctx.ins_set([
                Air::PushIVar {
                    dst,
                    width: Width::PTR,
                    src: other,
                },
                Air::PushIVar {
                    dst: dst.add(Width::SIZE),
                    width: Width::SIZE,
                    src: other.add(Width::SIZE),
                },
            ]);
        }
        TyKind::Str => panic!("cannot assign to str"),
        TyKind::Unit => todo!(),
    }
}

fn eval_if(ctx: &mut AirCtx, if_: &If) {
    eval_or_assign_if(ctx, if_, None);
}

fn assign_if(ctx: &mut AirCtx, dst: OffsetVar, ty: Ty, if_: &If) {
    eval_or_assign_if(ctx, if_, Some((dst, ty)));
}

fn eval_or_assign_if(ctx: &mut AirCtx, if_: &If, dst: Option<(OffsetVar, Ty)>) {
    ctx.in_var_scope(|ctx| {
        let condition = OffsetVar::zero(ctx.anon_var(Ty::BOOL));

        ctx.push_pop_sp(|ctx| {
            assign_expr(ctx, condition, Ty::BOOL, if_.condition);
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

// TODO: perhaps unify eval and assign entirely? Creating unnecessary anon_var for expressions that
// cannot return a type does incur overhead in the bytecode, but like, not a lot?
fn eval_expr(ctx: &mut AirCtx, expr: &Expr) {
    match &expr {
        Expr::If(if_) => {
            eval_if(ctx, if_);
        }
        Expr::Loop(loop_) => {
            ctx.in_var_scope(|ctx| {
                let sp = OffsetVar::zero(ctx.anon_var(Ty::USIZE));
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
        Expr::While(while_) => {
            ctx.in_var_scope(|ctx| {
                let condition = OffsetVar::zero(ctx.anon_var(Ty::BOOL));

                ctx.push_pop_sp(|ctx| {
                    let exit = ctx.new_block();
                    let loop_block = ctx.in_loop(exit, |ctx, loop_block| {
                        let post_condition = ctx.in_scope(|ctx, _| {
                            air_block(ctx, &while_.block);
                            ctx.ins(Air::Jmp(loop_block));
                        });

                        assign_expr(ctx, condition, Ty::BOOL, while_.condition);
                        ctx.ins(Air::MovIVar(Reg::A, condition, Width::BOOL));
                        ctx.ins(Air::IfElse {
                            condition: Reg::A,
                            then: post_condition,
                            otherwise: exit,
                        });
                    });
                    ctx.ins(Air::Jmp(loop_block));
                    ctx.set_active_block(exit);
                });
            });
        }
        Expr::For(for_) => {
            match for_.iterable {
                // TODO: check for when start is LARGER than end
                Expr::Range(range) => {
                    ctx.in_var_scope(|ctx| {
                        ctx.push_pop_sp(|ctx| {
                            let ty = ctx.var_ty(&for_.iter);
                            let sign = match ty.0 {
                                TyKind::Int(int) => int.sign(),
                                _ => unreachable!(),
                            };

                            let anon_it = OffsetVar::zero(ctx.anon_var(ty));
                            assign_expr(ctx, anon_it, ty, range.start.unwrap());
                            let it = OffsetVar::zero(ctx.new_var_registered(&for_.iter, ty));
                            assign_var_other(ctx, it, anon_it, ty);

                            let width = ty.expect_int().width();

                            let end = OffsetVar::zero(ctx.anon_var(ty));
                            assign_expr(ctx, end, ty, range.end.unwrap());
                            if range.inclusive {
                                add!(ctx, width, sign, end, end, 1);
                            }

                            let continue_ = OffsetVar::zero(ctx.anon_var(Ty::BOOL));

                            // TODO: this is pretty unecessary.
                            //
                            // The issue is that continue will jump to loop_block and skip adding
                            // to the anon_it if it is in post_condition, but you can't add 1 to
                            // anon_it before the first iteration, so... loop_block needs more
                            // context?
                            let add_factor = OffsetVar::zero(ctx.anon_var(ty));
                            ctx.ins(Air::PushIConst(
                                add_factor,
                                ConstData::Bits(Bits::from_u64(0)),
                            ));

                            let exit = ctx.new_block();
                            let loop_block = ctx.in_loop(exit, |ctx, loop_block| {
                                let post_condition = ctx.in_scope(|ctx, _| {
                                    // reset iter to anon_it
                                    assign_var_other(ctx, it, anon_it, ty);

                                    // perform user code
                                    air_block(ctx, &for_.block);

                                    ctx.ins(Air::Jmp(loop_block));
                                });

                                // add assign anon_it
                                add!(ctx, width, sign, anon_it, anon_it, add_factor);
                                ctx.ins(Air::PushIConst(
                                    add_factor,
                                    ConstData::Bits(Bits::from_u64(1)),
                                ));

                                // break if end is met
                                ge!(ctx, width, sign, continue_, anon_it, end);
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
                expr => {
                    ctx.in_var_scope(|ctx| {
                        ctx.push_pop_sp(|ctx| {
                            // load address of the array into `arr_ptr`
                            let arr_ty = expr.infer_abs(ctx).unwrap();
                            let arr_var = extract_var_from_expr(ctx, arr_ty, expr);
                            let arr_ptr = OffsetVar::zero(ctx.anon_var(Ty::ANON_PTR));

                            match arr_ty.0 {
                                TyKind::Array(_, _) => {
                                    ctx.ins_set([
                                        Air::Addr(Reg::A, arr_var),
                                        Air::PushIReg {
                                            dst: arr_ptr,
                                            width: Width::PTR,
                                            src: Reg::A,
                                        },
                                    ]);
                                }
                                TyKind::Ref(TyKind::Slice(_)) => {
                                    ctx.ins(Air::PushIVar {
                                        dst: arr_ptr,
                                        width: Width::PTR,
                                        src: arr_var,
                                    });
                                }
                                _ => unreachable!(),
                            }

                            let (len, iter_ty, elem_size) = match arr_ty.0 {
                                TyKind::Array(len, inner) => {
                                    let inner_ty_ref = ctx.tys.intern_kind(TyKind::Ref(inner));
                                    let len_var = OffsetVar::zero(ctx.anon_var(Ty::USIZE));
                                    ctx.ins(Air::PushIConst(
                                        len_var,
                                        ConstData::Bits(Bits::from_u64(*len as u64)),
                                    ));
                                    (len_var, inner_ty_ref, inner.size(&ctx.tys))
                                }
                                TyKind::Ref(TyKind::Slice(inner)) => {
                                    let inner_ty_ref = ctx.tys.intern_kind(TyKind::Ref(inner));
                                    let len = OffsetVar::zero(ctx.anon_var(Ty::USIZE));
                                    ctx.ins(Air::PushIVar {
                                        dst: len,
                                        width: Width::SIZE,
                                        src: arr_var.add(Width::PTR),
                                    });
                                    (len, inner_ty_ref, inner.size(&ctx.tys))
                                }
                                _ => panic!("invalid iterable type"),
                            };

                            let it = OffsetVar::zero(ctx.new_var_registered(&for_.iter, iter_ty));
                            assign_var_other(ctx, it, arr_ptr, iter_ty);

                            let it_count = OffsetVar::zero(ctx.anon_var(Ty::USIZE));
                            ctx.ins(Air::PushIConst(
                                it_count,
                                ConstData::Bits(Bits::from_u64(0)),
                            ));

                            let continue_ = OffsetVar::zero(ctx.anon_var(Ty::BOOL));

                            // TODO: these are both unnecessary, see above
                            let add_factor = OffsetVar::zero(ctx.anon_var(iter_ty));
                            ctx.ins(Air::PushIConst(
                                add_factor,
                                ConstData::Bits(Bits::from_u64(0)),
                            ));
                            let elem_offset = OffsetVar::zero(ctx.anon_var(Ty::PTR));
                            ctx.ins(Air::PushIConst(
                                elem_offset,
                                ConstData::Bits(Bits::from_u64(0)),
                            ));

                            let exit = ctx.new_block();
                            let loop_block = ctx.in_loop(exit, |ctx, loop_block| {
                                let post_condition = ctx.in_scope(|ctx, _| {
                                    assign_var_other(ctx, it, arr_ptr, iter_ty);

                                    air_block(ctx, &for_.block);

                                    ctx.ins(Air::Jmp(loop_block));
                                });

                                add!(ctx, Width::SIZE, Sign::U, it_count, it_count, add_factor);
                                ctx.ins(Air::PushIConst(
                                    add_factor,
                                    ConstData::Bits(Bits::from_u64(1)),
                                ));
                                add!(ctx, Width::SIZE, Sign::U, arr_ptr, arr_ptr, elem_offset);
                                ctx.ins(Air::PushIConst(
                                    elem_offset,
                                    ConstData::Bits(Bits::from_u64(elem_size as u64)),
                                ));

                                // break if end is met
                                ge!(ctx, Width::SIZE, Sign::U, continue_, it_count, len);
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
        _ => {
            let ty = expr.infer_abs(ctx).unwrap();
            let dst = OffsetVar::zero(ctx.anon_var(ty));
            assign_expr(ctx, dst, ty, expr);
        }
    }
}

fn generate_args(ctx: &mut AirCtx, sig: &Sig, args: &[Expr]) -> Args {
    let ident = ctx.expect_ident(sig.ident);
    if ident == "print" || ident == "println" {
        return print_generate_args(ctx, args);
    }

    assert_eq!(args.len(), sig.params.len());
    if args.is_empty() {
        return Args::default();
    }

    let mut call_args = Args {
        vars: Vec::with_capacity(sig.params.len()),
    };

    for (expr, param) in args.iter().zip(sig.params.iter()) {
        let (ident, ty) = match &param {
            Param::Named { ident, ty, .. } => (*ident, *ty),
            param => panic!("invalid param: {param:#?}"),
        };

        let var = ctx.func_arg_var(ident, ty);
        ctx.ins(Air::SAlloc(var, ty.size(&ctx.tys)));
        let the_fn_param = OffsetVar::zero(var);
        assign_expr(ctx, the_fn_param, ty, expr);
        call_args.vars.push((ty, the_fn_param.var));
    }

    call_args
}

fn print_generate_args(ctx: &mut AirCtx, args: &[Expr]) -> Args {
    assert!(!args.is_empty());

    let mut call_args = Args {
        vars: Vec::with_capacity(args.len()),
    };

    for expr in args.iter() {
        let ty = match expr.infer(ctx) {
            InferTy::Int => ctx.tys.intern_kind(TyKind::Int(IntTy::ISIZE)),
            InferTy::Float => ctx.tys.intern_kind(TyKind::Float(FloatTy::F64)),
            InferTy::Ty(ty) => ty,
        };

        let the_fn_param = OffsetVar::zero(ctx.anon_var(ty));
        assign_expr(ctx, the_fn_param, ty, expr);
        call_args.vars.push((ty, the_fn_param.var));
    }

    call_args
}

fn generate_method_args(ctx: &mut AirCtx, sig: &Sig, callee: &Expr, args: &[Expr]) -> Args {
    assert_eq!(args.len(), sig.params.len().saturating_sub(1));

    let mut call_args = Args {
        vars: Vec::with_capacity(sig.params.len() + 1),
    };

    for (expr, param) in std::iter::once(callee)
        .chain(args.iter())
        .zip(sig.params.iter())
    {
        match &param {
            Param::Named { ident, ty, .. } => {
                let var = ctx.func_arg_var(*ident, *ty);
                ctx.ins(Air::SAlloc(var, ty.size(&ctx.tys)));
                let the_fn_param = OffsetVar::zero(var);
                assign_expr(ctx, the_fn_param, *ty, expr);
                call_args.vars.push((*ty, the_fn_param.var));
            }
            Param::Slf(ident) => {
                let self_ty = ctx.tys.intern_kind(TyKind::Ref(sig.method_self.unwrap().0));
                let var = ctx.func_arg_var(*ident, self_ty);
                let expr_ty = expr.infer_abs(ctx).unwrap();
                let expr_var = extract_var_from_expr(ctx, expr_ty, expr);
                ctx.ins_set([
                    Air::SAlloc(var, self_ty.size(&ctx.tys)),
                    Air::Addr(Reg::A, expr_var),
                    Air::PushIReg {
                        dst: OffsetVar::zero(var),
                        width: Width::PTR,
                        src: Reg::A,
                    },
                ]);

                call_args.vars.push((self_ty, var));
            }
        };
    }

    call_args
}

fn define_struct(ctx: &mut AirCtx, def: &StructDef, dst: OffsetVar) {
    for field in def.fields.iter() {
        let strukt = ctx.tys.strukt(def.id);
        let field_offset = strukt.field_offset(&ctx.tys, field.name.id);
        let ty = strukt.field_ty(field.name.id);

        assign_expr(
            ctx,
            OffsetVar::new(dst.var, dst.offset + field_offset as usize),
            ty,
            &field.expr,
        );
    }
}

fn air_assign_stmt(ctx: &mut AirCtx, assign: &Assign) {
    let ty = assign.lhs.infer_abs(ctx).unwrap();
    let var = extract_var_from_expr(ctx, ty, &assign.lhs);
    assign_stmt_with_var_and_ty(ctx, assign, var, ty);
}

fn assign_stmt_with_var_and_ty(ctx: &mut AirCtx, stmt: &Assign, var: OffsetVar, ty: Ty) {
    match stmt.kind {
        AssignKind::Equals => {
            assign_expr(ctx, var, ty, &stmt.rhs);
        }
        _ => {
            let other = OffsetVar::zero(ctx.anon_var(ty));
            assign_expr(ctx, other, ty, &stmt.rhs);
            match ty.0 {
                TyKind::Int(int_ty) => {
                    let width = int_ty.width();
                    let sign = int_ty.sign();
                    match stmt.kind {
                        AssignKind::Mul => {
                            mul!(ctx, width, sign, var, var, other);
                        }
                        AssignKind::Div => {
                            div!(ctx, width, sign, var, var, other);
                        }
                        AssignKind::Rem => {
                            rem!(ctx, width, sign, var, var, other);
                        }

                        AssignKind::Add => {
                            add!(ctx, width, sign, var, var, other);
                        }
                        AssignKind::Sub => {
                            sub!(ctx, width, sign, var, var, other);
                        }

                        AssignKind::Xor => {
                            xor!(ctx, width, var, var, other);
                        }
                        AssignKind::And => {
                            and!(ctx, width, var, var, other);
                        }
                        AssignKind::Or => {
                            or!(ctx, width, var, var, other);
                        }

                        AssignKind::Shl => {
                            shl!(ctx, width, sign, var, var, other);
                        }
                        AssignKind::Shr => {
                            shr!(ctx, width, sign, var, var, other);
                        }

                        AssignKind::Equals => unreachable!(),
                    }
                }
                TyKind::Float(float_ty) => {
                    let width = float_ty.width();
                    match stmt.kind {
                        AssignKind::Add => {
                            fadd!(ctx, width, var, var, other);
                        }
                        AssignKind::Sub => {
                            fsub!(ctx, width, var, var, other);
                        }
                        AssignKind::Mul => {
                            fmul!(ctx, width, var, var, other);
                        }
                        AssignKind::Div => {
                            fdiv!(ctx, width, var, var, other);
                        }
                        AssignKind::Rem => {
                            frem!(ctx, width, var, var, other);
                        }
                        _ => unreachable!(),
                    }
                }
                _ => unreachable!(),
            }
        }
    }
}

#[track_caller]
fn assign_lit(ctx: &mut AirCtx, lit: &Lit, var: OffsetVar, ty: Ty) {
    match ty.0 {
        TyKind::Int(int_ty) => match lit.kind {
            LitKind::Int(int) => {
                let width = int_ty.width();
                ctx.ins(Air::PushIConst(
                    var,
                    ConstData::Bits(Bits::from_width(*int as u64, width)),
                ));
            }
            LitKind::Float(_) => panic!("invalid op"),
        },
        TyKind::Float(float_ty) => match lit.kind {
            LitKind::Float(float) => {
                ctx.ins(Air::PushIConst(
                    var,
                    ConstData::Bits(Bits::from_width_float(*float, float_ty.width())),
                ));
            }
            LitKind::Int(_) => panic!("invalid op"),
        },
        TyKind::Ref(_) => match lit.kind {
            LitKind::Int(int) => {
                ctx.ins(Air::PushIConst(
                    var,
                    ConstData::Bits(Bits::from_u64(*int as u64)),
                ));
            }
            LitKind::Float(_) => panic!("invalid op"),
        },
        TyKind::Array(_, _)
        | TyKind::Slice(_)
        | TyKind::Bool
        | TyKind::Unit
        | TyKind::Str
        | TyKind::Struct(_) => {
            panic!("cannot assign lit to {ty:?}")
        }
    }
}

// TODO: return to optimized return strategy
fn air_return(ctx: &mut AirCtx, ty: Ty, end: &Expr) {
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
