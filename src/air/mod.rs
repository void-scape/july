use crate::ir::lit::{Lit, LitKind};
use crate::ir::sig::{Param, Sig};
use crate::ir::strukt::StructDef;
use crate::ir::ty::store::TyId;
use crate::ir::ty::{IWidth, IntTy, Sign, Ty};
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
#[derive(Debug)]
pub enum Air<'a> {
    Ret,

    // TODO: recursion won't work with the current model, we simply override vars
    Call(&'a Sig<'a>, Args),

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

    IfElse {
        /// The evalauted boolean condition.
        condition: Reg,
        then: BlockId,
        otherwise: BlockId,
    },
    Jmp(BlockId),

    // TODO: rename Push
    PushIConst(OffsetVar, IntKind, ConstData),
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

    /// Binary operations use [`Reg::A`] and [`Reg::B`], then store result in [`Reg::A`].
    AddAB,
    MulAB,
    SubAB,
    EqAB,

    /// Exit with code stored in [`Reg::A`].
    Exit,
    /// The address of `fmt` should be loaded into [`Reg::A`]. The number of characters should be
    /// loaded into [`Reg::B`].
    Printf,
    /// The address of `fmt` should be loaded into [`Reg::A`].
    PrintCStr,
}

pub type Addr = u64;

#[derive(Debug)]
pub enum ConstData {
    Int(i64),
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

    #[track_caller]
    pub fn next_block(&self, block: BlockId) -> Option<&[Air<'a>]> {
        self.blocks
            .get(&BlockId(block.0 + 1))
            .map(|range| &self.instrs[range.start..range.end])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(usize);

#[derive(Debug)]
pub struct AirFuncBuilder<'a> {
    pub func: &'a Func<'a>,
    instrs: Vec<Vec<Air<'a>>>,
}

impl<'a> AirFuncBuilder<'a> {
    pub fn new(func: &'a Func<'a>) -> Self {
        Self {
            instrs: Vec::new(),
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
        self.instrs.last_mut().expect("no active block").push(instr);
    }

    #[track_caller]
    pub fn insert_active_set(&mut self, instrs: impl IntoIterator<Item = Air<'a>>) {
        self.instrs
            .last_mut()
            .expect("no active block")
            .extend(instrs);
    }

    #[track_caller]
    pub fn insert(&mut self, block: BlockId, instr: Air<'a>) {
        self.instrs
            .get_mut(block.0)
            .expect("invalid block id")
            .push(instr);
    }

    #[track_caller]
    pub fn build(self) -> AirFunc<'a> {
        assert!(!self.instrs.is_empty());
        AirFunc::new(self.func, self.instrs)
    }
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
    pub deref: bool,
}

impl OffsetVar {
    pub fn new(var: Var, offset: impl Sized) -> Self {
        Self {
            var,
            offset: offset.size(),
            deref: false,
        }
    }

    pub fn new_deref(var: Var, offset: impl Sized, deref: bool) -> Self {
        Self {
            var,
            offset: offset.size(),
            deref,
        }
    }

    pub fn zero(var: Var) -> Self {
        Self::new(var, 0)
    }

    pub fn zero_deref(var: Var) -> Self {
        Self::new_deref(var, 0, true)
    }

    pub fn add(&self, offset: impl Sized) -> Self {
        Self {
            var: self.var,
            offset: self.offset + offset.size(),
            deref: self.deref,
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

impl IntKind {
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
    if func.has_attr(Attr::Intrinsic) {
        return lower_intrinsic(ctx, func);
    }

    ctx.start(func);
    init_params(ctx, func);

    if ctx.tys.is_unit(func.sig.ty) {
        air_block(ctx, func.sig, &func.block);
        ctx.ins(Air::Ret);
    } else {
        let dst = OffsetVar::zero(ctx.anon_var(func.sig.ty));
        assign_air_block(ctx, &func.sig, dst, func.sig.ty, &func.block);
        ctx.ret_var(dst, func.sig.ty);
    }
    ctx.finish()
}

pub fn lower_intrinsic<'a, 'b>(ctx: &'b mut AirCtx<'a>, func: &'a Func) -> AirFunc<'a> {
    match ctx.expect_ident(func.sig.ident) {
        "exit" => exit(ctx, func),
        "printf" => printf(ctx, func),
        "c_str" => c_str(ctx, func),
        "print_c_str" => print_c_str(ctx, func),
        i => unimplemented!("intrinsic: {i}"),
    }
}

pub fn exit<'a, 'b>(ctx: &'b mut AirCtx<'a>, func: &'a Func) -> AirFunc<'a> {
    ctx.start(func);
    let Param { ident, ty, .. } = func.sig.params.iter().next().unwrap();
    if ctx.get_var(ident.id).is_none() {
        ctx.new_var_registered_no_salloc(ident.id, *ty);
    }

    let var = ctx.expect_var(ident.id);
    ctx.ins_set([
        Air::MovIVar(Reg::A, OffsetVar::zero(var), IntKind::I32),
        Air::Exit,
    ]);
    ctx.finish()
}

pub fn c_str<'a, 'b>(ctx: &'b mut AirCtx<'a>, func: &'a Func) -> AirFunc<'a> {
    ctx.start(func);
    let Param { ident, ty, .. } = func.sig.params.iter().next().unwrap();
    if ctx.get_var(ident.id).is_none() {
        ctx.new_var_registered_no_salloc(ident.id, *ty);
    }

    let var = ctx.expect_var(ident.id);
    ctx.ins_set([
        Air::Addr(Reg::A, OffsetVar::new(var, IntKind::USIZE)),
        Air::Ret,
    ]);
    ctx.finish()
}

pub fn print_c_str<'a, 'b>(ctx: &'b mut AirCtx<'a>, func: &'a Func) -> AirFunc<'a> {
    ctx.start(func);
    let Param { ident, ty, .. } = func.sig.params.iter().next().unwrap();
    if ctx.get_var(ident.id).is_none() {
        ctx.new_var_registered_no_salloc(ident.id, *ty);
    }

    let var = ctx.expect_var(ident.id);
    ctx.ins_set([
        Air::MovIVar(Reg::A, OffsetVar::zero(var), IntKind::PTR),
        Air::PrintCStr,
        Air::Ret,
    ]);
    ctx.finish()
}

pub fn printf<'a, 'b>(ctx: &'b mut AirCtx<'a>, func: &'a Func) -> AirFunc<'a> {
    assert_eq!(func.sig.params.len(), 1);
    ctx.start(func);
    let Param { ident, ty, .. } = func.sig.params.iter().next().unwrap();
    if ctx.get_var(ident.id).is_none() {
        ctx.new_var_registered_no_salloc(ident.id, *ty);
    }

    let var = ctx.expect_var(ident.id);
    ctx.ins_set([
        Air::MovIVar(Reg::A, OffsetVar::new(var, IntKind::U64), IntKind::PTR),
        Air::MovIVar(Reg::B, OffsetVar::zero(var), IntKind::U64),
        Air::Printf,
        Air::Ret,
    ]);
    ctx.finish()
}

fn init_params(ctx: &mut AirCtx, func: &Func) {
    for Param { ident, ty, .. } in func.sig.params.iter() {
        if ctx.get_var(ident.id).is_none() {
            ctx.new_var_registered_no_salloc(ident.id, *ty);
        }
    }
}

fn air_block<'a>(ctx: &mut AirCtx<'a>, sig: &'a Sig<'a>, block: &'a Block<'a>) {
    block_stmts(ctx, sig, block.stmts);
    match block.end {
        Some(end) => {
            // TODO: ensure that this is always unit
            eval_expr(ctx, sig, end);
        }
        None => {}
    }
}

fn assign_air_block<'a>(
    ctx: &mut AirCtx<'a>,
    sig: &'a Sig<'a>,
    dst: OffsetVar,
    ty: TyId,
    block: &'a Block<'a>,
) {
    block_stmts(ctx, sig, block.stmts);
    if let Some(end) = &block.end {
        assign_expr(ctx, sig, dst, ty, end);
    } else {
        // TODO: need analysis of return statements
        //println!("{block:?}");
        //assert!(ty == ctx.tys.unit());
    }
}

fn block_stmts<'a>(ctx: &mut AirCtx<'a>, sig: &'a Sig<'a>, stmts: &'a [Stmt<'a>]) {
    for stmt in stmts.iter() {
        match stmt {
            Stmt::Semi(stmt) => match stmt {
                SemiStmt::Let(let_) => air_let_stmt(ctx, sig, let_),
                SemiStmt::Assign(assign) => air_assign_stmt(ctx, sig, assign),
                SemiStmt::Ret(ret) => match &ret.expr {
                    Some(expr) => {
                        air_return(ctx, sig, sig.ty, &expr);
                    }
                    None => ctx.ins(Air::Ret),
                },
                SemiStmt::Expr(expr) => {
                    eval_expr(ctx, sig, expr);
                }
            },
            Stmt::Open(expr) => eval_expr(ctx, sig, expr),
        }
    }
}

fn air_bin_semi<'a>(ctx: &mut AirCtx<'a>, bin: &'a BinOp) {
    air_bin_semi_expr(ctx, &bin.lhs);
    air_bin_semi_expr(ctx, &bin.rhs);
}

fn air_bin_semi_expr<'a>(ctx: &mut AirCtx<'a>, expr: &'a Expr) {
    match &expr {
        Expr::Bin(bin) => air_bin_semi(ctx, bin),
        Expr::Call(call @ Call { sig, .. }) => {
            let args = generate_args(ctx, sig, call);
            ctx.call(sig, args)
        }
        Expr::Lit(_) | Expr::Ident(_) => {}
        _ => todo!(),
    }
}

fn air_let_stmt<'a>(ctx: &mut AirCtx<'a>, sig: &'a Sig<'a>, stmt: &'a Let) {
    match stmt.lhs {
        LetTarget::Ident(ident) => {
            let ty = ctx.var_ty(ident.id);
            let dst = ctx.new_var_registered(ident.id, ty);
            assign_expr(ctx, sig, OffsetVar::zero(dst), ty, &stmt.rhs);
        }
    }
}

fn assign_expr<'a>(
    ctx: &mut AirCtx<'a>,
    sig: &'a Sig<'a>,
    dst: OffsetVar,
    ty: TyId,
    expr: &'a Expr,
) {
    match &expr {
        Expr::Bool(_, val) => {
            assert_eq!(ty, TyId::BOOL);
            ctx.ins(Air::PushIConst(
                dst,
                IntKind::BOOL,
                ConstData::Int(if *val { 1 } else { 0 }),
            ));
        }
        Expr::Str(_, str) => {
            assert_eq!(ty, TyId::STR_LIT);
            let (entry, len) = ctx.str_lit(str);
            ctx.ins_set([
                Air::PushIConst(dst, IntKind::U64, ConstData::Int(len as i64)),
                Air::PushIConst(dst.add(IntKind::U64), IntKind::I64, ConstData::Ptr(entry)),
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
            define_struct(ctx, sig, def, dst);
        }
        Expr::Call(call) => {
            assert_eq!(ty, call.sig.ty);

            let args = generate_args(ctx, sig, call);
            ctx.ins(Air::Call(&call.sig, args));
            match ctx.tys.ty(call.sig.ty) {
                Ty::Int(ty) => {
                    ctx.ins(Air::PushIReg {
                        dst,
                        kind: ty.kind(),
                        src: Reg::A,
                    });
                }
                Ty::Ref(_) => {
                    ctx.ins(Air::PushIReg {
                        dst,
                        kind: IntKind::PTR,
                        src: Reg::A,
                    });
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
                        kind: IntKind::I8,
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
            assign_if(ctx, sig, dst, ty, if_);
        }
        Expr::Ref(ref_) => {
            assert!(ctx.tys.ty(ty).is_ref());
            match ref_.inner {
                Expr::Ident(ident) => {
                    let var = ctx.expect_var(ident.id);
                    ctx.ins_set([
                        Air::Addr(Reg::A, OffsetVar::zero(var)),
                        Air::PushIReg {
                            dst,
                            kind: IntKind::PTR,
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
                        ctx.ins_set([
                            Air::PushIConst(
                                var,
                                ctx.tys.ty(inner_ty).expect_int().kind(),
                                ConstData::Int(*int),
                            ),
                            Air::Addr(Reg::A, var),
                            Air::PushIReg {
                                dst,
                                kind: IntKind::PTR,
                                src: Reg::A,
                            },
                        ])
                    }
                },
                _ => todo!(),
            }
        }
        Expr::Loop(_) => {
            // this means that a loop is the last statement, and in order to leave we must return, so
            // just ignore this for now
            //assert_eq!(ty, TyId::UNIT);
            eval_expr(ctx, sig, expr);
        }
        Expr::Enum(_) | Expr::Block(_) => todo!(),

        Expr::Deref(_) => todo!(),
    }
}

fn assign_var_other<'a>(ctx: &mut AirCtx<'a>, dst: OffsetVar, other: OffsetVar, ty: TyId) {
    match ctx.tys.ty(ty) {
        Ty::Int(ty) => {
            ctx.ins(Air::PushIVar {
                dst,
                kind: ty.kind(),
                src: other,
            });
        }
        Ty::Ref(_) => {
            ctx.ins(Air::PushIVar {
                dst,
                kind: IntKind::PTR,
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
                kind: IntKind::I8,
                src: other,
            });
        }
        Ty::Str => {
            panic!("cannot assign to str");
        }
        Ty::Unit => todo!(),
    }
}

fn assign_if<'a>(ctx: &mut AirCtx<'a>, sig: &'a Sig<'a>, dst: OffsetVar, ty: TyId, if_: &If<'a>) {
    let bool_ = ctx.tys.bool();
    let condition = OffsetVar::zero(ctx.anon_var(bool_));

    assign_expr(ctx, sig, condition, bool_, if_.condition);
    ctx.ins(Air::MovIVar(Reg::A, condition, IntKind::BOOL));

    match (if_.block, if_.otherwise) {
        (Expr::Block(then), Some(Expr::Block(otherwise))) => {
            let enter_block = ctx.active_block();
            let then = ctx.in_new_block(|ctx, _| {
                assign_air_block(ctx, ctx.active_sig(), dst, ty, then);
            });
            let otherwise = ctx.in_new_block(|ctx, _| {
                assign_air_block(ctx, ctx.active_sig(), dst, ty, otherwise);
            });

            ctx.ins_in_block(
                enter_block,
                Air::IfElse {
                    condition: Reg::A,
                    then,
                    otherwise,
                },
            );

            let exit = ctx.new_block();
            ctx.ins_in_block(then, Air::Jmp(exit));
            ctx.ins_in_block(otherwise, Air::Jmp(exit));
        }
        (Expr::Block(then), None) => {
            let enter_block = ctx.active_block();
            let then = ctx.in_new_block(|ctx, _| {
                assign_air_block(ctx, ctx.active_sig(), dst, ty, then);
            });
            let otherwise = ctx.new_block();
            ctx.ins_in_block(then, Air::Jmp(otherwise));

            ctx.ins_in_block(
                enter_block,
                Air::IfElse {
                    condition: Reg::A,
                    then,
                    otherwise,
                },
            );
        }
        _ => unimplemented!(),
    }
}

fn eval_if<'a>(ctx: &mut AirCtx<'a>, sig: &'a Sig<'a>, if_: &If<'a>) {
    let bool_ = ctx.tys.bool();
    let condition = OffsetVar::zero(ctx.anon_var(bool_));

    assign_expr(ctx, sig, condition, bool_, if_.condition);
    ctx.ins(Air::MovIVar(Reg::A, condition, IntKind::BOOL));

    match (if_.block, if_.otherwise) {
        (Expr::Block(then), Some(Expr::Block(otherwise))) => {
            let enter_block = ctx.active_block();
            let then = ctx.in_new_block(|ctx, _| {
                air_block(ctx, ctx.active_sig(), then);
            });
            let otherwise = ctx.in_new_block(|ctx, _| {
                air_block(ctx, ctx.active_sig(), otherwise);
            });

            ctx.ins_in_block(
                enter_block,
                Air::IfElse {
                    condition: Reg::A,
                    then,
                    otherwise,
                },
            );

            let exit = ctx.new_block();
            ctx.ins_in_block(then, Air::Jmp(exit));
            ctx.ins_in_block(otherwise, Air::Jmp(exit));
        }
        (Expr::Block(then), None) => {
            let enter_block = ctx.active_block();
            let then = ctx.in_new_block(|ctx, _| {
                air_block(ctx, ctx.active_sig(), then);
            });
            let otherwise = ctx.new_block();

            ctx.ins_in_block(
                enter_block,
                Air::IfElse {
                    condition: Reg::A,
                    then,
                    otherwise,
                },
            );
            ctx.ins_in_block(then, Air::Jmp(otherwise));
        }
        _ => unimplemented!(),
    }
}

fn eval_expr<'a>(ctx: &mut AirCtx<'a>, sig: &'a Sig<'a>, expr: &'a Expr) {
    match &expr {
        Expr::Access(_) | Expr::Str(_, _) | Expr::Lit(_) | Expr::Bool(_, _) | Expr::Ident(_) => {}
        Expr::Bin(bin) => {
            eval_bin_op(ctx, sig, bin);
        }
        Expr::Struct(def) => {
            let dst = OffsetVar::zero(ctx.anon_var(ctx.tys.struct_ty_id(def.id)));
            define_struct(ctx, sig, def, dst);
        }
        Expr::Call(call) => {
            let args = generate_args(ctx, sig, call);
            ctx.ins(Air::Call(&call.sig, args));
        }
        Expr::Enum(_) => {
            todo!()
        }
        Expr::If(if_) => eval_if(ctx, sig, if_),
        Expr::Block(_) => todo!(),
        Expr::Loop(block) => {
            let loop_ = ctx.new_block();
            air_block(ctx, sig, block);
            // TODO: break;
            ctx.ins(Air::Jmp(loop_));
        }
        Expr::Ref(_) => todo!(),

        Expr::Deref(_) => todo!(),
    }
}

fn generate_args<'a>(ctx: &mut AirCtx<'a>, sig: &'a Sig<'a>, call: &'a Call) -> Args {
    assert_eq!(call.args.len(), call.sig.params.len());
    if call.args.is_empty() {
        return Args::default();
    }

    let mut args = Args {
        vars: Vec::with_capacity(call.sig.params.len()),
    };

    for (expr, param) in call.args.iter().zip(call.sig.params.iter()) {
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

        assign_expr(ctx, sig, the_fn_param, param.ty, expr);
        args.vars.push(the_fn_param);
    }

    args
}

fn define_struct<'a>(ctx: &mut AirCtx<'a>, sig: &'a Sig<'a>, def: &'a StructDef, dst: OffsetVar) {
    for field in def.fields.iter() {
        let strukt = ctx.tys.strukt(def.id);
        let field_offset = strukt.field_offset(ctx, field.name.id);
        let ty = strukt.field_ty(field.name.id);

        assign_expr(
            ctx,
            sig,
            OffsetVar::new(dst.var, dst.offset + field_offset as usize),
            ty,
            &field.expr,
        );
    }
}

fn air_assign_stmt<'a>(ctx: &mut AirCtx<'a>, sig: &'a Sig<'a>, stmt: &'a Assign) {
    let (var, ty) = match &stmt.lhs {
        AssignTarget::Ident(ident) => (
            OffsetVar::zero(ctx.expect_var(ident.id)),
            ctx.var_ty(ident.id),
        ),
        AssignTarget::Access(access) => aquire_accessor_field(ctx, access),
    };

    match stmt.kind {
        AssignKind::Equals => {
            assign_expr(ctx, sig, var, ty, &stmt.rhs);
        }
        AssignKind::Add => {
            let tmp = OffsetVar::zero(ctx.anon_var(ty));
            let kind = ctx.tys.ty(ty).expect_int().kind();
            ctx.ins(Air::PushIVar {
                dst: tmp,
                kind,
                src: var,
            });
            assign_expr(ctx, sig, var, ty, &stmt.rhs);
            add!(ctx, kind, var, tmp, var);
        }
    }
}

#[track_caller]
fn assign_lit(ctx: &mut AirCtx, lit: &Lit, var: OffsetVar, ty: TyId) {
    match ctx.tys.ty(ty) {
        Ty::Int(int_ty) => match lit.kind {
            LitKind::Int(int) => {
                ctx.ins(Air::PushIConst(var, int_ty.kind(), ConstData::Int(*int)));
            }
        },
        Ty::Ref(_) => match lit.kind {
            LitKind::Int(int) => {
                ctx.ins(Air::PushIConst(var, IntKind::PTR, ConstData::Int(*int)));
            }
        },
        Ty::Bool => panic!("invalid operation"),
        Ty::Unit => panic!("cannot assign lit to unit"),
        Ty::Str => panic!("cannot assign lit to str"),
        Ty::Struct(_) => panic!("cannot assign lit to struct"),
    }
}

// TODO: return to optimized return stratedgy
fn air_return<'a>(ctx: &mut AirCtx<'a>, sig: &'a Sig<'a>, ty: TyId, end: &'a Expr) {
    let dst = OffsetVar::zero(ctx.anon_var(ty));
    assign_expr(ctx, sig, dst, ty, end);
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
    //        ctx.ins(Air::Call(&call.sig, args));
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
