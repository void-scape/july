use super::data::{Bss, BssEntry};
use super::{Air, AirFunc, AirFuncBuilder, BlockId, OffsetVar, Reg, Var};
use crate::ir::ctx::Ctx;
use crate::ir::ident::{Ident, IdentId};
use crate::ir::sig::Sig;
use crate::ir::ty::infer::SymbolTable;
use crate::ir::ty::store::{TyId, TyStore};
use crate::ir::ty::{Ty, TypeKey, Width};
use crate::ir::{self, *};
use std::collections::HashMap;
use std::ops::Deref;

pub struct AirCtx<'a> {
    pub tys: TyStore<'a>,
    ctx: &'a Ctx<'a>,
    key: &'a TypeKey,
    var_index: usize,
    pub tables: Vec<SymbolTable<Var>>,
    func_args: HashMap<Ident, Var>,
    ty_map: HashMap<Var, TyId>,
    bss: Bss,
    func: Option<FuncHash>,
    instr_builder: InstrBuilder<'a>,
}

impl<'a> AirCtx<'a> {
    pub fn new(ctx: &'a Ctx<'a>, key: &'a TypeKey) -> Self {
        Self {
            ctx,
            key,
            tys: ctx.tys.clone(),
            var_index: 0,
            tables: Vec::new(),
            func_args: HashMap::default(),
            ty_map: HashMap::default(),
            bss: Bss::default(),
            func: None,
            instr_builder: InstrBuilder::Const(Vec::new()),
        }
    }

    #[track_caller]
    pub fn active_sig(&self) -> &'a Sig<'a> {
        match &self.instr_builder {
            InstrBuilder::Func(b) => b.func.sig,
            InstrBuilder::Const(_) => panic!("cannot call function in const setting"),
        }
    }

    pub fn start_func(&mut self, func: &'a ir::Func) -> BlockId {
        self.func = Some(func.hash());
        let mut builder = AirFuncBuilder::new(func);
        let id = builder.new_block();
        self.instr_builder = InstrBuilder::Func(builder);
        id
    }

    pub fn start_const(&mut self) {
        self.func = None;
        self.instr_builder = InstrBuilder::Const(Vec::new());
    }

    #[track_caller]
    pub fn finish_func(&mut self) -> AirFunc<'a> {
        match &mut self.instr_builder {
            InstrBuilder::Func(b) => b.build(),
            InstrBuilder::Const(_) => panic!("called `finish_func` with const builder"),
        }
    }

    #[track_caller]
    pub fn finish_const(&mut self) -> Vec<Air<'a>> {
        match &mut self.instr_builder {
            InstrBuilder::Const(instrs) => std::mem::take(instrs),
            InstrBuilder::Func(_) => panic!("called `finish_const` with func builder"),
        }
    }

    #[track_caller]
    fn expect_func_builder(&self) -> &AirFuncBuilder<'a> {
        match &self.instr_builder {
            InstrBuilder::Func(b) => b,
            InstrBuilder::Const(_) => panic!("called `expect_func_builder` with const builder"),
        }
    }

    #[track_caller]
    fn expect_func_builder_mut(&mut self) -> &mut AirFuncBuilder<'a> {
        match &mut self.instr_builder {
            InstrBuilder::Func(b) => b,
            InstrBuilder::Const(_) => panic!("called `expect_func_builder` with const builder"),
        }
    }

    #[track_caller]
    pub fn new_block(&mut self) -> BlockId {
        self.expect_func_builder_mut().new_block()
    }

    #[track_caller]
    pub fn active_block(&self) -> BlockId {
        let builder = self.expect_func_builder();
        builder.active
    }

    pub fn loop_ctx(&self) -> Option<LoopCtx> {
        let builder = self.expect_func_builder();
        builder.loop_ctx
    }

    pub fn break_block(&self) -> Option<BlockId> {
        let builder = self.expect_func_builder();
        builder.loop_ctx.map(|l| l.breac)
    }

    pub fn loop_start(&self) -> Option<BlockId> {
        let builder = self.expect_func_builder();
        builder.loop_ctx.map(|l| l.start)
    }

    #[track_caller]
    pub fn in_loop(&mut self, breac: BlockId, f: impl FnOnce(&mut Self, BlockId)) -> BlockId {
        self.in_scope(|ctx, inner| {
            let prev_loop_ctx = ctx.loop_ctx();
            ctx.set_loop_ctx(Some(LoopCtx {
                breac,
                start: inner,
            }));
            f(ctx, inner);
            ctx.set_loop_ctx(prev_loop_ctx);
        })
    }

    #[track_caller]
    pub fn in_scope(&mut self, f: impl FnOnce(&mut Self, BlockId)) -> BlockId {
        let prev_block = self.active_block();

        let inner = self.new_block();
        self.set_active_block(inner);
        f(self, inner);
        self.set_active_block(prev_block);

        inner
    }

    pub fn in_var_scope<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R {
        self.tables.push(SymbolTable::default());
        let result = f(self);
        self.tables.pop();
        result
    }

    pub fn push_pop_sp<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R {
        // TODO: this sort of leaks?
        let sp = OffsetVar::zero(self.anon_var(TyId::USIZE));
        self.ins(Air::ReadSP(sp));
        let result = f(self);
        self.ins(Air::WriteSP(sp));
        result
    }

    pub fn set_active_block(&mut self, block: BlockId) {
        let builder = self.expect_func_builder_mut();
        assert!(block.0 < builder.instrs.len());
        builder.active = block;
    }

    pub fn set_loop_ctx(&mut self, loop_ctx: Option<LoopCtx>) {
        let builder = self.expect_func_builder_mut();
        builder.loop_ctx = loop_ctx;
    }

    pub fn new_var_registered_no_salloc(&mut self, ident: &Ident, ty: TyId) -> Var {
        let var = self.anon_var_no_salloc(ty);
        self.register_var(ident, var);
        var
    }

    #[track_caller]
    pub fn new_var_registered(&mut self, ident: &Ident, ty: TyId) -> Var {
        let var = self.anon_var(ty);
        self.register_var(ident, var);
        var
    }

    #[track_caller]
    pub fn new_func_arg_var_registered(&mut self, ident: &Ident, ty: TyId) -> Var {
        let var = self.anon_var_no_salloc(ty);
        self.func_args.insert(*ident, var);
        var
    }

    fn register_var(&mut self, ident: &Ident, var: Var) {
        if self.tables.is_empty() {
            self.tables.push(SymbolTable::default());
        }

        match self.tables.last_mut() {
            Some(table) => table.register(*ident, var),
            None => unreachable!(),
        }
    }

    pub fn anon_var_no_salloc(&mut self, ty: TyId) -> Var {
        let idx = self.var_index;
        self.var_index += 1;
        self.ty_map.insert(Var(idx), ty);

        Var(idx)
    }

    #[track_caller]
    pub fn anon_var(&mut self, ty: TyId) -> Var {
        let var = self.anon_var_no_salloc(ty);
        self.ins(Air::SAlloc(var, self.tys.ty(ty).size(self)));
        var
    }

    pub fn func_arg_var(&mut self, ident: Ident, ty: TyId) -> Var {
        match self.func_args.get(&ident) {
            Some(arg) => *arg,
            None => self.new_func_arg_var_registered(&ident, ty),
        }
    }

    pub fn var_meta(&self, ident: IdentId) -> Option<&(Ident, Var)> {
        self.tables.iter().rev().find_map(|t| t.symbol(ident))
    }

    pub fn var(&self, ident: IdentId) -> Option<Var> {
        self.var_meta(ident).map(|(_, var)| *var)
    }

    #[track_caller]
    pub fn expect_var(&self, ident: IdentId) -> Var {
        let builder = self.expect_func_builder();

        self.var(ident)
            .or_else(|| {
                builder
                    .func
                    .sig
                    .params
                    .iter()
                    .find(|p| p.ident.id == ident)
                    .map(|p| self.func_args.get(&p.ident).copied().unwrap())
            })
            .expect("invalid var ident")
    }

    #[track_caller]
    pub fn expect_var_ty(&self, var: Var) -> TyId {
        *self.ty_map.get(&var).expect("invalid var")
    }

    pub fn var_ty(&self, ident: &Ident) -> TyId {
        let set = self.key.ident_set(ident.id);
        match set.len() {
            0 => panic!("ident not keyed: {:?}", ident),
            1 => set[0].1,
            _ => {
                // choose the var type that is closest behind `ident`
                let mut prev = None;
                for (option, ty) in set.iter() {
                    if option.span.start == ident.span.start {
                        return *ty;
                    } else if option.span.start > ident.span.start {
                        match prev {
                            Some((_, ty)) => return ty,
                            None => panic!("ident not keyed: {:?}", ident),
                        }
                    }
                    prev = Some((option, *ty));
                }

                match prev {
                    Some((option, ty)) => {
                        if option.span.start <= ident.span.start {
                            return ty;
                        } else {
                            panic!("ident not keyed: {:?}", ident);
                        }
                    }
                    None => panic!("ident not keyed: {:?}", ident),
                }
            }
        }
    }

    pub fn str_lit(&mut self, str: &str) -> (BssEntry, usize) {
        self.bss.str_lit(str)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LoopCtx {
    pub start: BlockId,
    pub breac: BlockId,
}

pub const RET_REG: Reg = Reg::A;

impl<'a> AirCtx<'a> {
    pub fn ins(&mut self, instr: Air<'a>) {
        match &mut self.instr_builder {
            InstrBuilder::Const(instrs) => instrs.push(instr),
            InstrBuilder::Func(b) => b.insert_active(instr),
        }
    }

    pub fn ins_set(&mut self, instrs: impl IntoIterator<Item = Air<'a>>) {
        match &mut self.instr_builder {
            InstrBuilder::Const(konst) => konst.extend(instrs),
            InstrBuilder::Func(b) => b.insert_active_set(instrs),
        }
    }

    pub fn ret_var(&mut self, var: OffsetVar, ty: TyId) {
        match self.tys.ty(ty) {
            Ty::Bool => self.ret_ivar(var, Width::BOOL),
            Ty::Int(ty) => self.ret_ivar(var, ty.width()),
            Ty::Float(ty) => self.ret_ivar(var, ty.width()),
            Ty::Ref(_) => self.ret_ivar(var, Width::PTR),
            Ty::Array(_, _) | Ty::Struct(_) => self.ret_ptr(var),
            ty @ Ty::Unit | ty @ Ty::Str => panic!("cannot return {:?}", ty),
        }
    }

    pub fn ret_ivar(&mut self, var: OffsetVar, width: Width) {
        self.ins(Air::MovIVar(RET_REG, var, width));
        self.ins(Air::Ret);
    }

    pub fn ret_ptr(&mut self, var: OffsetVar) {
        self.ins(Air::Addr(RET_REG, var));
        self.ins(Air::Ret);
    }
}

impl<'a> Deref for AirCtx<'a> {
    type Target = Ctx<'a>;

    fn deref(&self) -> &Self::Target {
        &self.ctx
    }
}

enum InstrBuilder<'a> {
    Func(AirFuncBuilder<'a>),
    Const(Vec<Air<'a>>),
}
