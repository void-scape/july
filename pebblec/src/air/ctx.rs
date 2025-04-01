use super::data::{Bss, BssEntry};
use super::{Air, AirFunc, AirFuncBuilder, AirLinkage, AirSig, BlockId, OffsetVar, Reg, Var};
use crate::air::Args;
use crate::ir::ctx::Ctx;
use crate::ir::ident::{Ident, IdentId};
use crate::ir::sig::{Param, Sig};
use crate::ir::strukt::StructId;
use crate::ir::ty::infer::SymbolTable;
use crate::ir::ty::store::TyStore;
use crate::ir::ty::{Ty, TyKind, TypeKey, Width};
use crate::ir::{self, *};
use indexmap::IndexMap;
use pebblec_arena::BlobArena;
use std::ops::Deref;

#[derive(Debug)]
pub struct AirCtx<'a, 'ctx> {
    pub tys: TyStore,
    pub key: TypeKey,
    pub tables: Vec<SymbolTable<Var>>,
    pub air_sigs: IndexMap<IdentId, &'a AirSig<'a>>,
    pub impl_air_sigs: IndexMap<(StructId, IdentId), &'a AirSig<'a>>,
    pub storage: BlobArena,

    ctx: &'ctx Ctx<'ctx>,
    var_index: usize,
    func_args: IndexMap<Ident, Var>,
    ty_map: IndexMap<Var, Ty>,
    bss: Bss,
    func: Option<FuncHash>,
    instr_builder: InstrBuilder<'a, 'ctx>,
}

impl PartialEq for AirCtx<'_, '_> {
    fn eq(&self, other: &Self) -> bool {
        self.tys == other.tys
            && self.key == other.key
            && self.tables == other.tables
            && self.ctx == other.ctx
            && self.var_index == other.var_index
            && self.func_args == other.func_args
            && self.ty_map == other.ty_map
            && self.func == other.func
            && self.instr_builder == other.instr_builder
    }
}

fn sig_to_air_sig<'a>(ctx: &Ctx, storage: &BlobArena, sig: &Sig) -> &'a AirSig<'a> {
    &*storage.alloc(AirSig {
        ident: storage.alloc_str(ctx.expect_ident(sig.ident)),
        ty: sig.ty,
        params: if sig.params.is_empty() {
            &[]
        } else {
            storage.alloc_slice(
                &sig.params
                    .iter()
                    .map(|p| param_to_air_param(p))
                    .collect::<Vec<_>>(),
            )
        },
        // TODO: make this prettier
        linkage: match sig.linkage {
            sig::Linkage::Local => AirLinkage::Local,
            sig::Linkage::External { link } => AirLinkage::External {
                link: storage.alloc_str(link),
            },
        },
    })
}

fn param_to_air_param(param: &Param) -> Ty {
    match param {
        Param::Named { ty, .. } => *ty,
        _ => unreachable!(),
    }
}

fn method_sig_to_air_sig<'a>(
    ctx: &Ctx,
    storage: &BlobArena,
    strukt: Ty,
    sig: &Sig,
) -> &'a AirSig<'a> {
    &*storage.alloc(AirSig {
        ident: storage.alloc_str(ctx.expect_ident(sig.ident)),
        ty: sig.ty,
        params: if sig.params.is_empty() {
            &[]
        } else {
            storage.alloc_slice(
                &sig.params
                    .iter()
                    .map(|p| method_param_to_air_param(strukt, p))
                    .collect::<Vec<_>>(),
            )
        },
        // TODO: make this prettier
        linkage: match sig.linkage {
            sig::Linkage::Local => AirLinkage::Local,
            sig::Linkage::External { link } => AirLinkage::External {
                link: storage.alloc_str(link),
            },
        },
    })
}

fn method_param_to_air_param(strukt: Ty, param: &Param) -> Ty {
    match param {
        Param::Named { ty, .. } => *ty,
        Param::Slf(_) => strukt,
    }
}

impl<'a, 'ctx> AirCtx<'a, 'ctx> {
    pub fn new(ctx: &'ctx Ctx<'ctx>, key: TypeKey, mut tys: TyStore) -> Self {
        let storage = BlobArena::default();
        let air_sigs = ctx
            .sigs
            .iter()
            .map(|(ident, sig)| (*ident, sig_to_air_sig(ctx, &storage, sig)))
            .collect();
        let impl_air_sigs = ctx
            .impl_sigs
            .iter()
            .map(|(ids, sig)| {
                let strukt = tys.intern_kind(TyKind::Struct(ids.0)).0;
                let struct_ref = tys.intern_kind(TyKind::Ref(strukt));
                (*ids, method_sig_to_air_sig(ctx, &storage, struct_ref, sig))
            })
            .collect();

        Self {
            ctx,
            key,
            tys,
            air_sigs,
            impl_air_sigs,
            var_index: 0,
            tables: Vec::new(),
            func_args: IndexMap::default(),
            ty_map: IndexMap::default(),
            bss: Bss::default(),
            func: None,
            instr_builder: InstrBuilder::Const(Vec::new()),
            storage,
        }
    }

    pub fn into_inner(self) -> (BlobArena, Bss) {
        (self.storage, self.bss)
    }

    #[track_caller]
    pub fn active_sig(&self) -> &Sig {
        match &self.instr_builder {
            InstrBuilder::Func(b) => b.func.sig,
            InstrBuilder::Const(_) => panic!("cannot call function in const setting"),
        }
    }

    pub fn start_func(&mut self, func: &'ctx ir::Func) -> BlockId {
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
            InstrBuilder::Func(b) => b.build(&self.air_sigs, &self.impl_air_sigs),
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
    fn expect_func_builder(&self) -> &AirFuncBuilder<'a, 'ctx> {
        match &self.instr_builder {
            InstrBuilder::Func(b) => b,
            InstrBuilder::Const(_) => panic!("called `expect_func_builder` with const builder"),
        }
    }

    #[track_caller]
    fn expect_func_builder_mut(&mut self) -> &mut AirFuncBuilder<'a, 'ctx> {
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
        let sp = OffsetVar::zero(self.anon_var(Ty::USIZE));
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

    #[track_caller]
    pub fn new_var_registered(&mut self, ident: &Ident, ty: Ty) -> Var {
        let var = self.anon_var(ty);
        self.register_var(ident, var);
        var
    }

    #[track_caller]
    pub fn new_func_arg_var_registered(&mut self, ident: &Ident, ty: Ty) -> Var {
        let var = self.anon_var_no_salloc(ty);
        self.func_args.insert(*ident, var);
        var
    }

    pub fn register_var(&mut self, ident: &Ident, var: Var) {
        if self.tables.is_empty() {
            self.tables.push(SymbolTable::default());
        }

        match self.tables.last_mut() {
            Some(table) => table.register(*ident, var),
            None => unreachable!(),
        }
    }

    pub fn anon_var_no_salloc(&mut self, ty: Ty) -> Var {
        let idx = self.var_index;
        self.var_index += 1;
        self.ty_map.insert(Var(idx), ty);

        Var(idx)
    }

    #[track_caller]
    pub fn anon_var(&mut self, ty: Ty) -> Var {
        let var = self.anon_var_no_salloc(ty);
        self.ins(Air::SAlloc(var, ty.size(&self.tys)));
        var
    }

    pub fn func_arg_var(&mut self, ident: Ident, ty: Ty) -> Var {
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
                builder.func.sig.params.iter().find_map(|p| match &p {
                    Param::Named { ident: name, .. } => {
                        if name.id == ident {
                            self.func_args.get(name).copied()
                        } else {
                            None
                        }
                    }
                    Param::Slf(ident) => self.func_args.get(ident).copied(),
                })
            })
            .expect("invalid var ident")
    }

    #[track_caller]
    pub fn expect_var_ty(&self, var: Var) -> Ty {
        *self.ty_map.get(&var).expect("invalid var")
    }

    pub fn var_ty(&self, ident: &Ident) -> Ty {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopCtx {
    pub start: BlockId,
    pub breac: BlockId,
}

pub const RET_REG: Reg = Reg::A;

impl<'a, 'ctx> AirCtx<'a, 'ctx> {
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

    pub fn call(&mut self, sig: &Sig, args: Args) {
        let air_sig = self.air_sigs.get(&sig.ident).unwrap();
        self.ins(Air::Call(air_sig, args));
    }

    pub fn method_call(&mut self, sig: &Sig, strukt: StructId, args: Args) {
        let air_sig = self.impl_air_sigs.get(&(strukt, sig.ident)).unwrap();
        self.ins(Air::Call(air_sig, args));
    }

    pub fn ret_var(&mut self, var: OffsetVar, ty: Ty) {
        match ty.0 {
            TyKind::Bool => self.ret_ivar(var, Width::BOOL),
            TyKind::Int(ty) => self.ret_ivar(var, ty.width()),
            TyKind::Float(ty) => self.ret_ivar(var, ty.width()),
            TyKind::Array(_, _)
            | TyKind::Slice(_)
            | TyKind::Ref(TyKind::Str)
            | TyKind::Struct(_) => self.ret_ptr(var),
            TyKind::Ref(_) => self.ret_ivar(var, Width::PTR),
            ty @ TyKind::Unit | ty @ TyKind::Str => panic!("cannot return {:?}", ty),
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

impl<'a, 'ctx> Deref for AirCtx<'a, 'ctx> {
    type Target = Ctx<'ctx>;

    fn deref(&self) -> &Self::Target {
        &self.ctx
    }
}

#[derive(Debug, PartialEq)]
enum InstrBuilder<'a, 'ctx> {
    Func(AirFuncBuilder<'a, 'ctx>),
    Const(Vec<Air<'a>>),
}
