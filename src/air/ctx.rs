use super::data::{Bss, BssEntry};
use super::{Air, AirFunc, AirFuncBuilder, Args, BlockId, OffsetVar, Reg, Var};
use crate::ir::ctx::Ctx;
use crate::ir::ident::IdentId;
use crate::ir::sig::Sig;
use crate::ir::ty::store::TyId;
use crate::ir::ty::{Ty, TypeKey, VarHash, Width};
use crate::ir::{self, *};
use std::collections::HashMap;
use std::ops::Deref;

/// Contains relevant context for constructing [`Air`] instructions.
///
/// To start constructing a function, use [`AirCtx::start`], and to collect the resulting
/// [`AirFunc`], use [`AirCtx::finish`]. `AirCtx` reuses data structures for efficiency.
pub struct AirCtx<'a> {
    ctx: &'a Ctx<'a>,
    key: &'a TypeKey,
    var_index: usize,
    var_map: HashMap<VarHash, Var>,
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
            var_index: 0,
            var_map: HashMap::default(),
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
    fn expect_func_builder(&'a self) -> &'a AirFuncBuilder<'a> {
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

    pub fn ins_in_block(&mut self, block: BlockId, ins: Air<'a>) {
        self.expect_func_builder_mut().insert(block, ins);
    }

    #[track_caller]
    pub fn active_block(&self) -> BlockId {
        let builder = self.expect_func_builder();
        builder.active
    }

    #[track_caller]
    pub fn in_new_block(&mut self, f: impl FnOnce(&mut Self, BlockId)) -> BlockId {
        let prev = self.active_block();
        let inner = self.new_block();
        self.set_active_block(inner);
        f(self, inner);
        self.set_active_block(prev);
        inner
    }

    pub fn set_active_block(&mut self, block: BlockId) {
        let builder = self.expect_func_builder_mut();
        assert!(block.0 < builder.instrs.len());
        builder.active = block;
    }

    pub fn new_var_registered_with_hash(
        &mut self,
        ident: IdentId,
        hash: FuncHash,
        ty: TyId,
    ) -> Var {
        let var = self.anon_var(ty);
        let hash = VarHash::Func { func: hash, ident };
        assert!(self.var_map.get(&hash).is_none());
        self.var_map.insert(hash, var);
        var
    }

    pub fn new_var_registered_no_salloc(&mut self, ident: IdentId, ty: TyId) -> Var {
        let func = self.func.expect("AirCtx func hash not set is `lower_func`");
        let var = self.anon_var_no_salloc(ty);
        let hash = VarHash::Func { func, ident };
        assert!(self.var_map.get(&hash).is_none());
        self.var_map.insert(hash, var);
        var
    }

    #[track_caller]
    pub fn new_var_registered(&mut self, ident: IdentId, ty: TyId) -> Var {
        if matches!(self.instr_builder, InstrBuilder::Func(_)) {
            let func = self.func.expect("AirCtx func hash not set is `lower_func`");
            let var = self.anon_var(ty);
            let hash = VarHash::Func { func, ident };
            assert!(self.var_map.get(&hash).is_none());
            self.var_map.insert(hash, var);
            var
        } else {
            let var = self.anon_var(ty);
            let hash = VarHash::Const(ident);
            assert!(self.var_map.get(&hash).is_none());
            self.var_map.insert(hash, var);
            var
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

    pub fn get_var(&self, ident: IdentId) -> Option<Var> {
        self.get_var_with_hash(
            ident,
            self.func.expect("AirCtx func hash not set in `lower_func`"),
        )
    }

    pub fn get_const(&self, ident: IdentId) -> Option<Var> {
        self.var_map.get(&VarHash::Const(ident)).copied()
    }

    pub fn get_var_with_hash(&self, ident: IdentId, func: FuncHash) -> Option<Var> {
        self.var_map.get(&VarHash::Func { func, ident }).copied()
    }

    #[track_caller]
    pub fn expect_var(&self, ident: IdentId) -> Var {
        self.get_var(ident)
            .or_else(|| self.get_const(ident))
            .expect("invalid var ident")
    }

    #[track_caller]
    pub fn expect_var_ty(&self, var: Var) -> TyId {
        *self.ty_map.get(&var).expect("invalid var")
    }

    #[track_caller]
    pub fn var_ty(&self, ident: IdentId) -> TyId {
        self.key.ty(
            ident,
            self.func.expect("AirCtx func hash not set in `lower_func`"),
        )
    }

    pub fn str_lit(&mut self, str: &str) -> (BssEntry, usize) {
        self.bss.str_lit(str)
    }
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

    pub fn call(&mut self, sig: &'a Sig, args: Args) {
        self.ins(Air::Call(sig, args))
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
