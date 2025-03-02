use super::{Air, AirFunc, AirFuncBuilder, Args, BlockId, IntKind, OffsetVar, Reg, Var};
use crate::ir::ctx::Ctx;
use crate::ir::ident::IdentId;
use crate::ir::sig::Sig;
use crate::ir::ty::store::TyId;
use crate::ir::ty::{Ty, TypeKey};
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
    var_map: HashMap<(FuncHash, IdentId), Var>,
    ty_map: HashMap<Var, TyId>,
    func: Option<FuncHash>,
    func_builder: Option<AirFuncBuilder<'a>>,
}

impl<'a> AirCtx<'a> {
    pub fn new(ctx: &'a Ctx<'a>, key: &'a TypeKey) -> Self {
        Self {
            ctx,
            key,
            var_index: 0,
            var_map: HashMap::default(),
            ty_map: HashMap::default(),
            func: None,
            func_builder: None,
        }
    }

    #[track_caller]
    pub fn active_sig(&self) -> &'a Sig<'a> {
        self.func_builder.as_ref().expect("no active func").func.sig
    }

    pub fn start(&mut self, func: &'a ir::Func) -> BlockId {
        self.func = Some(func.hash());
        let mut builder = AirFuncBuilder::new(func);
        let id = builder.new_block();
        self.func_builder = Some(builder);
        id
    }

    #[track_caller]
    pub fn finish(&mut self) -> AirFunc<'a> {
        self.func_builder
            .take()
            .expect("cannot finish with no builder")
            .build()
    }

    #[track_caller]
    pub fn new_block(&mut self) -> BlockId {
        self.func_builder
            .as_mut()
            .expect("init func builder")
            .new_block()
    }

    pub fn ins_in_block(&mut self, block: BlockId, ins: Air<'a>) {
        self.func_builder
            .as_mut()
            .expect("init func builder")
            .insert(block, ins);
    }

    #[track_caller]
    pub fn active_block(&self) -> BlockId {
        let builder = self.func_builder.as_ref().expect("init func builder");

        if builder.instrs.is_empty() {
            panic!("expected an active block");
        }

        BlockId(builder.instrs.len().saturating_sub(1))
    }

    #[track_caller]
    pub fn in_new_block(&mut self, f: impl FnOnce(&mut Self, BlockId)) -> BlockId {
        let inner = self.new_block();
        f(self, inner);
        inner
    }

    pub fn new_var_registered_with_hash(
        &mut self,
        ident: IdentId,
        hash: FuncHash,
        ty: TyId,
    ) -> Var {
        let var = self.anon_var(ty);
        assert!(self.var_map.get(&(hash, ident)).is_none());
        self.var_map.insert((hash, ident), var);
        var
    }

    pub fn new_var_registered_no_salloc(&mut self, ident: IdentId, ty: TyId) -> Var {
        let func = self.func.expect("AirCtx func hash not set is `lower_func`");
        let var = self.anon_var_no_salloc(ty);
        assert!(self.var_map.get(&(func, ident)).is_none());
        self.var_map.insert((func, ident), var);
        var
    }

    #[track_caller]
    pub fn new_var_registered(&mut self, ident: IdentId, ty: TyId) -> Var {
        let func = self.func.expect("AirCtx func hash not set is `lower_func`");
        let var = self.anon_var(ty);
        assert!(self.var_map.get(&(func, ident)).is_none());
        self.var_map.insert((func, ident), var);
        var
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

    pub fn get_var_with_hash(&self, ident: IdentId, hash: FuncHash) -> Option<Var> {
        self.var_map.get(&(hash, ident)).copied()
    }

    #[track_caller]
    pub fn expect_var(&self, ident: IdentId) -> Var {
        self.get_var(ident).expect("invalid var ident")
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
}

pub const RET_REG: Reg = Reg::A;

impl<'a> AirCtx<'a> {
    pub fn ins(&mut self, instr: Air<'a>) {
        self.func_builder
            .as_mut()
            .expect("init func builder")
            .insert_active(instr);
    }

    pub fn ins_set(&mut self, instrs: &[Air<'a>]) {
        self.func_builder
            .as_mut()
            .expect("init func builder")
            .insert_active_set(instrs.iter().cloned());
    }

    pub fn ret_var(&mut self, var: OffsetVar, ty: TyId) {
        match self.tys.ty(ty) {
            Ty::Unit => panic!("return unit"),
            Ty::Bool => self.ret_ivar(var, IntKind::I64),
            Ty::Int(ty) => self.ret_ivar(var, ty.kind()),
            Ty::Struct(_) => self.ret_ptr(var),
        }
    }

    pub fn ret_iconst(&mut self, val: i64) {
        self.ins(Air::MovIConst(RET_REG, val));
        self.ins(Air::Ret);
    }

    pub fn ret_ivar(&mut self, var: OffsetVar, kind: IntKind) {
        self.ins(Air::MovIVar(RET_REG, var, kind));
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
