use super::var::*;
use crate::ir::ctx::Ctx;
use crate::ir::ident::IdentId;
use crate::ir::ty::{FullTy, TypeKey, VarHash};
use crate::ir::FuncHash;
use cranelift_codegen::ir::FuncRef;
use cranelift_frontend::{FunctionBuilder, Variable};
use cranelift_module::{FuncId, Module};
use cranelift_object::ObjectModule;
use std::collections::HashMap;
use std::ops::Deref;

pub struct GenCtx<'a> {
    pub builder: FunctionBuilder<'a>,
    func: FuncHash,
    alloc: VarAlloc,
    key: &'a TypeKey,
    ctx: &'a Ctx<'a>,
    module: &'a mut ObjectModule,
    func_map: &'a HashMap<IdentId, FuncId>,
}

impl<'a> Deref for GenCtx<'a> {
    type Target = Ctx<'a>;

    fn deref(&self) -> &Self::Target {
        &self.ctx
    }
}

impl<'a> GenCtx<'a> {
    pub fn new(
        builder: FunctionBuilder<'a>,
        func: FuncHash,
        ctx: &'a Ctx<'a>,
        module: &'a mut ObjectModule,
        key: &'a TypeKey,
        func_map: &'a HashMap<IdentId, FuncId>,
    ) -> Self {
        Self {
            alloc: VarAlloc::default(),
            func,
            builder,
            ctx,
            module,
            key,
            func_map,
        }
    }

    pub fn new_var(&mut self) -> Variable {
        self.alloc.new_var()
    }

    pub fn register(&mut self, ident: IdentId, var: Var) {
        self.alloc.register(ident, self.func, var);
    }

    #[track_caller]
    pub fn var(&self, ident: IdentId) -> &Var {
        self.alloc.var(ident, self.func)
    }

    #[track_caller]
    pub fn ty(&self, ident: IdentId) -> FullTy {
        self.key.ty(ident, self.func)
    }

    pub fn declare_func(&mut self, ident: IdentId) -> FuncRef {
        let id = self.func_map.get(&ident).unwrap();
        self.module.declare_func_in_func(*id, self.builder.func)
    }

    //pub fn add(&mut self, dest: Variable, lhs: impl IntoValue, rhs: impl IntoValue) {
    //    let v1 = lhs.into_value(self);
    //    let v2 = rhs.into_value(self);
    //    let (value, overflow) = self.builder.ins().uadd_overflow(v2, v1);
    //    self.builder.def_var(dest, value);
    //}
    //
    //pub fn sub(&mut self, dest: Variable, lhs: impl IntoValue, rhs: impl IntoValue) {
    //    let v1 = lhs.into_value(self);
    //    let v2 = rhs.into_value(self);
    //    let (value, overflow) = self.builder.ins().usub_overflow(v2, v1);
    //    self.builder.def_var(dest, value);
    //}
    //
    //pub fn mul(&mut self, dest: Variable, lhs: impl IntoValue, rhs: impl IntoValue) {
    //    let v1 = lhs.into_value(self);
    //    let v2 = rhs.into_value(self);
    //    let (value, overflow) = self.builder.ins().umul_overflow(v2, v1);
    //    self.builder.def_var(dest, value);
    //}
}

//trait IntoValue {
//    fn into_value(self, ctx: &mut GenCtx) -> Value;
//}
//
//impl IntoValue for Value {
//    fn into_value(self, _ctx: &mut GenCtx) -> Value {
//        self
//    }
//}
//
//impl IntoValue for Variable {
//    fn into_value(self, ctx: &mut GenCtx) -> Value {
//        ctx.builder.use_var(self)
//    }
//}

#[derive(Default)]
struct VarAlloc {
    index: u32,
    vars: HashMap<VarHash, Var>,
}

impl VarAlloc {
    pub fn new_var(&mut self) -> Variable {
        let idx = self.index;
        self.index += 1;

        Variable::from_u32(idx)
    }

    pub fn register(&mut self, ident: IdentId, func: FuncHash, var: Var) {
        self.vars.insert(VarHash { ident, func }, var);
    }

    #[track_caller]
    pub fn var(&self, ident: IdentId, func: FuncHash) -> &Var {
        self.vars
            .get(&VarHash { ident, func })
            .expect("invalid var")
    }
}
