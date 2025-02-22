use super::var::*;
use crate::air;
use crate::ir::ctx::Ctx;
use crate::ir::ident::IdentId;
use crate::ir::ty::{Ty, TypeKey, VarHash};
use crate::ir::FuncHash;
use cranelift_codegen::ir::FuncRef;
use cranelift_frontend::{FunctionBuilder, Variable};
use cranelift_module::{FuncId, Module};
use cranelift_object::ObjectModule;
use std::collections::HashMap;
use std::ops::Deref;

pub struct GenCtx<'a> {
    pub builder: FunctionBuilder<'a>,
    pub module: &'a mut ObjectModule,
    cached_vars: HashMap<air::Var, Var>,
    func: FuncHash,
    key: &'a TypeKey,
    ctx: &'a Ctx<'a>,
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
            cached_vars: HashMap::default(),
            func,
            builder,
            ctx,
            module,
            key,
            func_map,
        }
    }

    #[track_caller]
    pub fn ty(&self, ident: IdentId) -> Ty {
        self.key.ty(ident, self.func)
    }

    pub fn declare_func(&mut self, ident: IdentId) -> FuncRef {
        let id = self.func_map.get(&ident).unwrap();
        self.module.declare_func_in_func(*id, self.builder.func)
    }

    pub fn register_var(&mut self, air_var: air::Var, var: Var) {
        self.cached_vars.insert(air_var, var);
    }

    pub fn clvar(&self, air_var: air::Var) -> Variable {
        Variable::from_u32(air_var.index() as u32)
    }

    pub fn get_var(&self, air_var: air::Var) -> Option<Var> {
        self.cached_vars.get(&air_var).cloned()
    }

    #[track_caller]
    pub fn expect_var(&self, air_var: air::Var) -> Var {
        self.get_var(air_var).expect("invalid var")
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
