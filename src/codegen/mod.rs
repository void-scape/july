use crate::ir::ctx::Ctx;
use crate::ir::expr::{BinOpKind, Expr, ExprKind};
use crate::ir::func::Func;
use crate::ir::ident::IdentId;
use crate::ir::lit::Lit;
use crate::ir::stmt::Stmt;
use crate::ir::ty::{IntKind, Ty, TypeKey};
use cranelift_codegen::ir::types::*;
use cranelift_codegen::ir::InstBuilder;
use cranelift_codegen::ir::{AbiParam, Function, Signature, UserFuncName};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::{settings, Context};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_module::{FuncId, Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};
use std::collections::HashMap;

pub fn codegen(ctx: &Ctx, key: &TypeKey) -> Vec<u8> {
    let mut module = module();
    define_functions(ctx, key, &mut module);
    module.finish().emit().unwrap()
}

fn module() -> ObjectModule {
    ObjectModule::new(
        ObjectBuilder::new(
            cranelift_native::builder()
                .unwrap()
                .finish(cranelift_codegen::settings::Flags::new(
                    cranelift_codegen::settings::builder(),
                ))
                .unwrap(),
            b"ax",
            cranelift_module::default_libcall_names(),
        )
        .unwrap(),
    )
}

fn define_functions(ctx: &Ctx, key: &TypeKey, module: &mut ObjectModule) {
    let mut cl_ctx = Context::new();
    let mut builder = FunctionBuilderContext::new();

    let mut func_map = HashMap::<IdentId, FuncId>::new();
    for f in ctx.funcs.iter() {
        func_map.insert(f.sig.ident, register_function(module, ctx, key, f));
    }

    for (debug_name, f) in ctx.funcs.iter().enumerate() {
        define_function(
            &mut cl_ctx,
            &mut builder,
            module,
            ctx,
            key,
            f,
            &func_map,
            debug_name as u32,
        );
    }
}

fn register_function(module: &mut ObjectModule, ctx: &Ctx, key: &TypeKey, func: &Func) -> FuncId {
    module
        .declare_function(
            ctx.idents.ident(func.sig.ident),
            Linkage::Export,
            &build_sig(func, key),
        )
        .unwrap()
}

fn define_function(
    cl_ctx: &mut Context,
    builder: &mut FunctionBuilderContext,
    module: &mut ObjectModule,
    ctx: &Ctx,
    key: &TypeKey,
    func: &Func,
    func_map: &HashMap<IdentId, FuncId>,
    i: u32,
) {
    let cl_func = build_function(ctx, builder, module, key, func, func_map, i);
    verify_function(&cl_func);
    cl_ctx.func = cl_func;
    module
        .define_function(*func_map.get(&func.sig.ident).unwrap(), cl_ctx)
        .unwrap();
}

fn build_sig(func: &Func, _key: &TypeKey) -> Signature {
    let mut sig = Signature::new(CallConv::AppleAarch64);
    //for param in func.params.iter() {
    //    sig.params.push(param.ty.abi());
    //}

    match func.sig.ty {
        Ty::Unit => {}
        ty => {
            sig.returns.push(ty.abi());
        }
    }

    sig
}

fn build_function(
    ctx: &Ctx,
    fn_ctx: &mut FunctionBuilderContext,
    module: &mut ObjectModule,
    key: &TypeKey,
    func: &Func,
    func_map: &HashMap<IdentId, FuncId>,
    i: u32,
) -> Function {
    let mut f = Function::with_name_signature(UserFuncName::user(i, i), build_sig(func, key));
    let mut builder = FunctionBuilder::new(&mut f, fn_ctx);
    let mut registry = VarAlloc::default();

    let cl_block = builder.create_block();
    builder.switch_to_block(cl_block);

    let block = ctx.blocks.block(func.block);
    for stmt in block.stmts.iter() {
        match stmt {
            Stmt::Let(ident, _, expr) => {
                let expr = ctx.expr(*expr);
                let cl_var = registry.new_var();
                registry.register(ident.id, cl_var);
                builder.declare_var(cl_var, expr.clty(key));
                expr.assign(
                    module,
                    ctx,
                    cl_var,
                    key,
                    &mut registry,
                    &mut builder,
                    func_map,
                );
            }
            Stmt::Open(expr) | Stmt::Semi(expr) => {
                ctx.expr(*expr)
                    .define(module, ctx, key, &mut registry, &mut builder, func_map);
            }
        }
    }

    if let Some(end) = block.end {
        let end = ctx.expr(end);
        let cl_var = registry.new_var();
        builder.declare_var(cl_var, end.clty(key));
        end.assign(
            module,
            ctx,
            cl_var,
            key,
            &mut registry,
            &mut builder,
            func_map,
        );
        let value = builder.use_var(cl_var);
        builder.ins().return_(&[value]);
    }

    builder.seal_block(cl_block);
    builder.finalize();

    f
}

fn verify_function(func: &Function) {
    let flags = settings::Flags::new(settings::builder());
    let res = cranelift_codegen::verify_function(func, &flags);

    println!("{}", func.display());
    if let Err(errors) = res {
        panic!("{}", errors);
    }
}

impl Ty {
    pub fn abi(&self) -> AbiParam {
        AbiParam::new(self.clty())
    }

    pub fn clty(&self) -> Type {
        match self {
            Self::Int(kind) => match kind {
                IntKind::I32 => I32,
            },
            Self::Unit => panic!("handle unit abi"),
        }
    }
}

#[derive(Default)]
pub struct VarAlloc {
    index: u32,
    vars: HashMap<IdentId, Variable>,
}

impl VarAlloc {
    pub fn new_var(&mut self) -> Variable {
        let idx = self.index;
        self.index += 1;

        Variable::from_u32(idx)
    }

    pub fn register(&mut self, ident: IdentId, var: Variable) {
        self.vars.insert(ident, var);
    }

    pub fn get(&self, ident: IdentId) -> Option<Variable> {
        self.vars.get(&ident).copied()
    }
}

impl Expr {
    pub fn define(
        &self,
        module: &mut ObjectModule,
        ctx: &Ctx,
        key: &TypeKey,
        registry: &mut VarAlloc,
        builder: &mut FunctionBuilder,
        func_map: &HashMap<IdentId, FuncId>,
    ) {
        match self.kind {
            ExprKind::Bin(op, lhs, rhs) => match op.kind {
                BinOpKind::Multiply | BinOpKind::Add => {}
                BinOpKind::AddAssign => {
                    todo!()
                    //let lhs = ctx.expr(lhs);
                    //let lhs_var = registry.new_var();
                    //builder.declare_var(lhs_var, lhs.clty(key));
                    //lhs.assign(module, ctx, lhs_var, key, registry, builder, func_map);
                    //
                    //let rhs = ctx.expr(rhs);
                    //let rhs_var = registry.new_var();
                    //builder.declare_var(rhs_var, rhs.clty(key));
                    //rhs.assign(module, ctx, rhs_var, key, registry, builder, func_map);
                    //
                    //let lhs = builder.use_var(lhs_var);
                    //let rhs = builder.use_var(rhs_var);
                    //let (result, overflow) = builder.ins().uadd_overflow(lhs, rhs);
                    //builder.def_var(cl_var, result);
                }
            },
            ExprKind::Call(sig) => {
                let func =
                    module.declare_func_in_func(*func_map.get(&sig.ident).unwrap(), builder.func);
                builder.ins().call(func, &[]);
            }
            ExprKind::Ret(expr) => {
                if let Some(expr) = expr {
                    let var = registry.new_var();
                    builder.declare_var(var, self.clty(key));
                    ctx.expr(expr)
                        .assign(module, ctx, var, key, registry, builder, func_map);
                    let value = builder.use_var(var);
                    builder.ins().return_(&[value]);
                } else {
                    builder.ins().return_(&[]);
                }
            }
            expr => panic!("cannot define as line: {expr:#?}"),
        }
    }

    pub fn assign(
        &self,
        module: &mut ObjectModule,
        ctx: &Ctx,
        cl_var: Variable,
        key: &TypeKey,
        registry: &mut VarAlloc,
        builder: &mut FunctionBuilder,
        func_map: &HashMap<IdentId, FuncId>,
    ) {
        match self.kind {
            ExprKind::Ident(ident) => {
                let val = builder.use_var(registry.get(ident.id).unwrap());
                builder.def_var(cl_var, val);
            }
            ExprKind::Lit(id) => match ctx.lits.lit(id) {
                Lit::Int(int) => {
                    let value = builder.ins().iconst(self.clty(key), *int);
                    builder.def_var(cl_var, value);
                }
                Lit::Str(_) => todo!(),
            },
            ExprKind::Bin(op, lhs, rhs) => {
                let lhs = ctx.expr(lhs);
                let lhs_var = registry.new_var();
                builder.declare_var(lhs_var, lhs.clty(key));
                lhs.assign(module, ctx, lhs_var, key, registry, builder, func_map);

                let rhs = ctx.expr(rhs);
                let rhs_var = registry.new_var();
                builder.declare_var(rhs_var, rhs.clty(key));
                rhs.assign(module, ctx, rhs_var, key, registry, builder, func_map);

                match op.kind {
                    BinOpKind::Add => {
                        let lhs = builder.use_var(lhs_var);
                        let rhs = builder.use_var(rhs_var);
                        let (result, overflow) = builder.ins().uadd_overflow(lhs, rhs);
                        builder.def_var(cl_var, result);
                    }
                    BinOpKind::Multiply => {
                        let lhs = builder.use_var(lhs_var);
                        let rhs = builder.use_var(rhs_var);
                        let (result, overflow) = builder.ins().umul_overflow(lhs, rhs);
                        builder.def_var(cl_var, result);
                    }
                    BinOpKind::AddAssign => {
                        unreachable!()
                    }
                }
            }
            ExprKind::Call(sig) => {
                let func =
                    module.declare_func_in_func(*func_map.get(&sig.ident).unwrap(), builder.func);
                let call = builder.ins().call(func, &[]);
                let res = builder.inst_results(call)[0];
                builder.def_var(cl_var, res);
            }
            ExprKind::Ret(expr) => {
                if let Some(expr) = expr {
                    ctx.expr(expr)
                        .assign(module, ctx, cl_var, key, registry, builder, func_map);
                    let value = builder.use_var(cl_var);
                    builder.ins().return_(&[value]);
                }
            }
        }
    }

    pub fn clty(&self, key: &TypeKey) -> Type {
        key.ty(self.ty).clty()
    }
}
