use crate::ir::ctx::Ctx;
use crate::ir::ident::IdentId;
use crate::ir::lit::LitKind;
use crate::ir::ty::{IntKind, Ty, TypeKey};
use crate::ir::*;
use cranelift_codegen::ir::{types::*, FuncRef};
use cranelift_codegen::ir::{AbiParam, Function, Signature, UserFuncName};
use cranelift_codegen::ir::{InstBuilder, Value};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::{settings, Context};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_module::{FuncId, Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};
use std::collections::HashMap;
use std::ops::Deref;

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
            ctx.expect_ident(func.sig.ident),
            Linkage::Export,
            &build_sig(ctx, func, key),
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

fn build_sig(ctx: &Ctx, func: &Func, _key: &TypeKey) -> Signature {
    let mut sig = Signature::new(CallConv::AppleAarch64);
    //for param in func.params.iter() {
    //    sig.params.push(param.ty.abi());
    //}

    match func.sig.ty {
        Ty::Unit => {}
        ty => {
            sig.returns.push(ctx.abi(ty));
        }
    }

    sig
}

struct GenCtx<'a> {
    pub ctx: &'a Ctx<'a>,
    pub alloc: VarAlloc,
    pub module: &'a mut ObjectModule,
    pub builder: FunctionBuilder<'a>,
    pub key: &'a TypeKey,
    pub func_map: &'a HashMap<IdentId, FuncId>,
}

impl<'a> Deref for GenCtx<'a> {
    type Target = Ctx<'a>;

    fn deref(&self) -> &Self::Target {
        &self.ctx
    }
}

impl GenCtx<'_> {
    pub fn ident_clty(&self, ident: IdentId) -> Type {
        self.clty(self.key.ty(ident))
    }

    pub fn declare_func(&mut self, ident: IdentId) -> FuncRef {
        let id = self.func_map.get(&ident).unwrap();
        self.module.declare_func_in_func(*id, self.builder.func)
    }

    pub fn add(&mut self, dest: Variable, lhs: impl IntoValue, rhs: impl IntoValue) {
        let v1 = lhs.into_value(self);
        let v2 = rhs.into_value(self);
        let (value, overflow) = self.builder.ins().uadd_overflow(v2, v1);
        self.builder.def_var(dest, value);
    }

    pub fn sub(&mut self, dest: Variable, lhs: impl IntoValue, rhs: impl IntoValue) {
        let v1 = lhs.into_value(self);
        let v2 = rhs.into_value(self);
        let (value, overflow) = self.builder.ins().usub_overflow(v2, v1);
        self.builder.def_var(dest, value);
    }

    pub fn mul(&mut self, dest: Variable, lhs: impl IntoValue, rhs: impl IntoValue) {
        let v1 = lhs.into_value(self);
        let v2 = rhs.into_value(self);
        let (value, overflow) = self.builder.ins().umul_overflow(v2, v1);
        self.builder.def_var(dest, value);
    }
}

trait IntoValue {
    fn into_value(self, ctx: &mut GenCtx) -> Value;
}

impl IntoValue for Value {
    fn into_value(self, _ctx: &mut GenCtx) -> Value {
        self
    }
}

impl IntoValue for Variable {
    fn into_value(self, ctx: &mut GenCtx) -> Value {
        ctx.builder.use_var(self)
    }
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
    let mut f = Function::with_name_signature(UserFuncName::user(i, i), build_sig(ctx, func, key));
    let mut ctx = GenCtx {
        alloc: VarAlloc::default(),
        builder: FunctionBuilder::new(&mut f, fn_ctx),
        ctx,
        module,
        key,
        func_map,
    };

    let cl_block = ctx.builder.create_block();
    ctx.builder.switch_to_block(cl_block);

    for stmt in func.block.stmts.iter() {
        match stmt {
            Stmt::Semi(stmt) => match stmt {
                SemiStmt::Let(let_) => let_stmt(&mut ctx, let_),
                SemiStmt::Assign(assign) => assign_stmt(&mut ctx, assign),
                SemiStmt::Bin(bin) => bin_stmt(&mut ctx, bin),
                _ => todo!(),
            },
            Stmt::Open(stmt) => panic!("{stmt:#?}"),
        }
    }

    if let Some(end) = &func.block.end {
        if !func.sig.ty.is_unit() {
            let clty = ctx.clty(func.sig.ty);
            return_open(&mut ctx, clty, end);
        }
    }

    ctx.builder.seal_block(cl_block);
    ctx.builder.finalize();

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

impl Ctx<'_> {
    pub fn abi(&self, ty: Ty) -> AbiParam {
        AbiParam::new(self.clty(ty))
    }

    pub fn clty(&self, ty: Ty) -> Type {
        match ty {
            Ty::Int(kind) => match kind {
                IntKind::I8 => I8,
                IntKind::I16 => I16,
                IntKind::I32 => I32,
                IntKind::I64 => I64,
                IntKind::U8 => I8,
                IntKind::U16 => I16,
                IntKind::U32 => I32,
                IntKind::U64 => I64,
            },
            Ty::Unit => panic!("handle unit abi"),
        }
    }
}

impl GenCtx<'_> {
    pub fn clty(&self, ty: Ty) -> Type {
        self.ctx.clty(ty)
    }
}

fn return_open(ctx: &mut GenCtx, clty: Type, stmt: &OpenStmt) {
    match stmt {
        OpenStmt::Lit(lit) => match ctx.expect_lit(lit.kind) {
            LitKind::Int(int) => {
                let int = ctx.builder.ins().iconst(clty, int);
                ctx.builder.ins().return_(&[int]);
            }
            _ => todo!(),
        },
        OpenStmt::Ident(ident) => {
            let other = ctx.alloc.expect_var(ident.id);
            let value = ctx.builder.use_var(other);
            ctx.builder.ins().return_(&[value]);
        }
        OpenStmt::Bin(bin) => {
            let var = ctx.alloc.new_var();
            ctx.builder.declare_var(var, clty);
            bin_op_fill_var(ctx, var, bin, clty);

            let value = ctx.builder.use_var(var);
            ctx.builder.ins().return_(&[value]);
        }
    }
}

fn bin_stmt(_ctx: &mut GenCtx, stmt: &BinOp) {
    match stmt.kind {
        BinOpKind::Add => {}
        BinOpKind::Sub => {}
        BinOpKind::Mul => {}
    }
}

fn assign_stmt(ctx: &mut GenCtx, stmt: &Assign) {
    match stmt.lhs {
        AssignTarget::Ident(ident) => {
            let var = ctx.alloc.expect_var(ident.id);
            let clty = ctx.ident_clty(ident.id);

            match &stmt.rhs {
                AssignExpr::Lit(lit) => match ctx.expect_lit(lit.kind) {
                    LitKind::Int(int) => match stmt.kind {
                        AssignKind::Equals => {
                            let int = ctx.builder.ins().iconst(clty, int);
                            ctx.builder.def_var(var, int);
                        }
                        AssignKind::Add => {
                            let int = ctx.builder.ins().iconst(clty, int);
                            ctx.add(var, var, int);
                        }
                    },
                    _ => todo!(),
                },
                AssignExpr::Ident(ident) => match stmt.kind {
                    AssignKind::Equals => {
                        let other = ctx.alloc.expect_var(ident.id);
                        let value = ctx.builder.use_var(other);
                        ctx.builder.def_var(var, value);
                    }
                    AssignKind::Add => {
                        ctx.add(var, var, ctx.alloc.expect_var(ident.id));
                    }
                },
                AssignExpr::Bin(bin) => match stmt.kind {
                    AssignKind::Equals => {
                        bin_op_fill_var(ctx, var, bin, clty);
                    }
                    AssignKind::Add => {
                        let other = ctx.alloc.new_var();
                        ctx.builder.declare_var(other, clty);
                        bin_op_fill_var(ctx, other, bin, clty);
                        ctx.add(var, var, other);
                    }
                },
            }
        }
    }
}

fn let_stmt(ctx: &mut GenCtx, stmt: &Let) {
    match stmt.lhs {
        LetTarget::Ident(ident) => {
            let var = ctx.alloc.register(ident.id);
            let clty = ctx.ident_clty(ident.id);
            ctx.builder.declare_var(var, clty);

            match &stmt.rhs {
                LetExpr::Lit(lit) => match ctx.expect_lit(lit.kind) {
                    LitKind::Int(int) => {
                        let int = ctx.builder.ins().iconst(clty, int);
                        ctx.builder.def_var(var, int);
                    }
                    _ => todo!(),
                },
                LetExpr::Ident(ident) => {
                    let other = ctx.alloc.expect_var(ident.id);
                    let value = ctx.builder.use_var(other);
                    ctx.builder.def_var(var, value);
                }
                LetExpr::Bin(bin) => {
                    bin_op_fill_var(ctx, var, bin, clty);
                }
            }
        }
    }
}

fn bin_op_fill_var(ctx: &mut GenCtx, var: Variable, bin: &BinOp, clty: Type) {
    let lhs_var = ctx.alloc.new_var();
    ctx.builder.declare_var(lhs_var, clty);
    bin_op_expr_fill_var(ctx, lhs_var, &bin.lhs, clty);

    let rhs_var = ctx.alloc.new_var();
    ctx.builder.declare_var(rhs_var, clty);
    bin_op_expr_fill_var(ctx, rhs_var, &bin.rhs, clty);

    match bin.kind {
        BinOpKind::Add => {
            ctx.add(var, lhs_var, rhs_var);
        }
        BinOpKind::Sub => {
            ctx.sub(var, lhs_var, rhs_var);
        }
        BinOpKind::Mul => {
            ctx.mul(var, lhs_var, rhs_var);
        }
    }
}

fn bin_op_expr_fill_var(ctx: &mut GenCtx, var: Variable, expr: &BinOpExpr, clty: Type) {
    match expr {
        BinOpExpr::Lit(lit) => match ctx.expect_lit(lit.kind) {
            LitKind::Int(int) => {
                let int = ctx.builder.ins().iconst(clty, int);
                ctx.builder.def_var(var, int);
            }
            _ => todo!(),
        },
        BinOpExpr::Bin(bin) => {
            bin_op_fill_var(ctx, var, bin, clty);
        }
        BinOpExpr::Ident(ident) => {
            let other = ctx.alloc.expect_var(ident.id);
            let value = ctx.builder.use_var(other);
            ctx.builder.def_var(var, value);
        }
        BinOpExpr::Call(call) => {
            let func = ctx.declare_func(call.sig.ident);
            let call = ctx.builder.ins().call(func, &[]);
            let result = ctx.builder.inst_results(call);
            ctx.builder.def_var(var, result[0]);
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

    pub fn register(&mut self, ident: IdentId) -> Variable {
        let var = self.new_var();
        self.vars.insert(ident, var);
        var
    }

    pub fn get_var(&self, ident: IdentId) -> Option<Variable> {
        self.vars.get(&ident).copied()
    }

    #[track_caller]
    pub fn expect_var(&self, ident: IdentId) -> Variable {
        self.get_var(ident).expect("invalid var")
    }
}
