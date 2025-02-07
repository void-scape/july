use crate::ir::ctx::Ctx;
use crate::ir::ident::IdentId;
use crate::ir::lit::{Lit, LitKind};
use crate::ir::strukt::StructId;
use crate::ir::ty::{FullTy, IntKind, Ty, TypeKey};
use crate::ir::{self, *};
use cranelift_codegen::ir::InstBuilder;
use cranelift_codegen::ir::{types::*, StackSlot};
use cranelift_codegen::ir::{AbiParam, Function, Signature, UserFuncName};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings::Configurable;
use cranelift_codegen::{settings, Context};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{FuncId, Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};
use ctx::*;
use std::collections::HashMap;
use var::*;

mod ctx;
mod var;

pub fn codegen(ctx: &Ctx, key: &TypeKey) -> Vec<u8> {
    let mut module = module();
    define_functions(ctx, key, &mut module);
    module.finish().emit().unwrap()
}

fn module() -> ObjectModule {
    let mut builder = cranelift_codegen::settings::builder();
    builder.set("libcall_call_conv", "apple_aarch64").unwrap();
    builder.set("tls_model", "macho").unwrap();
    builder.set("is_pic", "true").unwrap();
    //builder.set("opt_level", "speed_and_size").unwrap();

    ObjectModule::new(
        ObjectBuilder::new(
            cranelift_native::builder()
                .unwrap()
                .finish(cranelift_codegen::settings::Flags::new(builder))
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
    let mut f = Function::with_name_signature(UserFuncName::user(i, i), build_sig(func, key));
    let mut ctx = GenCtx::new(
        FunctionBuilder::new(&mut f, builder),
        func.hash(),
        ctx,
        module,
        key,
        func_map,
    );
    build_function(&mut ctx, func);
    ctx.builder.finalize();

    verify_function(&f);
    cl_ctx.func = f;
    module
        .define_function(*func_map.get(&func.sig.ident).unwrap(), cl_ctx)
        .unwrap();
}

fn verify_function(func: &Function) {
    let mut builder = cranelift_codegen::settings::builder();
    builder.set("libcall_call_conv", "apple_aarch64").unwrap();
    builder.set("opt_level", "speed_and_size").unwrap();
    let flags = settings::Flags::new(builder);
    let res = cranelift_codegen::verify_function(func, &flags);

    println!("{}", func.display());
    if let Err(errors) = res {
        panic!("{}", errors);
    }
}

fn build_sig(func: &Func, _key: &TypeKey) -> Signature {
    let mut sig = Signature::new(CallConv::AppleAarch64);
    //for param in func.params.iter() {
    //    sig.params.push(param.ty.abi());
    //}

    match func.sig.ty {
        FullTy::Ty(ty) => {
            if !ty.is_unit() {
                sig.returns.push(AbiParam::new(ty.clty()));
            }
        }
        FullTy::Struct(_) => {
            // address to struct
            sig.returns.push(AbiParam::new(I64));
        }
    }

    sig
}

fn build_function(ctx: &mut GenCtx, func: &Func) {
    let cl_block = ctx.builder.create_block();
    ctx.builder.switch_to_block(cl_block);

    for stmt in func.block.stmts.iter() {
        match stmt {
            Stmt::Semi(stmt) => match stmt {
                SemiStmt::Let(let_) => let_stmt(ctx, let_),
                SemiStmt::Bin(bin) => bin_stmt(ctx, bin),
                SemiStmt::Call(call) => call_stmt(ctx, call),
                SemiStmt::Assign(assign) => assign_stmt(ctx, assign),
                _ => todo!(),
            },
            Stmt::Open(stmt) => panic!("{stmt:#?}"),
        }
    }

    if let Some(end) = &func.block.end {
        if !func.sig.ty.is_unit() {
            return_open(ctx, func.sig.ty, end);
        }
    } else {
        ctx.builder.ins().return_(&[]);
    }

    ctx.builder.seal_block(cl_block);
}

fn bin_stmt(_ctx: &mut GenCtx, stmt: &BinOp) {
    match stmt.kind {
        BinOpKind::Add => {}
        BinOpKind::Sub => {}
        BinOpKind::Mul => {}
        BinOpKind::Field => {}
    }
}

fn assign_stmt(ctx: &mut GenCtx, stmt: &Assign) {
    todo!();
    //match stmt.lhs {
    //    AssignTarget::Ident(ident) => {
    //        let var = ctx.var(ident.id);
    //        match &stmt.rhs {
    //            AssignExpr::Lit(lit) => match ctx.expect_lit(lit.kind) {
    //                LitKind::Int(int) => match stmt.kind {
    //                    AssignKind::Equals => {
    //                        let lit = define_lit(ctx, var.kind.expect_primitive().ty, lit);
    //                        ctx.builder
    //                            .def_var(var.kind.expect_primitive().clvar, lit.value(ctx));
    //                    }
    //                    AssignKind::Add => {
    //                        let int = ctx.builder.ins().iconst(clty.expect_cl(), int);
    //                        ctx.add(var, var, int);
    //                    }
    //                },
    //                _ => todo!(),
    //            },
    //            AssignExpr::Ident(ident) => match stmt.kind {
    //                AssignKind::Equals => {
    //                    let other = ctx.alloc.expect_var(ident.id);
    //                    let value = ctx.builder.use_var(other);
    //                    ctx.builder.def_var(var, value);
    //                }
    //                AssignKind::Add => {
    //                    ctx.add(var, var, ctx.alloc.expect_var(ident.id));
    //                }
    //            },
    //            AssignExpr::Bin(bin) => match stmt.kind {
    //                AssignKind::Equals => {
    //                    bin_op_fill_var(ctx, var, bin, clty);
    //                }
    //                AssignKind::Add => {
    //                    let other = ctx.alloc.new_var();
    //                    ctx.builder.declare_var(other, clty.expect_cl());
    //                    bin_op_fill_var(ctx, other, bin, clty);
    //                    ctx.add(var, var, other);
    //                }
    //            },
    //        }
    //    }
    //}
}

//fn let_stmt(ctx: &mut GenCtx, stmt: &Let) {
//    match stmt.lhs {
//        LetTarget::Ident(ident) => {
//            let clty = ctx.ident_clty(ident.id);
//            let var = ctx.declare(clty, ident.id);
//
//            let value = let_expr_value(ctx, &stmt.rhs, clty);
//            //ctx.builder.def_var(var, value);
//
//            // TODO: this is wrong.
//            match &stmt.rhs {
//                LetExpr::Call(call) => match ctx.clty(call.sig.ty) {
//                    ClType::Struct(s) => {
//                        //let id = ctx.expect_struct_id(s);
//                        //let layout = ctx.structs.expect_layout(id);
//                        //let slot = ctx.builder.create_sized_stack_slot(StackSlotData {
//                        //    kind: StackSlotKind::ExplicitSlot,
//                        //    size: layout.size as u32,
//                        //    align_shift: layout.align_shift(),
//                        //});
//
//                        //let addr = ctx.builder.ins().stack_addr(I32, slot, 0);
//                        //ctx.builder.def_var(var, addr);
//                        //let addr = ctx.builder.ins().sextend(I64, addr);
//                        //let size = ctx.builder.ins().iconst(I64, layout.size as i64);
//                        //let value = ctx.builder.ins().sextend(I64, value);
//                        //ctx.builder.def_var(var, value);
//                    }
//                    _ => {}
//                },
//                _ => {}
//            }
//        }
//    }
//}
//
//#[derive(Debug, Clone, Copy)]
//pub enum LetExprValue {
//    Cl(Value),
//    Struct(StackSlot),
//}
//
//impl Into<LetExprValue> for Value {
//    fn into(self) -> LetExprValue {
//        LetExprValue::Cl(self)
//    }
//}
//
//impl Into<LetExprValue> for StackSlot {
//    fn into(self) -> LetExprValue {
//        LetExprValue::Struct(self)
//    }
//}
//
//fn let_expr_value(ctx: &mut GenCtx, expr: &LetExpr, clty: ClType) -> LetExprValue {
//    match expr {
//        LetExpr::Lit(lit) => match ctx.expect_lit(lit.kind) {
//            LitKind::Int(int) => ctx.builder.ins().iconst(clty.expect_cl(), int).into(),
//            _ => todo!(),
//        },
//        LetExpr::Ident(ident) => {
//            let other = ctx.alloc.expect_var(ident.id);
//            match other {
//                Var::Cl(other) => ctx.builder.use_var(other).into(),
//                Var::Struct(slot) => {
//                    todo!()
//                }
//            }
//        }
//        LetExpr::Bin(bin) => {
//            let var = ctx.alloc.new_var();
//            ctx.builder.declare_var(var, clty.expect_cl());
//            bin_op_fill_var(ctx, var, bin, clty);
//            ctx.builder.use_var(var).into()
//        }
//        LetExpr::Call(call) => match clty {
//            ClType::Cl(_cl) => {
//                let func = ctx.declare_func(call.sig.ident);
//                let call = ctx.builder.ins().call(func, &[]);
//                ctx.builder.inst_results(call)[0].into()
//            }
//            ClType::Struct(_s) => {
//                todo!()
//            }
//        },
//        LetExpr::Struct(def) => {
//            let id = ctx.expect_struct_id(def.name.id);
//            let layout = ctx.structs.expect_layout(id);
//            let slot = ctx.builder.create_sized_stack_slot(StackSlotData {
//                kind: StackSlotKind::ExplicitSlot,
//                size: layout.size as u32,
//                align_shift: layout.align_shift(),
//            });
//
//            for field in def.fields.iter() {
//                let clty = ctx.clty(
//                    ctx.structs
//                        .strukt(id)
//                        .unwrap()
//                        .fields
//                        .iter()
//                        .find(|f| f.name.id == field.name.id)
//                        .map(|f| f.ty)
//                        .unwrap(),
//                );
//
//                //let value = let_expr_value(ctx, &field.expr, clty);
//                //match value {
//                //    LetExprValue::Cl(value) => {
//                //        let offset = *ctx
//                //            .structs
//                //            .offsets(id)
//                //            .unwrap()
//                //            .get(&field.name.id)
//                //            .unwrap() as i32;
//                //        ctx.builder.ins().stack_store(value, slot, offset);
//                //    }
//                //    LetExprValue::Struct(slot) => {}
//                //}
//            }
//
//            slot.into()
//        }
//    }
//}
//
//fn store_struct_in_offset(
//    ctx: &mut GenCtx,
//    offset: u32,
//    def: StructDef,
//    dst: StackSlot,
//    src: StackSlot,
//) {
//}
//
//fn bin_op_fill_var(ctx: &mut GenCtx, var: Variable, bin: &BinOp, clty: ClType) {
//    let lhs_var = ctx.alloc.new_var();
//    ctx.builder.declare_var(lhs_var, clty.expect_cl());
//    bin_op_expr_fill_var(ctx, bin.kind, lhs_var, &bin.lhs, clty);
//
//    let rhs_var = ctx.alloc.new_var();
//    ctx.builder.declare_var(rhs_var, clty.expect_cl());
//    bin_op_expr_fill_var(ctx, bin.kind, rhs_var, &bin.rhs, clty);
//
//    match bin.kind {
//        BinOpKind::Add => {
//            ctx.add(var, lhs_var, rhs_var);
//        }
//        BinOpKind::Sub => {
//            ctx.sub(var, lhs_var, rhs_var);
//        }
//        BinOpKind::Mul => {
//            ctx.mul(var, lhs_var, rhs_var);
//        }
//    }
//}
//
//fn bin_op_expr_fill_var(
//    ctx: &mut GenCtx,
//    kind: BinOpKind,
//    var: Variable,
//    expr: &BinOpExpr,
//    clty: ClType,
//) {
//    match expr {
//        BinOpExpr::Lit(lit) => match ctx.expect_lit(lit.kind) {
//            LitKind::Int(int) => {
//                let int = ctx.builder.ins().iconst(clty.expect_cl(), int);
//                ctx.builder.def_var(var, int);
//            }
//            _ => todo!(),
//        },
//        BinOpExpr::Bin(bin) => {
//            bin_op_fill_var(ctx, var, bin, clty);
//        }
//        BinOpExpr::Ident(ident) => match kind {
//            BinOpKind::Sub | BinOpKind::Mul | BinOpKind::Add => {
//                let other = ctx.alloc.expect_var(ident.id);
//                let value = ctx.builder.use_var(other.expect_cl());
//                ctx.builder.def_var(var, value);
//            }
//        },
//        BinOpExpr::Call(call) => {
//            let func = ctx.declare_func(call.sig.ident);
//            let call = ctx.builder.ins().call(func, &[]);
//            let result = ctx.builder.inst_results(call);
//            ctx.builder.def_var(var, result[0]);
//        }
//    }
//}

fn define_lit(ctx: &mut GenCtx, ty: Ty, lit: &Lit) -> Prim {
    let clty = ty.clty();
    let clvar = ctx.new_var();
    ctx.builder.declare_var(clvar, clty);
    match ctx.expect_lit(lit.kind) {
        LitKind::Int(int) => {
            let val = ctx.builder.ins().iconst(clty, int);
            ctx.builder.def_var(clvar, val);
        }
        _ => todo!(),
    }
    Prim { clvar, clty, ty }
}

fn eval_let_expr(ctx: &mut GenCtx, ty: FullTy, expr: &LetExpr) -> Var {
    match expr {
        LetExpr::Lit(lit) => Var::prim(ty, define_lit(ctx, ty.expect_ty(), lit)),
        LetExpr::Struct(def) => Var::strukt(ty, define_struct(ctx, def)),
        LetExpr::Call(call) => match call.sig.ty {
            FullTy::Struct(s) => {
                let func = ctx.declare_func(call.sig.ident);
                let call = ctx.builder.ins().call(func, &[]);
                let addr = ctx.builder.inst_results(call)[0];
                let id = ctx.expect_struct_id(s);
                Var::strukt(ty, copy_return_struct(ctx, id, addr))
            }
            _ => todo!(),
        },
        LetExpr::Ident(ident) => ctx.var(ident.id).clone(),
        LetExpr::Bin(bin) => eval_bin_op(ctx, ty, bin),
        _ => todo!(),
    }
}

fn eval_bin_op(ctx: &mut GenCtx, ty: FullTy, bin: &BinOp) -> Var {
    if bin.kind.is_primitive() {
        let clvar = ctx.new_var();
        ctx.builder.declare_var(clvar, ty.expect_ty().clty());
        let v1 = eval_bin_op_expr(ctx, ty, &bin.lhs)
            .kind
            .expect_primitive()
            .value(ctx);
        let v2 = eval_bin_op_expr(ctx, ty, &bin.rhs)
            .kind
            .expect_primitive()
            .value(ctx);

        match bin.kind {
            BinOpKind::Add => {
                let (value, overflow) = ctx.builder.ins().usub_overflow(v2, v1);
                ctx.builder.def_var(clvar, value);
            }
            BinOpKind::Sub => {
                let (value, overflow) = ctx.builder.ins().usub_overflow(v2, v1);
                ctx.builder.def_var(clvar, value);
            }
            BinOpKind::Mul => {
                let (value, overflow) = ctx.builder.ins().usub_overflow(v2, v1);
                ctx.builder.def_var(clvar, value);
            }
            _ => unreachable!(),
        }

        Var::prim(ty, Prim::new(ty, clvar))
    } else {
        match bin.kind {
            BinOpKind::Field => todo!(),
            _ => unreachable!(),
        }
    }
}

fn eval_bin_op_expr(ctx: &mut GenCtx, ty: FullTy, expr: &BinOpExpr) -> Var {
    match expr {
        BinOpExpr::Ident(ident) => ctx.var(ident.id).clone(),
        BinOpExpr::Lit(lit) => Var::prim(ty, define_lit(ctx, ty.expect_ty(), lit)),
        BinOpExpr::Bin(bin) => eval_bin_op(ctx, ty, bin),
        BinOpExpr::Call(call) => {
            let clty = ty.expect_ty().clty();
            let clvar = ctx.new_var();
            ctx.builder.declare_var(clvar, clty);

            let func = ctx.declare_func(call.sig.ident);
            let call = ctx.builder.ins().call(func, &[]);
            let results = ctx.builder.inst_results(call);

            ctx.builder.def_var(clvar, results[0]);
            Var::prim(
                ty,
                Prim {
                    clty,
                    clvar,
                    ty: ty.expect_ty(),
                },
            )
        }
    }
}

fn let_stmt(ctx: &mut GenCtx, stmt: &Let) {
    match stmt.lhs {
        LetTarget::Ident(ident) => {
            let var = eval_let_expr(ctx, ctx.ty(ident.id), &stmt.rhs);
            println!("register: {:?}", ident.id);
            ctx.register(ident.id, var);
        }
    }
}

fn call_stmt(ctx: &mut GenCtx, stmt: &Call) {
    let func = ctx.declare_func(stmt.sig.ident);
    ctx.builder.ins().call(func, &[]);
}

fn return_open(ctx: &mut GenCtx, ty: FullTy, stmt: &OpenStmt) {
    match stmt {
        OpenStmt::Lit(lit) => match ctx.expect_lit(lit.kind) {
            LitKind::Int(int) => {
                let clty = ty.expect_ty().clty();
                let int = ctx.builder.ins().iconst(clty, int);
                ctx.builder.ins().return_(&[int]);
            }
            _ => todo!(),
        },
        OpenStmt::Ident(ident) => {
            return_var(ctx, ident.id);
        }
        OpenStmt::Struct(s) => {
            let strukt = define_struct(ctx, s);
            let addr = ctx.builder.ins().stack_addr(I64, strukt.slot, 0);
            ctx.builder.ins().return_(&[addr]);
        }
        OpenStmt::Bin(bin) => match bin.kind {
            BinOpKind::Field => {
                access_struct_field(
                    ctx,
                    bin,
                    |ctx, strukt, ident, ty, slot, offset| {
                        let field_offset = strukt.field_offset(ctx, ident);
                        let val =
                            ctx.builder
                                .ins()
                                .stack_load(ty.clty(), slot, offset + field_offset);
                        ctx.builder.ins().return_(&[val]);
                    },
                    |_, _, _, _| todo!(),
                );
            }
            _ => todo!(),
        },
        _ => todo!(),
    }
}

fn access_struct_field(
    ctx: &mut GenCtx,
    bin: &BinOp,
    with_prim: impl Fn(&mut GenCtx, ir::strukt::Struct, IdentId, Ty, StackSlot, i32),
    with_struct: impl Fn(&ir::strukt::Struct, IdentId, StackSlot, i32),
) {
    let mut path = Vec::new();
    descend_bin_op_field(ctx, bin, &mut path);

    let mut offset = 0;
    let ident = *path.first().unwrap();
    let var = ctx.var(ident);
    let var = var.kind.expect_struct();
    let mut strukt = ctx.structs.strukt(var.id);
    let slot = var.slot;

    for (i, ident) in path.iter().enumerate().skip(1) {
        match strukt.field_ty(*ident) {
            FullTy::Ty(ty) => {
                with_prim(ctx, strukt.clone(), *ident, ty, slot, offset);
                break;
            }
            FullTy::Struct(s) => {
                if i == path.len() - 1 {
                    with_struct(strukt, s, slot, offset);
                } else {
                    offset += strukt.field_offset(ctx, *ident);
                    strukt = ctx.structs.expect_struct_ident(s);
                }
            }
        }
    }
}

fn descend_bin_op_field(ctx: &mut GenCtx, bin: &BinOp, accesses: &mut Vec<IdentId>) {
    if bin.kind == BinOpKind::Field {
        match bin.lhs {
            BinOpExpr::Ident(ident) => {
                if let BinOpExpr::Bin(bin) = &bin.rhs {
                    accesses.push(ident.id);
                    descend_bin_op_field(ctx, bin, accesses);
                } else {
                    let BinOpExpr::Ident(other) = bin.rhs else {
                        panic!()
                    };

                    accesses.push(other.id);
                    accesses.push(ident.id);
                }
            }
            _ => {}
        }
    }
}

fn return_var(ctx: &mut GenCtx, var: IdentId) {
    match &ctx.var(var).kind {
        VarKind::Primitive(prim) => {
            let val = ctx.builder.use_var(prim.clvar);
            //let val = prim.value(ctx);
            ctx.builder.ins().return_(&[val]);
        }
        VarKind::Struct(s) => {
            let slot = s.slot;
            let val = ctx.builder.ins().stack_addr(I64, slot, 0);
            ctx.builder.ins().return_(&[val]);
        }
    }
}

impl Ty {
    pub fn clty(&self) -> Type {
        match self {
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
