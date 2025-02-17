//use crate::air::{Air, AirFunc, Val};
//use crate::ir::ctx::Ctx;
//use crate::ir::ident::IdentId;
//use crate::ir::lit::{Lit, LitKind};
//use crate::ir::ty::{FullTy, IntKind, Ty, TypeKey};
//use crate::ir::*;
//use anstream::println;
//use cranelift_codegen::ir::types::*;
//use cranelift_codegen::ir::InstBuilder;
//use cranelift_codegen::ir::{AbiParam, Function, Signature, UserFuncName};
//use cranelift_codegen::isa::{CallConv, TargetFrontendConfig};
//use cranelift_codegen::settings::Configurable;
//use cranelift_codegen::{settings, Context};
//use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
//use cranelift_module::{FuncId, Linkage, Module};
//use cranelift_object::{ObjectBuilder, ObjectModule};
//use ctx::*;
//use std::collections::HashMap;
//use var::*;
//
//mod ctx;
//mod var;
//
//pub fn codegen(ctx: &Ctx, key: &TypeKey, funcs: &[AirFunc]) -> Vec<u8> {
//    let mut module = module();
//    define_functions(ctx, key, funcs, &mut module);
//    module.finish().emit().unwrap()
//}
//
//fn module() -> ObjectModule {
//    let mut builder = cranelift_codegen::settings::builder();
//    builder.set("libcall_call_conv", "apple_aarch64").unwrap();
//    builder.set("tls_model", "macho").unwrap();
//    builder.set("is_pic", "true").unwrap();
//    //builder.set("opt_level", "speed_and_size").unwrap();
//
//    ObjectModule::new(
//        ObjectBuilder::new(
//            cranelift_native::builder()
//                .unwrap()
//                .finish(cranelift_codegen::settings::Flags::new(builder))
//                .unwrap(),
//            b"ax",
//            cranelift_module::default_libcall_names(),
//        )
//        .unwrap(),
//    )
//}
//
//fn define_functions(ctx: &Ctx, key: &TypeKey, funcs: &[AirFunc], module: &mut ObjectModule) {
//    let mut cl_ctx = Context::new();
//    let mut builder = FunctionBuilderContext::new();
//
//    let mut func_map = HashMap::<IdentId, FuncId>::new();
//    for f in funcs.iter().map(|f| f.func) {
//        func_map.insert(f.sig.ident, register_function(module, ctx, key, f));
//    }
//
//    for (debug_name, f) in funcs.iter().enumerate() {
//        define_function(
//            &mut cl_ctx,
//            &mut builder,
//            module,
//            ctx,
//            key,
//            f,
//            &func_map,
//            debug_name as u32,
//        );
//    }
//}
//
//fn register_function(module: &mut ObjectModule, ctx: &Ctx, key: &TypeKey, func: &Func) -> FuncId {
//    module
//        .declare_function(
//            ctx.expect_ident(func.sig.ident),
//            Linkage::Export,
//            &build_sig(func, key),
//        )
//        .unwrap()
//}
//
//fn define_function(
//    cl_ctx: &mut Context,
//    builder: &mut FunctionBuilderContext,
//    module: &mut ObjectModule,
//    ctx: &Ctx,
//    key: &TypeKey,
//    func: &AirFunc,
//    func_map: &HashMap<IdentId, FuncId>,
//    i: u32,
//) {
//    let mut f = Function::with_name_signature(UserFuncName::user(i, i), build_sig(func.func, key));
//    let mut ctx = GenCtx::new(
//        FunctionBuilder::new(&mut f, builder),
//        func.func.hash(),
//        ctx,
//        module,
//        key,
//        func_map,
//    );
//    build_function(&mut ctx, func);
//    ctx.builder.finalize();
//
//    verify_function(&f);
//    cl_ctx.func = f;
//    module
//        .define_function(*func_map.get(&func.func.sig.ident).unwrap(), cl_ctx)
//        .unwrap();
//}
//
//fn verify_function(func: &Function) {
//    let mut builder = cranelift_codegen::settings::builder();
//    builder.set("libcall_call_conv", "apple_aarch64").unwrap();
//    builder.set("opt_level", "speed_and_size").unwrap();
//    let flags = settings::Flags::new(builder);
//    let res = cranelift_codegen::verify_function(func, &flags);
//
//    println!("{}", func.display());
//    if let Err(errors) = res {
//        panic!("{}", errors);
//    }
//}
//
//fn build_sig(func: &Func, _key: &TypeKey) -> Signature {
//    let mut sig = Signature::new(CallConv::AppleAarch64);
//    //for param in func.params.iter() {
//    //    sig.params.push(param.ty.abi());
//    //}
//
//    match func.sig.ty {
//        FullTy::Ty(ty) => {
//            if !ty.is_unit() {
//                sig.returns.push(AbiParam::new(ty.clty()));
//            }
//        }
//        FullTy::Struct(_) => {
//            // address to struct
//            sig.returns.push(AbiParam::new(I64));
//        }
//    }
//
//    sig
//}
//
//fn build_function(ctx: &mut GenCtx, func: &AirFunc) {
//    let cl_block = ctx.builder.create_block();
//    ctx.builder.switch_to_block(cl_block);
//
//    for instr in func.instrs.iter() {
//        match instr {
//            Air::Ret(val) => match val {
//                Some(val) => match val {
//                    Val::Lit(ty, lit) => match ctx.expect_lit(*lit) {
//                        LitKind::Int(int) => {
//                            let val = ctx.builder.ins().iconst(ty.clty(), int);
//                            ctx.builder.ins().return_(&[val]);
//                        }
//                        _ => todo!(),
//                    },
//                    Val::Ptr(var) => {
//                        let var = ctx.expect_var(*var);
//                        match var.kind {
//                            VarKind::Slot(slot) => {
//                                let addr = ctx.builder.ins().stack_addr(I64, slot, 0);
//                                ctx.builder.ins().return_(&[addr]);
//                            }
//                            VarKind::Primitive(prim) => {
//                                let addr = ctx.builder.ins().addr(I64, slot, 0);
//                                ctx.builder.ins().return_(&[addr]);
//                            }
//                        }
//                    }
//                    Val::Owned(var) => {
//                        let var = ctx.expect_var(*var);
//                        match var.kind {
//                            VarKind::Slot(slot) => {
//                                let addr = ctx.builder.ins().stack_addr(I64, slot, 0);
//                                ctx.builder.ins().return_(&[addr]);
//                            }
//                            VarKind::Primitive(prim) => {
//                                assert_eq!(FullTy::Ty(prim.ty), func.func.sig.ty);
//                                let val = ctx.builder.use_var(prim.clvar);
//                                ctx.builder.ins().return_(&[val]);
//                            }
//                        }
//                    }
//                },
//                None => {
//                    ctx.builder.ins().return_(&[]);
//                }
//            },
//            Air::Alloc(var, ty) => match ty {
//                FullTy::Struct(id) => {
//                    let strukt = Var::slot(FullTy::Struct(*id), var::allocate_struct(ctx, *id));
//                    ctx.register_var(*var, strukt);
//                }
//                FullTy::Ty(prim_ty) => {
//                    let clvar = ctx.clvar(*var);
//                    ctx.builder.declare_var(clvar, prim_ty.clty());
//                    let prim = Var::prim(*ty, Prim::new(*ty, clvar));
//                    ctx.register_var(*var, prim);
//                }
//            },
//            Air::Store(key, offset, val) => {
//                let var = ctx.expect_var(*key);
//                match var.kind {
//                    VarKind::Slot(slot) => match val {
//                        Val::Lit(ty, lit) => match ctx.expect_lit(*lit) {
//                            LitKind::Int(int) => {
//                                let val = ctx.builder.ins().iconst(ty.clty(), int);
//                                ctx.builder.ins().stack_store(val, slot, *offset);
//                            }
//                            _ => todo!(),
//                        },
//                        _ => todo!(),
//                    },
//                    VarKind::Primitive(prim) => match val {
//                        _ => todo!(),
//                    },
//                }
//            }
//            Air::CallAssign(var, sig) => {
//                let func_ref = ctx.declare_func(sig.ident);
//                match sig.ty {
//                    FullTy::Ty(ty) => {
//                        todo!()
//                    }
//                    FullTy::Struct(id) => {
//                        let slot = match ctx.get_var(*var) {
//                            Some(strukt) => match strukt.kind {
//                                VarKind::Slot(slot) => slot,
//                                VarKind::Primitive(prim) => todo!(),
//                            },
//                            None => {
//                                let slot = allocate_struct(ctx, id);
//                                ctx.register_var(*var, Var::slot(sig.ty, slot));
//                                slot
//                            }
//                        };
//
//                        let func = ctx.declare_func(sig.ident);
//                        let call = ctx.builder.ins().call(func, &[]);
//                        let addr = ctx.builder.inst_results(call)[0];
//                        copy_return_struct(ctx, slot, id, addr);
//                    }
//                }
//            }
//            Air::Load { dst, src, ty } => {
//                todo!()
//                //let var = ctx.expect_var(var);
//                //match var.kind {
//                //    VarKind::Slot(slot) => {
//                //
//                //    }
//                //}
//            }
//        }
//    }
//
//    //for stmt in func.block.stmts.iter() {
//    //    match stmt {
//    //        Stmt::Semi(stmt) => match stmt {
//    //            SemiStmt::Let(let_) => let_stmt(ctx, let_),
//    //            SemiStmt::Bin(bin) => bin_stmt(ctx, bin),
//    //            SemiStmt::Call(call) => call_stmt(ctx, call),
//    //            SemiStmt::Assign(assign) => assign_stmt(ctx, assign),
//    //            _ => todo!(),
//    //        },
//    //        Stmt::Open(stmt) => panic!("{stmt:#?}"),
//    //    }
//    //}
//    //
//    //if let Some(end) = &func.block.end {
//    //    if !func.sig.ty.is_unit() {
//    //        return_open(ctx, func.sig.ty, end);
//    //    } else {
//    //        return_unit(ctx, end);
//    //    }
//    //} else {
//    //    ctx.builder.ins().return_(&[]);
//    //}
//
//    ctx.builder.seal_block(cl_block);
//}
//
////fn bin_stmt(_ctx: &mut GenCtx, stmt: &BinOp) {
////    match stmt.kind {
////        BinOpKind::Add => {}
////        BinOpKind::Sub => {}
////        BinOpKind::Mul => {}
////        BinOpKind::Field => {}
////    }
////}
////
////fn assign_stmt(ctx: &mut GenCtx, stmt: &Assign) {
////    println!("{:#?}", stmt);
////    let var = match &stmt.lhs {
////        AssignTarget::Ident(ident) => ctx.var(ident.id),
////        AssignTarget::Field(field) => eval_struct_field(ctx, field),
////    };
////
////    match &stmt.rhs {
////        AssignExpr::Lit(lit) => match ctx.expect_lit(lit.kind) {
////            LitKind::Int(int) => match stmt.kind {
////                AssignKind::Equals => match var.kind {
////                    VarKind::Primitive(prim) => {
////                        let lit = define_lit(ctx, var.kind.expect_primitive().ty, lit);
////                        let val = lit.value(ctx);
////                        ctx.builder.def_var(prim.clvar, val);
////                    }
////                    VarKind::Offset(s, o) => {
////                        let lit = define_lit(ctx, o.ty.expect_ty(), lit);
////                        let val = lit.value(ctx);
////                        ctx.builder.ins().stack_store(val, s.slot, o.offset);
////                    }
////                    kind => panic!("{:#?}", kind),
////                },
////                AssignKind::Add => match var.kind {
////                    VarKind::Primitive(prim) => {
////                        let v1 = ctx.builder.ins().iconst(prim.clty, int);
////                        let v2 = prim.value(ctx);
////                        let (value, _) = ctx.builder.ins().uadd_overflow(v2, v1);
////                        ctx.builder.def_var(prim.clvar, value);
////                    }
////                    _ => unreachable!(),
////                },
////            },
////            _ => todo!(),
////        },
////        AssignExpr::Bin(bin) => {
////            let ty = var.ty;
////            let val = eval_bin_op(ctx, ty, bin);
////            match &val.kind {
////                VarKind::Primitive(other) => match var.kind {
////                    VarKind::Primitive(var) => match stmt.kind {
////                        AssignKind::Add => {
////                            let v1 = var.value(ctx);
////                            let v2 = other.value(ctx);
////                            let (value, _) = ctx.builder.ins().uadd_overflow(v2, v1);
////                            ctx.builder.def_var(var.clvar, value);
////                        }
////                        AssignKind::Equals => {
////                            let val = other.value(ctx);
////                            ctx.builder.def_var(var.clvar, val);
////                        }
////                    },
////                    _ => unreachable!(),
////                },
////                VarKind::Struct(other) => match var.kind {
////                    VarKind::Struct(var) => {
////                        copy_struct_to(ctx, &var, other);
////                    }
////                    _ => unreachable!(),
////                },
////                _ => unreachable!(),
////            }
////        }
////        AssignExpr::Struct(other) => {
////            debug_assert_eq!(FullTy::Struct(other.name.id), var.ty);
////            // TODO: define and insert in same step
////            let other = define_struct(ctx, other);
////            match &var.kind {
////                VarKind::Struct(s) => {
////                    copy_struct_to(ctx, s, &other);
////                }
////                VarKind::Offset(s, o) => {
////                    copy_struct_to_slot(ctx, s.slot, &other, o.offset);
////                }
////                VarKind::Primitive(_) => todo!(),
////                VarKind::Enum(e) => {
////                    todo!()
////                }
////            }
////        }
////        _ => todo!(),
////    }
////}
////
////fn define_lit(ctx: &mut GenCtx, ty: Ty, lit: &Lit) -> Prim {
////    let clty = ty.clty();
////    let clvar = ctx.new_var();
////    ctx.builder.declare_var(clvar, clty);
////    match ctx.expect_lit(lit.kind) {
////        LitKind::Int(int) => {
////            let val = ctx.builder.ins().iconst(clty, int);
////            ctx.builder.def_var(clvar, val);
////        }
////        _ => todo!(),
////    }
////    Prim { clvar, clty, ty }
////}
////
////fn eval_let_expr(ctx: &mut GenCtx, ty: FullTy, expr: &LetExpr) -> Var {
////    match expr {
////        LetExpr::Lit(lit) => Var::prim(ty, define_lit(ctx, ty.expect_ty(), lit)),
////        LetExpr::Struct(def) => Var::strukt(ty, define_struct(ctx, def)),
////        LetExpr::Enum(def) => Var::enom(ty, define_enum(ctx, def)),
////        LetExpr::Call(call) => match call.sig.ty {
////            FullTy::Struct(s) => {
////                let func = ctx.declare_func(call.sig.ident);
////                let call = ctx.builder.ins().call(func, &[]);
////                let addr = ctx.builder.inst_results(call)[0];
////                let id = ctx.expect_struct_id(s);
////                Var::strukt(ty, copy_return_struct(ctx, id, addr))
////            }
////            _ => todo!(),
////        },
////        LetExpr::Ident(ident) => ctx.var(ident.id).clone(),
////        LetExpr::Bin(bin) => eval_bin_op(ctx, ty, bin),
////    }
////}
////
////fn eval_bin_op(ctx: &mut GenCtx, ty: FullTy, bin: &BinOp) -> Var {
////    if bin.kind.is_primitive() {
////        let clvar = ctx.new_var();
////        ctx.builder.declare_var(clvar, ty.expect_ty().clty());
////        let v1 = eval_bin_op_expr(ctx, ty, &bin.lhs)
////            .kind
////            .expect_primitive()
////            .value(ctx);
////        let v2 = eval_bin_op_expr(ctx, ty, &bin.rhs)
////            .kind
////            .expect_primitive()
////            .value(ctx);
////
////        match bin.kind {
////            BinOpKind::Add => {
////                let (value, _) = ctx.builder.ins().uadd_overflow(v2, v1);
////                ctx.builder.def_var(clvar, value);
////            }
////            BinOpKind::Sub => {
////                let (value, _) = ctx.builder.ins().usub_overflow(v2, v1);
////                ctx.builder.def_var(clvar, value);
////            }
////            BinOpKind::Mul => {
////                let (value, _) = ctx.builder.ins().umul_overflow(v2, v1);
////                ctx.builder.def_var(clvar, value);
////            }
////            _ => unreachable!(),
////        }
////
////        Var::prim(ty, Prim::new(ty, clvar))
////    } else {
////        match &bin.kind {
////            BinOpKind::Field => eval_struct_field(ctx, bin),
////            _ => unreachable!(),
////        }
////    }
////}
////
////fn eval_bin_op_expr(ctx: &mut GenCtx, ty: FullTy, expr: &BinOpExpr) -> Var {
////    match expr {
////        BinOpExpr::Ident(ident) => ctx.var(ident.id).clone(),
////        BinOpExpr::Lit(lit) => Var::prim(ty, define_lit(ctx, ty.expect_ty(), lit)),
////        BinOpExpr::Bin(bin) => eval_bin_op(ctx, ty, bin),
////        BinOpExpr::Call(call) => {
////            let clty = ty.expect_ty().clty();
////            let clvar = ctx.new_var();
////            ctx.builder.declare_var(clvar, clty);
////
////            let func = ctx.declare_func(call.sig.ident);
////            let call = ctx.builder.ins().call(func, &[]);
////            let results = ctx.builder.inst_results(call);
////
////            ctx.builder.def_var(clvar, results[0]);
////            Var::prim(
////                ty,
////                Prim {
////                    clty,
////                    clvar,
////                    ty: ty.expect_ty(),
////                },
////            )
////        }
////    }
////}
////
////fn let_stmt(ctx: &mut GenCtx, stmt: &Let) {
////    match stmt.lhs {
////        LetTarget::Ident(ident) => {
////            let var = eval_let_expr(ctx, ctx.ty(ident.id), &stmt.rhs);
////            println!("register: {:?}", ident.id);
////            ctx.register(ident.id, var);
////        }
////    }
////}
////
////fn call_stmt(ctx: &mut GenCtx, stmt: &Call) {
////    let func = ctx.declare_func(stmt.sig.ident);
////    ctx.builder.ins().call(func, &[]);
////}
////
////fn return_open(ctx: &mut GenCtx, ty: FullTy, stmt: &OpenStmt) {
////    match stmt {
////        OpenStmt::Lit(lit) => match ctx.expect_lit(lit.kind) {
////            LitKind::Int(int) => {
////                let clty = ty.expect_ty().clty();
////                let int = ctx.builder.ins().iconst(clty, int);
////                ctx.builder.ins().return_(&[int]);
////            }
////            _ => todo!(),
////        },
////        OpenStmt::Ident(ident) => {
////            return_var(ctx, ty, &ctx.var(ident.id));
////        }
////        OpenStmt::Struct(s) => {
////            let strukt = define_struct(ctx, s);
////            let addr = ctx.builder.ins().stack_addr(I64, strukt.slot, 0);
////            ctx.builder.ins().return_(&[addr]);
////        }
////        OpenStmt::Call(call) => {
////            let func = ctx.declare_func(call.sig.ident);
////            let call = ctx.builder.ins().call(func, &[]);
////            let result = ctx.builder.inst_results(call)[0];
////            ctx.builder.ins().return_(&[result]);
////        }
////        OpenStmt::Bin(bin) => {
////            let var = eval_bin_op(ctx, ty, bin);
////            return_var(ctx, ty, &var);
////        }
////    }
////}
////
////fn return_unit(ctx: &mut GenCtx, stmt: &OpenStmt) {
////    match stmt {
////        OpenStmt::Call(call) => {
////            let func = ctx.declare_func(call.sig.ident);
////            ctx.builder.ins().call(func, &[]);
////        }
////        _ => unreachable!(),
////    }
////}
////
////fn eval_struct_field(ctx: &mut GenCtx, bin: &BinOp) -> Var {
////    let mut path = Vec::new();
////    descend_bin_op_field(ctx, bin, &mut path);
////
////    let mut offset = 0;
////    let ident = *path.first().unwrap();
////    let var = ctx.var(ident);
////    let var = var.kind.expect_struct();
////    let mut strukt = ctx.structs.strukt(var.id);
////
////    for (i, ident) in path.iter().enumerate().skip(1) {
////        match strukt.field_ty(*ident) {
////            FullTy::Ty(ty) => {
////                debug_assert!(i == path.len() - 1);
////                return Var::field(
////                    *var,
////                    FullTy::Ty(ty),
////                    strukt.field_offset(ctx, *ident) + offset,
////                );
////            }
////            FullTy::Struct(s) => {
////                if i == path.len() - 1 {
////                    return Var::field(*var, FullTy::Struct(s), strukt.field_offset(ctx, *ident));
////                } else {
////                    offset += strukt.field_offset(ctx, *ident);
////                    strukt = ctx.structs.expect_struct_ident(s);
////                }
////            }
////        }
////    }
////
////    panic!("invalid field");
////}
////
////fn descend_bin_op_field(ctx: &mut GenCtx, bin: &BinOp, accesses: &mut Vec<IdentId>) {
////    if bin.kind == BinOpKind::Field {
////        match bin.lhs {
////            BinOpExpr::Ident(ident) => {
////                if let BinOpExpr::Bin(bin) = &bin.rhs {
////                    accesses.push(ident.id);
////                    descend_bin_op_field(ctx, bin, accesses);
////                } else {
////                    let BinOpExpr::Ident(other) = bin.rhs else {
////                        panic!()
////                    };
////
////                    accesses.push(other.id);
////                    accesses.push(ident.id);
////                }
////            }
////            _ => {}
////        }
////    }
////}
////
////fn return_var(ctx: &mut GenCtx, ty: FullTy, var: &Var) {
////    match &var.kind {
////        VarKind::Primitive(prim) => {
////            let val = ctx.builder.use_var(prim.clvar);
////            ctx.builder.ins().return_(&[val]);
////        }
////        VarKind::Struct(s) => {
////            let slot = s.slot;
////            let val = ctx.builder.ins().stack_addr(I64, slot, 0);
////            ctx.builder.ins().return_(&[val]);
////        }
////        VarKind::Enum(e) => {
////            let val = ctx.builder.use_var(e.clvar);
////            ctx.builder.ins().return_(&[val]);
////        }
////        VarKind::Offset(s, o) => match ty {
////            FullTy::Ty(out) => {
////                let ty = o.ty.expect_ty();
////                debug_assert_eq!(ty, out);
////                let val = ctx.builder.ins().stack_load(ty.clty(), s.slot, o.offset);
////                ctx.builder.ins().return_(&[val]);
////            }
////            FullTy::Struct(strukt) => {
////                let ty = o.ty.expect_struct();
////                debug_assert_eq!(ty, strukt);
////                let val = ctx.builder.ins().stack_addr(I64, s.slot, o.offset);
////                ctx.builder.ins().return_(&[val]);
////            }
////        },
////    }
////}
//
//impl Ty {
//    pub fn clty(&self) -> Type {
//        match self {
//            Ty::Int(kind) => match kind {
//                IntKind::I8 => I8,
//                IntKind::I16 => I16,
//                IntKind::I32 => I32,
//                IntKind::I64 => I64,
//                IntKind::U8 => I8,
//                IntKind::U16 => I16,
//                IntKind::U32 => I32,
//                IntKind::U64 => I64,
//            },
//            Ty::Unit => panic!("handle unit abi"),
//        }
//    }
//}
