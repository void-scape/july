use crate::ir::ctx::Ctx;
use crate::ir::ident::IdentId;
use crate::ir::lit::LitKind;
use crate::ir::sig::Sig;
use crate::ir::ty::{FullTy, IntKind, Ty, TypeKey};
use crate::ir::*;
use std::collections::HashMap;
use std::ops::Deref;

#[derive(Debug)]
pub struct AirFunc<'a> {
    pub func: &'a Func,
    pub instrs: Vec<Air>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Var(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reg {
    A,
    B,
}

//impl Reg {
//    pub fn other(&self) -> Self {
//        match self {
//            Self::A => Self::B,
//            Self::B => Self::A,
//        }
//    }
//}

#[derive(Debug, Clone, Copy)]
pub enum Air {
    Ret,
    Call(Sig),

    /// Swap the A and B registers
    SwapReg,
    MovIVar(Reg, Var),
    MovIConst(Reg, i64),

    PushIConst(Var, IntKind, i64),
    PushIReg {
        dst: Var,
        kind: IntKind,
        src: Reg,
    },
    PushIVar {
        dst: Var,
        kind: IntKind,
        src: Var,
    },

    /// Add registers A and B together, store result in A.
    AddAB,
    /// Multiply registers A and B together, store result in A.
    MulAB,
    /// Subtract registers A and B together, store result in A.
    SubAB,
}

struct AirCtx<'a> {
    ctx: &'a Ctx<'a>,
    key: &'a TypeKey,
    instrs: Vec<Air>,
    var_index: usize,
    var_map: HashMap<IdentId, Var>,
    ty_map: HashMap<Var, FullTy>,
    func: FuncHash,
}

impl<'a> Deref for AirCtx<'a> {
    type Target = Ctx<'a>;

    fn deref(&self) -> &Self::Target {
        &self.ctx
    }
}

impl AirCtx<'_> {
    pub fn new_var_registered(&mut self, ident: IdentId, ty: FullTy) -> Var {
        let var = self.anon_var(ty);
        self.var_map.insert(ident, var);
        var
    }

    pub fn anon_var(&mut self, ty: FullTy) -> Var {
        let idx = self.var_index;
        self.var_index += 1;
        self.ty_map.insert(Var(idx), ty);
        Var(idx)
    }

    #[track_caller]
    pub fn expect_var(&self, ident: IdentId) -> Var {
        *self.var_map.get(&ident).expect("invalid var ident")
    }

    #[track_caller]
    pub fn expect_var_ty(&self, var: Var) -> FullTy {
        *self.ty_map.get(&var).expect("invalid var")
    }

    #[track_caller]
    pub fn ty(&self, ident: IdentId) -> FullTy {
        self.key.ty(ident, self.func)
    }
}

impl AirCtx<'_> {
    pub fn ins(&mut self, instr: Air) {
        self.instrs.push(instr);
    }

    pub fn ins_set(&mut self, instrs: &[Air]) {
        self.instrs.extend(instrs.iter().copied());
    }

    pub fn ret_iconst(&mut self, val: i64) {
        self.ins(Air::MovIConst(Reg::A, val));
        self.ins(Air::Ret);
    }

    pub fn ret_ivar(&mut self, var: Var) {
        let ty = self.expect_var_ty(var);
        match ty {
            FullTy::Ty(ty) => match ty {
                Ty::Int(_) => {
                    self.ins(Air::MovIVar(Reg::A, var));
                    self.ins(Air::Ret);
                }
                Ty::Unit => todo!(),
            },
            FullTy::Struct(_) => todo!(),
        }
    }
}

pub fn lower_func<'a>(ctx: &Ctx, key: &TypeKey, func: &'a Func) -> AirFunc<'a> {
    let mut ctx = AirCtx {
        ctx,
        key,
        instrs: Vec::new(),
        var_index: 0,
        var_map: HashMap::default(),
        ty_map: HashMap::default(),
        func: func.hash(),
    };

    for stmt in func.block.stmts.iter() {
        match stmt {
            Stmt::Semi(stmt) => match stmt {
                SemiStmt::Let(let_) => air_let_stmt(&mut ctx, let_),
                SemiStmt::Assign(assign) => air_assign_stmt(&mut ctx, assign),
                stmt => todo!("{stmt:#?}"),
            },
            Stmt::Open(stmt) => panic!("{stmt:#?}"),
        }
    }

    if let Some(end) = &func.block.end {
        air_return(&mut ctx, func.sig.ty, end);
    } else {
        ctx.ins(Air::Ret);
    }

    AirFunc {
        func,
        instrs: ctx.instrs,
    }
}

fn air_let_stmt(ctx: &mut AirCtx, stmt: &Let) {
    match stmt.lhs {
        LetTarget::Ident(ident) => match &stmt.rhs {
            LetExpr::Lit(lit) => match ctx.expect_lit(lit.kind) {
                LitKind::Int(int) => {
                    let var = ctx.new_var_registered(ident.id, ctx.ty(ident.id));
                    let ty = ctx.ty(ident.id).expect_ty();
                    match ty {
                        Ty::Int(kind) => {
                            ctx.ins(Air::PushIConst(var, kind, int));
                        }
                        Ty::Unit => unreachable!(),
                    }
                }
                _ => todo!(),
            },
            LetExpr::Bin(bin) => {
                if bin.kind.is_primitive() {
                    let var = ctx.new_var_registered(ident.id, ctx.ty(ident.id));
                    let kind = match ctx.ty(ident.id) {
                        FullTy::Ty(ty) => match ty {
                            Ty::Int(kind) => kind,
                            Ty::Unit => panic!(),
                        },
                        FullTy::Struct(_) => panic!(),
                    };
                    eval_prim_bin_op(ctx, var, kind, bin);
                } else {
                    todo!()
                }
            }
            //LetExpr::Struct(def) => {
            //    let ty = FullTy::Struct(def.id);
            //    let var = ctx.new_var_registered(ident.id, ty);
            //    define_struct(ctx, var, def);
            //}
            //LetExpr::Call(Call { sig, .. }) => {
            //    if sig.ty.is_unit() {
            //        todo!()
            //    } else {
            //        let var = ctx.new_var_registered(ident.id, sig.ty);
            //        ctx.ins(Air::CallAssign(var, *sig));
            //        //match sig.ty {
            //        //    FullTy::Struct(id) => {
            //        //        panic!();
            //        //    }
            //        //    _ => todo!()
            //        //}
            //    }
            //}
            _ => todo!(),
        },
    }
}

fn air_assign_stmt(ctx: &mut AirCtx, stmt: &Assign) {
    println!("{stmt:#?}");
    match stmt.lhs {
        AssignTarget::Ident(ident) => match &stmt.rhs {
            AssignExpr::Lit(lit) => match ctx.expect_lit(lit.kind) {
                LitKind::Int(int) => {
                    let var = ctx.expect_var(ident.id);
                    let ty = ctx.ty(ident.id).expect_ty();
                    match stmt.kind {
                        AssignKind::Add => match ty {
                            Ty::Int(kind) => {
                                Add(kind).visit_leaf(ctx, var, var, int);
                            }
                            Ty::Unit => unreachable!(),
                        },
                        AssignKind::Equals => match ty {
                            Ty::Int(kind) => {
                                ctx.ins(Air::PushIConst(var, kind, int));
                            }
                            Ty::Unit => unreachable!(),
                        },
                    }
                }
                _ => todo!(),
            },
            AssignExpr::Bin(bin) => {
                if bin.kind.is_primitive() {
                    let var = ctx.expect_var(ident.id);
                    let kind = match ctx.ty(ident.id) {
                        FullTy::Ty(ty) => match ty {
                            Ty::Int(kind) => kind,
                            Ty::Unit => panic!(),
                        },
                        FullTy::Struct(_) => panic!(),
                    };
                    match stmt.kind {
                        AssignKind::Equals => {
                            eval_prim_bin_op(ctx, var, kind, bin);
                        }
                        AssignKind::Add => {
                            let rhs = ctx.anon_var(ctx.ty(ident.id));
                            eval_prim_bin_op(ctx, rhs, kind, bin);
                            Add(kind).visit_leaf(ctx, var, var, rhs);
                        }
                    }
                } else {
                    todo!()
                }
            }
            _ => todo!(),
        },
        _ => todo!(),
    }
}

fn eval_prim_bin_op(ctx: &mut AirCtx, dst: Var, kind: IntKind, bin: &BinOp) {
    println!("{bin:#?}");
    assert!(bin.kind.is_primitive());
    match bin.kind {
        BinOpKind::Add => Add(kind).visit(ctx, kind, dst, bin),
        BinOpKind::Mul => Mul(kind).visit(ctx, kind, dst, bin),
        BinOpKind::Sub => Sub(kind).visit(ctx, kind, dst, bin),
        _ => todo!(),
    }
}

trait BinOpVisitor
where
    Self: AllBinOpLeaves,
{
    fn visit(&self, ctx: &mut AirCtx, kind: IntKind, dst: Var, bin: &BinOp) {
        match &bin.lhs {
            BinOpExpr::Lit(lit) => match ctx.expect_lit(lit.kind) {
                LitKind::Int(lhs_lit) => match &bin.rhs {
                    BinOpExpr::Lit(lit) => match ctx.expect_lit(lit.kind) {
                        LitKind::Int(rhs_lit) => {
                            self.visit_leaf(ctx, dst, lhs_lit, rhs_lit);
                        }
                        _ => todo!(),
                    },
                    BinOpExpr::Bin(bin) => {
                        eval_prim_bin_op(ctx, dst, kind, bin);
                        self.visit_leaf(ctx, dst, lhs_lit, dst);
                    }
                    BinOpExpr::Call(Call { sig, .. }) => {
                        assert!(sig.ty.is_ty_and(|ty| match ty {
                            Ty::Int(k) => {
                                *k == kind
                            }
                            _ => false,
                        }));
                        ctx.ins(Air::Call(*sig));
                        self.visit_leaf(ctx, dst, lhs_lit, Reg::A);
                    }
                    BinOpExpr::Ident(ident) => {
                        let rhs = ctx.expect_var(ident.id);
                        self.visit_leaf(ctx, dst, lhs_lit, rhs);
                    }
                },
                _ => todo!(),
            },
            BinOpExpr::Ident(ident) => {
                let var = ctx.expect_var(ident.id);
                match &bin.rhs {
                    BinOpExpr::Lit(lit) => match ctx.expect_lit(lit.kind) {
                        LitKind::Int(lit) => {
                            self.visit_leaf(ctx, dst, var, lit);
                        }
                        _ => todo!(),
                    },
                    BinOpExpr::Bin(bin) => {
                        eval_prim_bin_op(ctx, dst, kind, bin);
                        self.visit_leaf(ctx, dst, var, dst);
                    }
                    BinOpExpr::Ident(ident) => {
                        self.visit_leaf(ctx, dst, var, ctx.expect_var(ident.id));
                    }
                    _ => todo!(),
                }
            }
            BinOpExpr::Call(Call { sig, .. }) => {
                assert!(sig.ty.is_ty_and(|ty| match ty {
                    Ty::Int(k) => {
                        *k == kind
                    }
                    _ => false,
                }));
                match &bin.rhs {
                    BinOpExpr::Lit(lit) => match ctx.expect_lit(lit.kind) {
                        LitKind::Int(lit) => {
                            ctx.ins(Air::Call(*sig));
                            self.visit_leaf(ctx, dst, Reg::A, lit);
                        }
                        _ => todo!(),
                    },
                    BinOpExpr::Bin(bin) => {
                        eval_prim_bin_op(ctx, dst, kind, bin);
                        ctx.ins(Air::Call(*sig));
                        self.visit_leaf(ctx, dst, Reg::A, dst);
                    }
                    BinOpExpr::Ident(ident) => {
                        ctx.ins(Air::Call(*sig));
                        self.visit_leaf(ctx, dst, Reg::A, ctx.expect_var(ident.id));
                    }
                    _ => todo!(),
                }
            }
            _ => todo!(),
        }
    }
}

trait BinOpLeaf<L, R> {
    fn visit_leaf(&self, ctx: &mut AirCtx, dst: Var, lhs: L, rhs: R);
}

trait AllBinOpLeaves:
    BinOpLeaf<i64, i64>
    + BinOpLeaf<i64, Var>
    + BinOpLeaf<Var, i64>
    + BinOpLeaf<Var, Var>
    + BinOpLeaf<Reg, Var>
    + BinOpLeaf<Var, Reg>
    + BinOpLeaf<Reg, i64>
    + BinOpLeaf<i64, Reg>
{
}

crate::impl_prim_bin_op_visitor!(Add, AddAB, +);
crate::impl_prim_bin_op_visitor!(Sub, SubAB, -);
crate::impl_prim_bin_op_visitor!(Mul, MulAB, *);

fn air_return(ctx: &mut AirCtx, ty: FullTy, end: &OpenStmt) {
    if ty.is_unit() {
        todo!()
    } else {
        match end {
            OpenStmt::Lit(lit) => match ctx.expect_lit(lit.kind) {
                LitKind::Int(int) => {
                    ctx.ret_iconst(int);
                }
                _ => todo!(),
            },
            OpenStmt::Ident(ident) => {
                let var = ctx.expect_var(ident.id);
                let out_ty = ctx.expect_var_ty(var);
                assert_eq!(out_ty, ty);

                match out_ty {
                    FullTy::Ty(ty) => match ty {
                        Ty::Int(_) => ctx.ret_ivar(var),
                        Ty::Unit => unreachable!(),
                    },
                    _ => todo!(),
                }
            }
            OpenStmt::Bin(bin) => {
                if bin.kind.is_primitive() {
                    match ty {
                        FullTy::Ty(ty) => match ty {
                            Ty::Int(kind) => {
                                let dst = ctx.anon_var(FullTy::Ty(ty));
                                eval_prim_bin_op(ctx, dst, kind, bin);
                                ctx.ret_ivar(dst);
                            }
                            Ty::Unit => unreachable!(),
                        },
                        FullTy::Struct(_) => unreachable!(),
                    }
                } else {
                    todo!();
                }
            }
            OpenStmt::Call(Call { sig, .. }) => {
                if ty.is_primitive() {
                    assert_eq!(sig.ty, ty);
                    ctx.ins(Air::Call(*sig));
                    ctx.ins(Air::Ret);
                } else {
                    todo!();
                }
            }
            //OpenStmt::Ident(ident) => match ctx.ty(ident.id) {
            //    FullTy::Struct(_) => ctx.ins(Air::Ret(Some(Val::Ptr(ctx.expect_var(ident.id))))),
            //    _ => todo!(),
            //},
            //OpenStmt::Bin(bin) => {
            //    let out = ctx.new_var_with_ty(ty);
            //    eval_bin_op(ctx, bin, out);
            //    ctx.ins(Air::Ret(Some(Val::Owned(out))));
            //}
            _ => todo!(),
        }
    }
}

//fn eval_bin_op(ctx: &mut AirCtx, bin: &BinOp, dst: Var) {
//    match bin.kind {
//        BinOpKind::Field => eval_struct_field(ctx, bin, dst),
//        _ => {
//            let lhs = eval_bin_op_expr(ctx, &bin.lhs, dst);
//            let rhs = eval_bin_op_expr(ctx, &bin.rhs, dst);
//
//            match bin.kind {
//                _ => todo!(),
//            }
//        }
//    }
//}
//
//fn eval_bin_op_expr(ctx: &mut AirCtx, expr: &BinOpExpr, dst: Var) {
//    match expr {
//        BinOpExpr::Ident(ident) => {
//            todo!()
//        }
//        BinOpExpr::Lit(lit) => {
//            todo!()
//        }
//        BinOpExpr::Call(call) => {
//            todo!()
//        }
//        BinOpExpr::Bin(bin) => eval_bin_op(ctx, bin, dst),
//    }
//}
//
//fn eval_struct_field(ctx: &mut AirCtx, bin: &BinOp, dst: Var) {
//    let mut path = Vec::new();
//    descend_bin_op_field(ctx, bin, &mut path);
//
//    let mut offset = 0;
//    let ident = *path.first().unwrap();
//    let var = ctx.expect_var(ident);
//    match ctx.expect_var_ty(var) {
//        FullTy::Struct(id) => {
//            let mut strukt = ctx.structs.strukt(id);
//            for (i, ident) in path.iter().enumerate().skip(1) {
//                match strukt.field_ty(*ident) {
//                    FullTy::Ty(ty) => {
//                        debug_assert!(i == path.len() - 1);
//                        let offset = strukt.field_offset(ctx, *ident) + offset;
//                        let src = OffsetVar::new(var, offset);
//                        let dst = OffsetVar::zero(dst);
//                        ctx.ins(Air::Load {
//                            src,
//                            dst,
//                            ty: FullTy::Ty(ty),
//                        });
//                        return;
//                        //return var;
//                        //return Var::field(
//                        //    *var,
//                        //    FullTy::Ty(ty),
//                        //    strukt.field_offset(ctx, *ident) + offset,
//                        //);
//                    }
//                    FullTy::Struct(s) => {
//                        if i == path.len() - 1 {
//                            let offset = strukt.field_offset(ctx, *ident) + offset;
//                            let src = OffsetVar::new(var, offset);
//                            let dst = OffsetVar::zero(dst);
//                            ctx.ins(Air::Load {
//                                src,
//                                dst,
//                                ty: FullTy::Struct(s),
//                            });
//                            return;
//                            //return Var::field(*var, FullTy::Struct(s), strukt.field_offset(ctx, *ident));
//                        } else {
//                            offset += strukt.field_offset(ctx, *ident);
//                            strukt = ctx.structs.strukt(s);
//                        }
//                    }
//                }
//            }
//
//            panic!("invalid field");
//        }
//        FullTy::Ty(_) => {
//            panic!("invalid field access")
//        }
//    }
//}
//
//fn descend_bin_op_field(ctx: &mut AirCtx, bin: &BinOp, accesses: &mut Vec<IdentId>) {
//    if bin.kind == BinOpKind::Field {
//        match bin.lhs {
//            BinOpExpr::Ident(ident) => {
//                if let BinOpExpr::Bin(bin) = &bin.rhs {
//                    accesses.push(ident.id);
//                    descend_bin_op_field(ctx, bin, accesses);
//                } else {
//                    let BinOpExpr::Ident(other) = bin.rhs else {
//                        panic!()
//                    };
//
//                    accesses.push(other.id);
//                    accesses.push(ident.id);
//                }
//            }
//            _ => {}
//        }
//    }
//}
//
//fn define_struct(ctx: &mut AirCtx, var: Var, def: &StructDef) {
//    for field in def.fields.iter() {
//        let decl = ctx.structs.strukt(def.id);
//        let field_id = field.name.id;
//        let ty = decl.field_ty(field_id);
//        let offset = decl.field_offset(ctx, field_id);
//        define_struct_field(ctx, var, ty, &field.expr, offset);
//    }
//}
//
//fn define_struct_field(ctx: &mut AirCtx, var: Var, ty: FullTy, expr: &LetExpr, offset: Offset) {
//    match ty {
//        FullTy::Ty(ty) => match expr {
//            LetExpr::Lit(lit) => {
//                ctx.ins(Air::Store(var, offset, Val::Lit(ty, lit.kind)));
//            }
//            _ => todo!(),
//        },
//        FullTy::Struct(id) => match &expr {
//            LetExpr::Struct(def) => {
//                assert_eq!(def.id, id);
//
//                for field in def.fields.iter() {
//                    let decl = ctx.structs.strukt(def.id);
//                    let field_id = field.name.id;
//                    let ty = decl.field_ty(field_id);
//                    let field_offset = decl.field_offset(ctx, field_id);
//                    define_struct_field(ctx, var, ty, &field.expr, offset + field_offset);
//                }
//            }
//            _ => todo!(),
//        },
//    }
//}

#[macro_export]
macro_rules! impl_prim_bin_op_visitor {
    ($name:ident, $instr:ident, $op:tt) => {
        struct $name(IntKind);

        impl BinOpVisitor for $name {}

        impl AllBinOpLeaves for $name {}

        impl BinOpLeaf<i64, i64> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: Var, lhs: i64, rhs: i64) {
                ctx.ins(Air::PushIConst(dst, self.0, lhs $op rhs));
            }
        }

        impl BinOpLeaf<i64, Var> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: Var, lhs: i64, rhs: Var) {
                ctx.ins_set(&[
                    Air::MovIVar(Reg::A, rhs),
                    Air::MovIConst(Reg::B, lhs),
                    Air::$instr,
                    Air::PushIReg {
                        dst,
                        kind: self.0,
                        src: Reg::A,
                    },
                ]);
            }
        }

        impl BinOpLeaf<Var, i64> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: Var, lhs: Var, rhs: i64) {
                ctx.ins_set(&[
                    Air::MovIVar(Reg::A, lhs),
                    Air::MovIConst(Reg::B, rhs),
                    Air::$instr,
                    Air::PushIReg {
                        dst,
                        kind: self.0,
                        src: Reg::A,
                    },
                ]);
            }
        }

        impl BinOpLeaf<Var, Var> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: Var, lhs: Var, rhs: Var) {
                ctx.ins_set(&[
                    Air::MovIVar(Reg::A, rhs),
                    Air::MovIVar(Reg::B, lhs),
                    Air::$instr,
                    Air::PushIReg {
                        dst,
                        kind: self.0,
                        src: Reg::A,
                    },
                ]);
            }
        }

        impl BinOpLeaf<Reg, i64> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: Var, lhs: Reg, rhs: i64) {
                if lhs == Reg::A {
                    ctx.ins_set(&[
                        Air::SwapReg,
                        Air::MovIConst(Reg::A, rhs),
                        Air::$instr,
                        Air::PushIReg {
                            dst,
                            kind: self.0,
                            src: Reg::A,
                        },
                    ]);
                } else {
                    ctx.ins_set(&[
                        Air::MovIConst(Reg::A, rhs),
                        Air::$instr,
                        Air::PushIReg {
                            dst,
                            kind: self.0,
                            src: Reg::A,
                        },
                    ]);
                }
            }
        }

        impl BinOpLeaf<i64, Reg> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: Var, lhs: i64, rhs: Reg) {
                if rhs == Reg::B {
                    ctx.ins_set(&[
                        Air::SwapReg,
                        Air::MovIConst(Reg::B, lhs),
                        Air::$instr,
                        Air::PushIReg {
                            dst,
                            kind: self.0,
                            src: Reg::A,
                        },
                    ]);
                } else {
                    ctx.ins_set(&[
                        Air::MovIConst(Reg::B, lhs),
                        Air::$instr,
                        Air::PushIReg {
                            dst,
                            kind: self.0,
                            src: Reg::A,
                        },
                    ]);
                }
            }
        }

        impl BinOpLeaf<Reg, Var> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: Var, lhs: Reg, rhs: Var) {
                if lhs == Reg::A {
                    ctx.ins_set(&[
                        Air::SwapReg,
                        Air::MovIVar(Reg::A, rhs),
                        Air::$instr,
                        Air::PushIReg {
                            dst,
                            kind: self.0,
                            src: Reg::A,
                        },
                    ]);
                } else {
                    ctx.ins_set(&[
                        Air::MovIVar(Reg::A, rhs),
                        Air::$instr,
                        Air::PushIReg {
                            dst,
                            kind: self.0,
                            src: Reg::A,
                        },
                    ]);
                }
            }
        }

        impl BinOpLeaf<Var, Reg> for $name {
            fn visit_leaf(&self, ctx: &mut AirCtx, dst: Var, lhs: Var, rhs: Reg) {
                if rhs == Reg::B {
                    ctx.ins_set(&[
                        Air::SwapReg,
                        Air::MovIVar(Reg::B, lhs),
                        Air::$instr,
                        Air::PushIReg {
                            dst,
                            kind: self.0,
                            src: Reg::A,
                        },
                    ]);
                } else {
                    ctx.ins_set(&[
                        Air::MovIVar(Reg::B, lhs),
                        Air::$instr,
                        Air::PushIReg {
                            dst,
                            kind: self.0,
                            src: Reg::A,
                        },
                    ]);
                }
            }
        }
    }
}
