use super::ctx::GenCtx;
use crate::ir::enom::{EnumDef, EnumId};
use crate::ir::ident::IdentId;
use crate::ir::lit::LitKind;
use crate::ir::strukt::{StructDef, StructId};
use crate::ir::ty::{Ty, Ty};
use crate::ir::LetExpr;
use cranelift_codegen::ir::types::I64;
use cranelift_codegen::ir::{InstBuilder, Type, Value};
use cranelift_codegen::ir::{StackSlot, StackSlotData, StackSlotKind};
use cranelift_frontend::Variable;
use cranelift_module::Module;

#[derive(Debug, Clone, Copy)]
pub struct Var {
    pub ty: Ty,
    pub kind: VarKind,
}

impl Var {
    pub fn prim(ty: Ty, prim: Prim) -> Self {
        Self {
            kind: VarKind::Primitive(prim),
            ty,
        }
    }

    pub fn slot(ty: Ty, slot: StackSlot) -> Self {
        Self {
            kind: VarKind::Slot(slot),
            ty,
        }
    }

    //pub fn enom(ty: Ty, enom: Enum) -> Self {
    //    Self {
    //        kind: VarKind::Enum(enom),
    //        ty,
    //    }
    //}
    //
    //pub fn field(strukt: Struct, ty: Ty, offset: i32) -> Self {
    //    Self {
    //        kind: VarKind::Offset(strukt, StackOffset { ty, offset }),
    //        ty,
    //    }
    //}
}

#[derive(Debug, Clone, Copy)]
pub enum VarKind {
    Slot(StackSlot),
    Primitive(Prim),
    //Struct(Struct),
    //Enum(Enum),
    ///// Not necessarily a field (e.g. struct.inner.field)
    //Offset(Struct, StackOffset),
}

impl VarKind {
    //#[track_caller]
    //pub fn expect_primitive(&self) -> &Prim {
    //    match self {
    //        Self::Primitive(prim) => prim,
    //        t => panic!("expected primitive, got: {t:?}"),
    //    }
    //}
    //
    //#[track_caller]
    //pub fn expect_struct(&self) -> &Struct {
    //    match self {
    //        Self::Struct(s) => s,
    //        t => panic!("expected struct, got: {t:?}"),
    //    }
    //}
}

#[derive(Debug, Clone, Copy)]
pub struct Prim {
    pub ty: Ty,
    pub clty: Type,
    pub clvar: Variable,
}

impl Prim {
    #[track_caller]
    pub fn new(ty: Ty, clvar: Variable) -> Self {
        Self {
            clty: ty.expect_ty().clty(),
            ty: ty.expect_ty(),
            clvar,
        }
    }

    pub fn value(&self, ctx: &mut GenCtx) -> Value {
        ctx.builder.use_var(self.clvar)
    }

    pub fn assign(&self, ctx: &mut GenCtx, other: &Self) {
        let val = other.value(ctx);
        ctx.builder.def_var(self.clvar, val);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Struct {
    pub id: StructId,
    pub slot: StackSlot,
}

#[derive(Debug, Clone, Copy)]
pub struct StackOffset {
    pub offset: i32,
    pub ty: Ty,
}

impl Struct {
    pub fn addr(&self, ctx: &mut GenCtx) -> Value {
        ctx.builder.ins().stack_addr(I64, self.slot, 0)
    }

    //pub fn field_value(&self, ctx: &mut GenCtx, field: IdentId) -> Value {
    //    let def = ctx.struct_def(self.id);
    //    let field = def.field_ty(field)
    //    ctx.builder
    //        .ins()
    //        .stack_load(ty.clty(), slot, offset + field_offset)
    //}

    pub fn assign_field(&self, ctx: &mut GenCtx, field: IdentId, var: Var) {
        //match var.kind {
        //    VarKind::Primitive(prim) => {
        //        let val = prim.value(ctx);
        //        let offset = ctx.struct_def(self.id).field_offset(ctx, field);
        //        ctx.builder.ins().stack_store(val, self.slot, offset);
        //    }
        //    VarKind::Struct(s) => {
        //        //let layout = ctx.structs.layout(s.id);
        //        //let offset = self.fields.get(&field).expect("invalid field").offset;
        //        todo!();
        //
        //        //let dest = ctx.builder.ins().stack_addr(I64, self.slot, offset);
        //        ////let src = addr;
        //        //let size = ctx.builder.ins().iconst(I64, layout.size as i64);
        //        //
        //        //ctx.builder.call_memcpy(
        //        //    TargetFrontendConfig {
        //        //        default_call_conv: CallConv::AppleAarch64,
        //        //        pointer_width: PointerWidth::U64,
        //        //        page_size_align_log2: 14,
        //        //    },
        //        //    dest,
        //        //    src,
        //        //    size,
        //        //);
        //    }
        //    _ => todo!(),
        //}
    }
}

pub fn copy_return_struct(ctx: &mut GenCtx, slot: StackSlot, strukt: StructId, addr: Value) {
    let layout = ctx.structs.layout(strukt);
    let dest = ctx.builder.ins().stack_addr(I64, slot, 0);
    let src = addr;
    let size = ctx.builder.ins().iconst(I64, layout.size as i64);
    ctx.builder
        .call_memcpy(ctx.module.target_config(), dest, src, size);
}

//pub fn copy_struct_to(ctx: &mut GenCtx, dst: &Struct, src: &Struct) {
//    let layout = ctx.structs.layout(dst.id);
//
//    let dest = ctx.builder.ins().stack_addr(I64, dst.slot, 0);
//    let src = ctx.builder.ins().stack_addr(I64, src.slot, 0);
//    let size = ctx.builder.ins().iconst(I64, layout.size as i64);
//
//    ctx.builder
//        .call_memcpy(ctx.module.target_config(), dest, src, size);
//}
//
//pub fn copy_struct_to_slot(ctx: &mut GenCtx, dst: StackSlot, src: &Struct, offset: i32) {
//    let layout = ctx.structs.layout(src.id);
//
//    let dest = ctx.builder.ins().stack_addr(I64, dst, offset);
//    let src = ctx.builder.ins().stack_addr(I64, src.slot, 0);
//    let size = ctx.builder.ins().iconst(I64, layout.size as i64);
//
//    ctx.builder
//        .call_memcpy(ctx.module.target_config(), dest, src, size);
//}
//
//pub fn define_struct(ctx: &mut GenCtx, strukt: &StructDef) -> Struct {
//    let id = ctx.expect_struct_id(strukt.name.id);
//    let new_struct = allocate_struct(ctx, id);
//    for field in strukt.fields.iter() {
//        let def = ctx.structs.strukt(id);
//        let ty = def.field_ty(field.name.id);
//        let offset = def.field_offset(ctx, field.name.id);
//        define_struct_field(ctx, ty, &field.expr, new_struct.slot, offset)
//    }
//
//    new_struct
//}
//
//fn define_struct_field(ctx: &mut GenCtx, ty: Ty, expr: &LetExpr, slot: StackSlot, offset: i32) {
//    match expr {
//        LetExpr::Lit(lit) => match ctx.expect_lit(lit.kind) {
//            LitKind::Int(int) => {
//                let val = ctx.builder.ins().iconst(ty.expect_ty().clty(), int);
//                ctx.builder.ins().stack_store(val, slot, offset);
//            }
//            _ => todo!(),
//        },
//        LetExpr::Struct(def) => {
//            let id = ctx.expect_struct_id(def.name.id);
//            for field in def.fields.iter() {
//                let ty = ctx.structs.strukt(id).field_ty(field.name.id);
//                let field_offset = ctx.structs.strukt(id).field_offset(ctx, field.name.id);
//                define_struct_field(ctx, ty, &field.expr, slot, offset + field_offset);
//            }
//        }
//        LetExpr::Call(call) => match call.sig.ty {
//            Ty::Struct(s) => {
//                todo!()
//                //let func = ctx.declare_func(call.sig.ident);
//                //let call = ctx.builder.ins().call(func, &[]);
//                //let addr = ctx.builder.inst_results(call)[0];
//                //let id = ctx.expect_struct_id(s);
//                //Var::strukt(ty, copy_return_struct(ctx, id, addr))
//            }
//            _ => todo!(),
//        },
//        _ => todo!(),
//    }
//}

pub fn allocate_struct(ctx: &mut GenCtx, strukt: StructId) -> StackSlot {
    let layout = ctx.structs.layout(strukt);
    ctx.builder.create_sized_stack_slot(StackSlotData {
        kind: StackSlotKind::ExplicitSlot,
        size: layout.size as u32,
        align_shift: layout.align_shift(),
    })
}

#[derive(Debug, Clone, Copy)]
pub struct Enum {
    pub id: EnumId,
    pub clvar: Variable,
    //pub slot: StackSlot,
}

//pub fn define_enum(ctx: &mut GenCtx, def: &EnumDef) -> Enum {
//    let id = ctx.expect_enum_id(def.name.id);
//    //let clvar = ctx.new_var();
//    ctx.builder.declare_var(clvar, I64);
//
//    let enom = ctx.enums.expect_enum_ident(def.name.id);
//    let val = enom.variant_val(ctx, def.variant.name.id);
//    let val = ctx.builder.ins().iconst(I64, val as i64);
//    ctx.builder.def_var(clvar, val);
//
//    Enum { clvar, id }
//    //let new_struct = allocate_struct(ctx, id);
//    //for field in strukt.fields.iter() {
//    //    let def = ctx.structs.strukt(id);
//    //    let ty = def.field_ty(field.name.id);
//    //    let offset = def.field_offset(ctx, field.name.id);
//    //    define_struct_field(ctx, ty, &field.expr, new_struct.slot, offset)
//    //}
//
//    //new_struct
//}
