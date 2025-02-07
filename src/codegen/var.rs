use super::ctx::GenCtx;
use crate::ir::ident::IdentId;
use crate::ir::lit::LitKind;
use crate::ir::strukt::{StructDef, StructId};
use crate::ir::ty::{FullTy, Ty};
use crate::ir::LetExpr;
use cranelift_codegen::ir::types::I64;
use cranelift_codegen::ir::{InstBuilder, Type, Value};
use cranelift_codegen::ir::{StackSlot, StackSlotData, StackSlotKind};
use cranelift_codegen::isa::{CallConv, TargetFrontendConfig};
use cranelift_frontend::Variable;
use std::collections::HashMap;
use target_lexicon::PointerWidth;

#[derive(Debug, Clone)]
pub struct Var {
    pub ty: FullTy,
    pub kind: VarKind,
}

impl Var {
    pub fn prim(ty: FullTy, prim: Prim) -> Self {
        Self {
            kind: VarKind::Primitive(prim),
            ty,
        }
    }

    pub fn strukt(ty: FullTy, strukt: Struct) -> Self {
        Self {
            kind: VarKind::Struct(strukt),
            ty,
        }
    }
}

#[derive(Debug, Clone)]
pub enum VarKind {
    Primitive(Prim),
    Struct(Struct),
}

impl VarKind {
    #[track_caller]
    pub fn expect_primitive(&self) -> &Prim {
        match self {
            Self::Primitive(prim) => prim,
            Self::Struct(s) => panic!("expected primitive, got: {s:?}"),
        }
    }

    #[track_caller]
    pub fn expect_struct(&self) -> &Struct {
        match self {
            Self::Struct(s) => s,
            Self::Primitive(p) => panic!("expected struct, got: {p:?}"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Prim {
    pub ty: Ty,
    pub clty: Type,
    pub clvar: Variable,
}

impl Prim {
    #[track_caller]
    pub fn new(ty: FullTy, clvar: Variable) -> Self {
        Self {
            clty: ty.expect_ty().clty(),
            ty: ty.expect_ty(),
            clvar,
        }
    }

    pub fn value(&self, ctx: &mut GenCtx) -> Value {
        ctx.builder.use_var(self.clvar)
    }
}

#[derive(Debug, Clone)]
pub struct Struct {
    pub id: StructId,
    pub fields: HashMap<IdentId, StackOffset>,
    pub slot: StackSlot,
}

#[derive(Debug, Clone)]
pub struct StackOffset {
    pub offset: i32,
    pub ty: FullTy,
}

impl Struct {
    pub fn assign_field(&self, ctx: &mut GenCtx, field: IdentId, var: Var) {
        match var.kind {
            VarKind::Primitive(prim) => {
                let val = prim.value(ctx);
                ctx.builder.ins().stack_store(
                    val,
                    self.slot,
                    self.fields.get(&field).expect("invalid field").offset,
                );
            }
            VarKind::Struct(s) => {
                let layout = ctx.structs.layout(s.id);
                let offset = self.fields.get(&field).expect("invalid field").offset;

                //let dest = ctx.builder.ins().stack_addr(I64, self.slot, offset);
                ////let src = addr;
                //let size = ctx.builder.ins().iconst(I64, layout.size as i64);
                //
                //ctx.builder.call_memcpy(
                //    TargetFrontendConfig {
                //        default_call_conv: CallConv::AppleAarch64,
                //        pointer_width: PointerWidth::U64,
                //        page_size_align_log2: 14,
                //    },
                //    dest,
                //    src,
                //    size,
                //);
            }
            _ => todo!(),
        }
    }

    #[track_caller]
    pub fn field_offset(&self, field: IdentId) -> i32 {
        self.fields.get(&field).expect("invalid field").offset
    }

    #[track_caller]
    pub fn field_ty(&self, field: IdentId) -> FullTy {
        self.fields.get(&field).expect("invalid field").ty
    }
}

pub fn copy_return_struct(ctx: &mut GenCtx, strukt: StructId, addr: Value) -> Struct {
    let new_struct = allocate_struct(ctx, strukt);
    let layout = ctx.structs.layout(strukt);

    let dest = ctx.builder.ins().stack_addr(I64, new_struct.slot, 0);
    let src = addr;
    let size = ctx.builder.ins().iconst(I64, layout.size as i64);

    ctx.builder.call_memcpy(
        TargetFrontendConfig {
            default_call_conv: CallConv::AppleAarch64,
            pointer_width: PointerWidth::U64,
            page_size_align_log2: 14,
        },
        dest,
        src,
        size,
    );

    new_struct
}

pub fn define_struct(ctx: &mut GenCtx, strukt: &StructDef) -> Struct {
    let id = ctx.expect_struct_id(strukt.name.id);
    let new_struct = allocate_struct(ctx, id);
    for field in strukt.fields.iter() {
        let ty = new_struct.field_ty(field.name.id);
        let offset = new_struct.field_offset(field.name.id);
        define_struct_field(ctx, ty, &field.expr, new_struct.slot, offset)
    }

    new_struct
}

fn define_struct_field(ctx: &mut GenCtx, ty: FullTy, expr: &LetExpr, slot: StackSlot, offset: i32) {
    match expr {
        LetExpr::Lit(lit) => match ctx.expect_lit(lit.kind) {
            LitKind::Int(int) => {
                let val = ctx.builder.ins().iconst(ty.expect_ty().clty(), int);
                ctx.builder.ins().stack_store(val, slot, offset);
            }
            _ => todo!(),
        },
        LetExpr::Struct(def) => {
            let id = ctx.expect_struct_id(def.name.id);
            for field in def.fields.iter() {
                let ty = ctx.structs.strukt(id).field_ty(field.name.id);
                let field_offset = ctx.structs.strukt(id).field_offset(ctx, field.name.id);
                define_struct_field(ctx, ty, &field.expr, slot, offset + field_offset);
            }
        }
        LetExpr::Call(call) => match call.sig.ty {
            FullTy::Struct(s) => {
                todo!()
                //let func = ctx.declare_func(call.sig.ident);
                //let call = ctx.builder.ins().call(func, &[]);
                //let addr = ctx.builder.inst_results(call)[0];
                //let id = ctx.expect_struct_id(s);
                //Var::strukt(ty, copy_return_struct(ctx, id, addr))
            }
            _ => todo!(),
        },
        _ => todo!(),
    }
}

fn allocate_struct(ctx: &mut GenCtx, strukt: StructId) -> Struct {
    let layout = ctx.structs.layout(strukt);
    let slot = ctx.builder.create_sized_stack_slot(StackSlotData {
        kind: StackSlotKind::ExplicitSlot,
        size: layout.size as u32,
        align_shift: layout.align_shift(),
    });

    let field_map = ctx.structs.fields(strukt);
    let fields = field_map
        .fields
        .iter()
        .map(|(ident, (ty, offset))| {
            (
                *ident,
                StackOffset {
                    offset: *offset as i32,
                    ty: *ty,
                },
            )
        })
        .collect();

    Struct {
        id: strukt,
        slot,
        fields,
    }
}
