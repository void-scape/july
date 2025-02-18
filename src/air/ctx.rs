use super::{Air, OffsetVar, Reg, Var};
use crate::ir::ctx::Ctx;
use crate::ir::ident::IdentId;
use crate::ir::sig::Sig;
use crate::ir::ty::{FullTy, IntKind, TypeKey};
use crate::ir::{self, *};
use std::collections::HashMap;
use std::ops::Deref;

/// Contains relevant context for constructing [`Air`] instructions.
///
/// To start constructing a function, use [`AirCtx::start`], and to collect the resulting
/// instructions, use [`AirCtx::finish`]. `AirCtx` reuses data structures for efficiency.
pub struct AirCtx<'a> {
    ctx: &'a Ctx<'a>,
    key: &'a TypeKey,
    instrs: Vec<Air>,
    var_index: usize,
    var_map: HashMap<(FuncHash, IdentId), Var>,
    ty_map: HashMap<Var, FullTy>,
    func: Option<FuncHash>,
}

impl<'a> AirCtx<'a> {
    pub fn new(ctx: &'a Ctx<'a>, key: &'a TypeKey) -> Self {
        Self {
            ctx,
            key,
            instrs: Vec::new(),
            var_index: 0,
            var_map: HashMap::default(),
            ty_map: HashMap::default(),
            func: None,
        }
    }

    pub fn start(&mut self, func: &ir::Func) {
        assert!(self.instrs.is_empty(), "lost function instructions");
        self.var_map.clear();
        self.ty_map.clear();
        self.func = Some(func.hash());
    }

    pub fn finish(&mut self) -> Vec<Air> {
        self.instrs.drain(..).collect()
    }

    pub fn new_var_registered(&mut self, ident: IdentId, ty: FullTy) -> Var {
        let var = self.anon_var(ty);
        self.var_map.insert(
            (
                self.func.expect("AirCtx func hash not set is `lower_func`"),
                ident,
            ),
            var,
        );
        var
    }

    pub fn anon_var(&mut self, ty: FullTy) -> Var {
        let idx = self.var_index;
        self.var_index += 1;
        self.ty_map.insert(Var(idx), ty);

        match ty {
            FullTy::Ty(ty) => {
                self.ins(Air::SAlloc(Var(idx), ty.size()));
            }
            FullTy::Struct(id) => {
                let size = self.structs.layout(id).size;
                self.ins(Air::SAlloc(Var(idx), size));
            }
        }

        Var(idx)
    }

    #[track_caller]
    pub fn expect_var(&self, ident: IdentId) -> Var {
        *self
            .var_map
            .get(&(
                self.func.expect("AirCtx func hash not set in `lower_func`"),
                ident,
            ))
            .expect("invalid var ident")
    }

    #[track_caller]
    pub fn expect_var_ty(&self, var: Var) -> FullTy {
        *self.ty_map.get(&var).expect("invalid var")
    }

    #[track_caller]
    pub fn ty(&self, ident: IdentId) -> FullTy {
        self.key.ty(
            ident,
            self.func.expect("AirCtx func hash not set in `lower_func`"),
        )
    }
}

pub const RET_REG: Reg = Reg::A;

impl AirCtx<'_> {
    pub fn ins(&mut self, instr: Air) {
        self.instrs.push(instr);
    }

    pub fn ins_set(&mut self, instrs: &[Air]) {
        self.instrs.extend(instrs.iter().copied());
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

    pub fn call(&mut self, sig: &Sig) {
        self.ins(Air::Call(*sig))
    }
}

impl<'a> Deref for AirCtx<'a> {
    type Target = Ctx<'a>;

    fn deref(&self) -> &Self::Target {
        &self.ctx
    }
}
