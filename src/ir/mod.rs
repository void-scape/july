pub mod prelude {
    use crate::lex::buffer::{Span, TokenBuffer, TokenId, TokenQuery};
    use std::collections::HashMap;

    #[derive(Debug)]
    pub struct Ctx<'a> {
        tokens: &'a TokenBuffer<'a>,
        idents: IdentStore<'a>,
        blocks: BlockStore,
        funcs: FuncStore,
        ty: TyRegistry<'a>,
    }

    impl<'a> Ctx<'a> {
        pub fn new(tokens: &'a TokenBuffer<'a>) -> Self {
            Self {
                idents: IdentStore::default(),
                blocks: BlockStore::default(),
                funcs: FuncStore::default(),
                ty: TyRegistry::default(),
                tokens,
            }
        }

        pub fn store_func(&mut self, func: Func) -> FuncId {
            self.funcs.store(func)
        }

        pub fn store_block(&mut self, block: Block) -> BlockId {
            self.blocks.store(block)
        }

        /// Panics if `ident` does not point to a [`crate::lex::kind::TokenKind::Ident`].
        #[track_caller]
        pub fn store_ident(&mut self, ident: TokenId) -> IdentId {
            self.idents.store(self.tokens.ident(ident))
        }

        /// Panics if `ty` does not point to a [`crate::lex::kind::TokenKind::Ident`].
        #[track_caller]
        pub fn ty(&self, ty: TokenId) -> Ty {
            self.ty
                .ty_str(self.tokens.ident(ty))
                .unwrap_or(Ty::Unresolved(ty))
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct IdentId(usize);

    #[derive(Debug, Default)]
    pub struct IdentStore<'a> {
        map: HashMap<&'a str, IdentId>,
        buf: Vec<&'a str>,
    }

    impl<'a> IdentStore<'a> {
        pub fn store(&mut self, ident: &'a str) -> IdentId {
            if let Some(id) = self.map.get(ident) {
                *id
            } else {
                let id = IdentId(self.buf.len());
                self.map.insert(ident, id);
                self.buf.push(ident);
                id
            }
        }

        pub fn ident(&self, id: IdentId) -> &'a str {
            self.buf.get(id.0).expect("invalid ident id")
        }
    }

    #[derive(Clone, Copy)]
    pub struct FuncId(usize);

    #[derive(Debug, Default)]
    pub struct FuncStore {
        funcs: Vec<Func>,
    }

    impl FuncStore {
        pub fn store(&mut self, func: Func) -> FuncId {
            let idx = self.funcs.len();
            self.funcs.push(func);
            FuncId(idx)
        }

        pub fn func(&self, id: FuncId) -> &Func {
            self.funcs.get(id.0).expect("invalid func id")
        }
    }

    #[derive(Debug)]
    pub struct Func {
        name: IdentId,
        params: Vec<LetInstr>,
        ret: Ty,
        block: BlockId,
    }

    impl Func {
        pub fn new(name: IdentId, params: Vec<LetInstr>, ret: Ty, block: BlockId) -> Self {
            Self {
                name,
                params,
                ret,
                block,
            }
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct BlockId(usize);

    #[derive(Debug, Default)]
    pub struct BlockStore {
        blocks: Vec<Block>,
    }

    impl BlockStore {
        pub fn store(&mut self, block: Block) -> BlockId {
            let idx = self.blocks.len();
            self.blocks.push(block);
            BlockId(idx)
        }

        pub fn block(&self, id: BlockId) -> &Block {
            self.blocks.get(id.0).expect("invalid block id")
        }
    }

    #[derive(Debug)]
    pub struct Block {
        span: Span,
        instrs: Vec<Instr>,
        ret: Ty,
    }

    impl Block {
        pub fn new(span: Span, instrs: Vec<Instr>, ret: Ty) -> Self {
            Self { span, instrs, ret }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct TyId(usize);

    #[derive(Debug)]
    pub struct TyRegistry<'a> {
        symbol_map: HashMap<&'a str, TyId>,
        ty_map: HashMap<TyId, Ty>,
        tys: Vec<&'a str>,
    }

    impl Default for TyRegistry<'_> {
        fn default() -> Self {
            let mut slf = Self {
                symbol_map: HashMap::default(),
                ty_map: HashMap::default(),
                tys: Vec::new(),
            };

            slf.register_ty("i32", Ty::Int(IntKind::I32));

            slf
        }
    }

    impl<'a> TyRegistry<'a> {
        pub fn register(&mut self, ty: &'a str) -> TyId {
            let idx = self.tys.len();
            self.tys.push(ty);
            self.symbol_map.insert(ty, TyId(idx));
            TyId(idx);
            todo!();
        }

        pub fn ty_str(&self, ty: &'a str) -> Option<Ty> {
            self.symbol_map
                .get(ty)
                .map(|id| self.ty_map.get(id).copied())?
        }

        fn register_ty(&mut self, ty_str: &'a str, ty: Ty) {
            let id = TyId(self.tys.len());
            self.ty_map.insert(id, ty);
            self.tys.push(ty_str);
            self.symbol_map.insert(ty_str, id);
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub enum Ty {
        Unresolved(TokenId),
        Void,
        Int(IntKind),
    }

    #[derive(Debug, Clone, Copy)]
    pub enum IntKind {
        I32,
    }

    #[derive(Debug)]
    pub enum Instr {
        Let(LetInstr),
        Ret(Expr),
    }

    #[derive(Debug)]
    pub struct LetInstr {
        lhs: IdentId,
        rhs: Expr,
    }

    #[derive(Debug)]
    pub enum Expr {
        //Ident(IdentId),
        Lit(i64),
    }
}
