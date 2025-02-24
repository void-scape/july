use super::enom::EnumStore;
use super::ident::*;
use super::sig::Sig;
use super::strukt::StructId;
use super::ty::{store::TyId, store::TyStore, *};
use super::Func;
use crate::arena::BlobArena;
use crate::diagnostic::{Diag, Msg, Sourced};
use crate::lex::buffer::{Buffer, Span, TokenBuffer, TokenId, TokenQuery};
use annotate_snippets::Level;
use std::collections::{HashMap, HashSet};

#[derive(Debug)]
pub struct Ctx<'a> {
    pub tokens: &'a TokenBuffer<'a>,
    pub idents: IdentStore<'a>,
    pub tys: TyStore,
    pub enums: EnumStore,
    pub funcs: Vec<Func<'a>>,
    arena: BlobArena,
    sigs: HashMap<IdentId, &'a Sig<'a>>,
}

impl<'a> Ctx<'a> {
    pub fn new(tokens: &'a TokenBuffer<'a>) -> Self {
        Self {
            idents: IdentStore::default(),
            sigs: HashMap::default(),
            tys: TyStore::default(),
            enums: EnumStore::default(),
            arena: BlobArena::default(),
            funcs: Vec::new(),
            tokens,
        }
    }

    #[track_caller]
    pub fn report_error<S: SpannedCtx>(&self, s: S, err: impl Into<String>) -> Diag<'a> {
        Diag::sourced(
            err.into().as_str(),
            self.tokens.source(),
            Msg::error(s.ctx_span(self), ""),
        )
        .loc(std::panic::Location::caller())
    }

    #[track_caller]
    pub fn report_note(&self, span: Span, err: impl Into<String>) -> Diag<'a> {
        Diag::sourced(
            err.into().as_str(),
            self.tokens.source(),
            Msg::note(span, ""),
        )
        .level(Level::Note)
        .loc(std::panic::Location::caller())
    }

    #[track_caller]
    pub fn report_help<S: SpannedCtx>(&self, s: S, err: impl Into<String>) -> Diag<'a> {
        Diag::sourced(
            err.into().as_str(),
            self.tokens.source(),
            Msg::help(s.ctx_span(self), ""),
        )
        .level(Level::Help)
        .loc(std::panic::Location::caller())
    }

    #[track_caller]
    pub fn mismatch<E: CtxFmt, G: CtxFmt>(&self, span: Span, expected: E, got: G) -> Diag<'a> {
        self.report_error(
            span,
            format!(
                "mismatched types: expected `{}`, got `{}`",
                expected.ctx_fmt(self),
                got.ctx_fmt(self)
            ),
        )
        .loc(std::panic::Location::caller())
    }

    #[track_caller]
    pub fn undeclared<U: SpannedCtxFmt>(&self, u: U) -> Diag<'a> {
        let (span, str) = u.spanned_ctx_fmt(self);
        self.report_error(span, format!("`{}` is not declared", str))
            .loc(std::panic::Location::caller())
    }

    #[track_caller]
    pub fn errors(
        &self,
        title: impl Into<String>,
        msgs: impl IntoIterator<Item = Msg>,
    ) -> Diag<'a> {
        let mut diag = Diag::new(title, Sourced::new(self.tokens.source(), Level::Error));
        for msg in msgs {
            diag = diag.msg(msg);
        }
        diag.loc(std::panic::Location::caller())
    }

    pub fn ty_str(&'a self, ty: TyId) -> &'a str {
        self.tys.ty(ty).as_str(self)
    }

    pub fn store_funcs(&mut self, funcs: Vec<Func<'a>>) {
        self.funcs.extend(funcs.into_iter());
    }

    pub fn struct_name(&self, id: StructId) -> &str {
        let strukt = self.tys.strukt(id);
        self.expect_ident(strukt.name.id)
    }

    #[track_caller]
    pub fn expect_struct_id(&self, id: IdentId) -> StructId {
        self.tys.expect_struct_id(id)
    }

    /// Panics if `T` requires drop
    #[track_caller]
    pub fn intern<T>(&self, item: T) -> &'a T {
        self.arena.alloc(item)
    }

    /// Panics if `T` requires drop
    #[track_caller]
    pub fn intern_slice<T>(&self, slice: &[T]) -> &'a [T] {
        if !slice.is_empty() {
            self.arena.alloc_slice(slice)
        } else {
            &[]
        }
    }

    //pub fn store_enums(&mut self, enums: Vec<Enum>) {
    //    for enom in enums.into_iter() {
    //        self.enums.store(enom);
    //    }
    //}

    //#[track_caller]
    //pub fn expect_enum_id(&self, id: IdentId) -> EnumId {
    //    self.enums.expect_enum_id(id)
    //}

    pub fn build_type_layouts(&mut self) {
        let mut tys = self.tys.clone();
        tys.build_layouts(self).unwrap();
        self.tys = tys;

        //todo!();
        //let (layouts, variants) = self.enums.build_layouts(self).unwrap();
        //self.enums.layouts = layouts;
        //self.enums.variants = variants;
    }

    pub fn store_sigs(&mut self, sigs: Vec<Sig<'a>>) -> Result<(), Diag<'a>> {
        let mut set = HashSet::<&Sig>::default();
        for sig in sigs.iter() {
            if !set.insert(sig) {
                return Err(self.errors(
                    "duplicate function names",
                    [Msg::error(sig.span, "Function is already defined")],
                ));
            }
        }

        for sig in sigs.into_iter() {
            self.sigs.insert(sig.ident, self.intern(sig));
        }

        Ok(())
    }

    pub fn get_sig(&self, ident: IdentId) -> Option<&'a Sig<'a>> {
        self.sigs.get(&ident).copied()
    }

    #[track_caller]
    pub fn store_ident(&mut self, ident: TokenId) -> Ident {
        Ident {
            span: self.span(ident),
            id: self.idents.store(self.tokens.ident(ident)),
        }
    }

    pub fn get_ident(&self, id: IdentId) -> Option<&'a str> {
        self.idents.get_ident(id)
    }

    #[track_caller]
    pub fn expect_ident(&self, id: IdentId) -> &'a str {
        self.get_ident(id).expect("invalid ident id")
    }
}

impl<'a> Buffer<'a> for Ctx<'a> {
    fn token_buffer(&'a self) -> &'a TokenBuffer<'a> {
        self.tokens
    }
}

pub trait CtxFmt {
    fn ctx_fmt<'a>(&'a self, ctx: &'a Ctx<'a>) -> &'a str;
}

impl<T> CtxFmt for &T
where
    T: CtxFmt,
{
    fn ctx_fmt<'a>(&'a self, ctx: &'a Ctx<'a>) -> &'a str {
        <T as CtxFmt>::ctx_fmt(self, ctx)
    }
}

impl CtxFmt for &'static str {
    fn ctx_fmt<'a>(&self, _: &Ctx<'a>) -> &'a str {
        self
    }
}

impl CtxFmt for String {
    fn ctx_fmt<'a>(&'a self, _: &Ctx<'a>) -> &'a str {
        &self
    }
}

impl CtxFmt for Ty {
    fn ctx_fmt<'a>(&'a self, ctx: &'a Ctx<'a>) -> &'a str {
        self.as_str(ctx)
    }
}

impl CtxFmt for TyId {
    fn ctx_fmt<'a>(&self, ctx: &'a Ctx<'a>) -> &'a str {
        ctx.ty_str(*self)
    }
}

impl CtxFmt for Ident {
    fn ctx_fmt<'a>(&'a self, ctx: &'a Ctx<'a>) -> &'a str {
        ctx.expect_ident(self.id)
    }
}

impl CtxFmt for IdentId {
    fn ctx_fmt<'a>(&'a self, ctx: &'a Ctx<'a>) -> &'a str {
        ctx.expect_ident(*self)
    }
}

pub trait SpannedCtx {
    fn ctx_span(&self, ctx: &Ctx) -> Span;
}

impl<T> SpannedCtx for &T
where
    T: SpannedCtx,
{
    fn ctx_span(&self, ctx: &Ctx) -> Span {
        <T as SpannedCtx>::ctx_span(self, ctx)
    }
}

impl SpannedCtx for TokenId {
    fn ctx_span(&self, ctx: &Ctx) -> Span {
        ctx.span(*self)
    }
}

impl SpannedCtx for Span {
    fn ctx_span(&self, _: &Ctx) -> Span {
        *self
    }
}

impl SpannedCtx for Ident {
    fn ctx_span(&self, _: &Ctx) -> Span {
        self.span
    }
}

pub trait SpannedCtxFmt: SpannedCtx + CtxFmt {
    fn spanned_ctx_fmt<'a>(&'a self, ctx: &'a Ctx<'a>) -> (Span, &'a str);
}

impl<T> SpannedCtxFmt for T
where
    T: SpannedCtx + CtxFmt,
{
    fn spanned_ctx_fmt<'a>(&'a self, ctx: &'a Ctx<'a>) -> (Span, &'a str) {
        (
            <Self as SpannedCtx>::ctx_span(self, ctx),
            <Self as CtxFmt>::ctx_fmt(self, ctx),
        )
    }
}

#[macro_export]
macro_rules! err {
    ($ctx:expr, $span:expr, $msg:expr) => {{
        use crate::ir::ctx::SpannedCtx;
        $ctx.report_error(
            ($span).ctx_span($ctx),
            $msg,
        )
    }};
    ($ctx:expr, $span:expr, $fmt:expr, $($args:expr,)*) => {{
        use crate::ir::ctx::{SpannedCtx, CtxFmt};
        $ctx.report_error(
            ($span).ctx_span($ctx),
            format!($fmt, $($args.ctx_fmt($ctx)),*)
        )
    }};
}

#[macro_export]
macro_rules! note {
    ($ctx:expr, $span:expr, $msg:expr) => {{
        use crate::ir::ctx::SpannedCtx;
        $ctx.report_note(
            ($span).ctx_span($ctx),
            $msg,
        )
    }};
    ($ctx:expr, $span:expr, $fmt:expr, $($args:expr,)*) => {{
        use crate::ir::ctx::{SpannedCtx, CtxFmt};
        $ctx.report_note(
            ($span).ctx_span($ctx),
            format!($fmt, $($args.ctx_fmt($ctx)),*)
        )
    }};
}

#[macro_export]
macro_rules! help {
    ($ctx:expr, $span:expr, $msg:expr) => {{
        use crate::ir::ctx::SpannedCtx;
        $ctx.report_help(
            ($span).ctx_span($ctx),
            $msg,
        )
    }};
    ($ctx:expr, $span:expr, $fmt:expr, $($args:expr,)*) => {{
        use crate::ir::ctx::{SpannedCtx, CtxFmt};
        $ctx.report_help(
            ($span).ctx_span($ctx),
            format!($fmt, $($args.ctx_fmt($ctx)),*)
        )
    }};
}
