use super::Func;
use super::enom::EnumStore;
use super::ident::*;
use super::sig::Sig;
use super::strukt::StructId;
use super::ty::{store::TyId, store::TyStore, *};
use indexmap::IndexMap;
use pebblec_arena::BlobArena;
use pebblec_parse::annotate_snippets::Level;
use pebblec_parse::diagnostic::{Diag, Msg, Sourced};
use pebblec_parse::lex::buffer::{Buffer, Span, TokenBuffer, TokenId, TokenQuery};
use pebblec_parse::lex::source::SourceMap;
use std::collections::HashMap;

#[derive(Debug)]
pub struct Ctx<'a> {
    pub source_map: SourceMap<'a>,
    pub active_source: usize,

    pub idents: IdentStore<'a>,
    pub tys: TyStore<'a>,
    pub enums: EnumStore,
    pub funcs: Vec<Func<'a>>,
    pub arena: BlobArena,
    pub sigs: IndexMap<IdentId, &'a Sig<'a>>,
    pub impl_sigs: IndexMap<(StructId, IdentId), &'a Sig<'a>>,
}

impl<'a> PartialEq for Ctx<'a> {
    #[inline]
    fn eq(&self, other: &Ctx<'a>) -> bool {
        self.source_map == other.source_map
            && self.idents == other.idents
            && self.tys == other.tys
            && self.enums == other.enums
            && self.funcs == other.funcs
            && self.sigs == other.sigs
            && self.impl_sigs == other.impl_sigs
    }
}

impl<'a> Ctx<'a> {
    pub fn new(source_map: SourceMap<'a>) -> Self {
        Self {
            source_map,
            active_source: 0,
            idents: IdentStore::default(),
            sigs: IndexMap::default(),
            impl_sigs: IndexMap::default(),
            tys: TyStore::new(),
            enums: EnumStore::default(),
            arena: BlobArena::default(),
            funcs: Vec::new(),
        }
    }

    #[track_caller]
    pub fn report_error<S: SpannedCtx>(&self, s: S, err: impl Into<String>) -> Diag<'a> {
        let span = s.ctx_span(self);
        Diag::sourced(err.into().as_str(), Msg::error_span(span))
            .loc(std::panic::Location::caller())
    }

    #[track_caller]
    pub fn report_warn<S: SpannedCtx>(&self, s: S, err: impl Into<String>) -> Diag<'a> {
        Diag::sourced(err.into().as_str(), Msg::warn_span(s.ctx_span(self)))
            .loc(std::panic::Location::caller())
    }

    #[track_caller]
    pub fn report_note(&self, span: Span, err: impl Into<String>) -> Diag<'a> {
        Diag::sourced(err.into().as_str(), Msg::note_span(span))
            .level(Level::Note)
            .loc(std::panic::Location::caller())
    }

    #[track_caller]
    pub fn report_help<S: SpannedCtx>(&self, s: S, err: impl Into<String>) -> Diag<'a> {
        let span = s.ctx_span(self);
        Diag::sourced(err.into().as_str(), Msg::help_span(span))
            .level(Level::Help)
            .loc(std::panic::Location::caller())
    }

    #[track_caller]
    pub fn mismatch<E: CtxFmt, G: CtxFmt>(&self, span: Span, expected: E, got: G) -> Diag<'a> {
        self.report_error(
            span,
            format!(
                "mismatched types: expected `{}`, got `{}`",
                expected.to_string(self),
                got.to_string(self)
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
        todo!()
        //let mut diag = Diag::new(
        //    title,
        //    Sourced::new(self.token_buffer().source(), Level::Error),
        //);
        //for msg in msgs {
        //    diag = diag.msg(msg);
        //}
        //diag.loc(std::panic::Location::caller())
    }

    pub fn ty_str(&self, ty: TyId) -> String {
        if ty == TyId::UNIT {
            String::from("()")
        } else {
            self.tys.ty(ty).to_string(self)
        }
    }

    pub fn store_funcs(&mut self, funcs: Vec<Func<'a>>) {
        self.funcs.extend(funcs.into_iter());
    }

    #[track_caller]
    pub fn expect_struct_id(&self, id: IdentId) -> StructId {
        self.tys.expect_struct_id(id)
    }

    /// Panics if `T` requires drop
    #[track_caller]
    pub fn intern<T>(&self, item: T) -> &'a T
    where
        T: Copy,
    {
        self.arena.alloc(item)
    }

    pub fn intern_str(&self, str: &str) -> &'a str {
        self.arena.alloc_str(str)
    }

    /// Panics if `T` requires drop
    #[track_caller]
    pub fn intern_slice<T>(&self, slice: &[T]) -> &'a [T]
    where
        T: Copy,
    {
        if !slice.is_empty() {
            self.arena.alloc_slice(slice)
        } else {
            &[]
        }
    }

    pub fn build_type_layouts(&mut self) {
        let mut tys = self.tys.clone();
        tys.build_layouts(self).unwrap();
        self.tys = tys;
    }

    pub fn store_sigs(&mut self, sigs: Vec<Sig<'a>>) -> Result<(), Diag<'a>> {
        for sig in sigs.into_iter() {
            if let Some(other) = self.sigs.insert(sig.ident, self.intern(sig)) {
                return Err(self
                    .report_error(
                        sig.span,
                        format!("`{}` is already defined", sig.ident.to_string(self)),
                    )
                    .msg(Msg::help(other.span, "previously defined here")));
            }
        }

        Ok(())
    }

    pub fn store_impl_sigs(
        &mut self,
        strukt: StructId,
        sigs: Vec<Sig<'a>>,
    ) -> Result<(), Diag<'a>> {
        for sig in sigs.into_iter() {
            if let Some(other) = self.impl_sigs.insert((strukt, sig.ident), self.intern(sig)) {
                return Err(self
                    .report_error(
                        sig.span,
                        format!("`{}` is already defined", sig.ident.to_string(self)),
                    )
                    .msg(Msg::help(other.span, "previously defined here")));
            }
        }

        Ok(())
    }

    pub fn get_sig(&self, ident: IdentId) -> Option<&'a Sig<'a>> {
        self.sigs.get(&ident).copied()
    }

    pub fn get_method_sig(&self, strukt: StructId, ident: IdentId) -> Option<&'a Sig<'a>> {
        self.impl_sigs.get(&(strukt, ident)).copied()
    }

    #[track_caller]
    pub fn store_ident(&mut self, ident: TokenId) -> Ident {
        // TODO: this is a stop gap measure, there needs to be more clarity over a tokens source
        self.active_source = ident.source as usize;
        let str = self.token_buffer().as_str(ident).to_string();
        Ident {
            span: self.span(ident),
            id: self.idents.store(&str),
        }
    }

    pub fn get_ident(&'a self, id: IdentId) -> Option<&'a str> {
        self.idents.get_ident(id)
    }

    #[track_caller]
    pub fn expect_ident(&'a self, id: IdentId) -> &'a str {
        self.get_ident(id).expect("invalid ident id")
    }
}

impl<'a> Buffer<'a> for Ctx<'a> {
    #[track_caller]
    fn token_buffer(&self) -> &TokenBuffer<'a> {
        self.source_map.buffer(self.active_source)
    }
}

pub trait CtxFmt {
    fn ctx_fmt<'a>(&'a self, ctx: &'a Ctx<'a>, buf: &mut String);

    fn to_string<'a>(&'a self, ctx: &'a Ctx<'a>) -> String {
        let mut buf = String::new();
        self.ctx_fmt(ctx, &mut buf);
        buf
    }
}

impl<T> CtxFmt for &T
where
    T: CtxFmt,
{
    fn ctx_fmt<'a>(&'a self, ctx: &'a Ctx<'a>, buf: &mut String) {
        <T as CtxFmt>::ctx_fmt(self, ctx, buf)
    }
}

impl CtxFmt for &'static str {
    fn ctx_fmt<'a>(&self, _: &Ctx<'a>, buf: &mut String) {
        buf.push_str(self);
    }
}

impl CtxFmt for String {
    fn ctx_fmt<'a>(&'a self, _: &Ctx<'a>, buf: &mut String) {
        buf.push_str(&self);
    }
}

impl<'s> CtxFmt for Ty<'s> {
    fn ctx_fmt<'a>(&'a self, ctx: &'a Ctx<'a>, buf: &mut String) {
        buf.push_str(&self.to_string(ctx));
    }
}

impl CtxFmt for TyId {
    fn ctx_fmt<'a>(&self, ctx: &'a Ctx<'a>, buf: &mut String) {
        buf.push_str(&ctx.ty_str(*self))
    }
}

impl CtxFmt for Ident {
    fn ctx_fmt<'a>(&'a self, ctx: &'a Ctx<'a>, buf: &mut String) {
        buf.push_str(ctx.expect_ident(self.id));
    }
}

impl CtxFmt for IdentId {
    fn ctx_fmt<'a>(&'a self, ctx: &'a Ctx<'a>, buf: &mut String) {
        buf.push_str(ctx.expect_ident(*self));
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
    fn spanned_ctx_fmt<'a>(&'a self, ctx: &'a Ctx<'a>) -> (Span, String);
}

impl<T> SpannedCtxFmt for T
where
    T: SpannedCtx + CtxFmt,
{
    fn spanned_ctx_fmt<'a>(&'a self, ctx: &'a Ctx<'a>) -> (Span, String) {
        (
            <Self as SpannedCtx>::ctx_span(self, ctx),
            <Self as CtxFmt>::to_string(self, ctx),
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
