use super::sig::Sig;
use super::ty::{store::TyStore, *};
use super::{Const, Func};
use indexmap::IndexMap;
use pebblec_arena::BlobArena;
use pebblec_parse::annotate_snippets::Level;
use pebblec_parse::diagnostic::{Diag, Msg};
use pebblec_parse::lex::buffer::{Span, TokenId, TokenQuery};
use pebblec_parse::lex::kind::TokenKind;
use pebblec_parse::lex::source::SourceMap;
use pebblec_parse::sym::{Ident, Symbol};
use std::borrow::{Borrow, Cow};
use std::collections::HashMap;

#[derive(Debug)]
pub struct Ctx<'a> {
    pub tys: TyStore,
    pub const_map: HashMap<Symbol, Const<'a>>,
    pub source_map: SourceMap,
    pub arena: BlobArena,
    pub funcs: Vec<Func<'a>>,
    pub sigs: IndexMap<Symbol, &'a Sig<'a>>,
    pub impl_sigs: IndexMap<(Ty, Symbol), &'a Sig<'a>>,
}

// TODO: move into deterministic test?
impl PartialEq for Ctx<'_> {
    #[inline]
    fn eq(&self, other: &Ctx) -> bool {
        // omits `arena`
        self.source_map == other.source_map
            && self.tys == other.tys
            && self.const_map == other.const_map
            && self.funcs == other.funcs
            && self.sigs == other.sigs
            && self.impl_sigs == other.impl_sigs
    }
}

impl<'a> Ctx<'a> {
    pub fn new(source_map: SourceMap) -> Self {
        Self {
            tys: TyStore::default(),
            const_map: HashMap::default(),
            source_map,
            arena: BlobArena::default(),
            funcs: Vec::default(),
            sigs: IndexMap::default(),
            impl_sigs: IndexMap::default(),
        }
    }

    pub fn store_funcs(&mut self, funcs: impl IntoIterator<Item = Func<'a>>) {
        self.funcs.extend(funcs.into_iter());
    }

    pub fn build_type_layouts(&mut self) {
        self.tys.build_layouts();
    }

    pub fn store_sigs(&mut self, sigs: impl IntoIterator<Item = Sig<'a>>) -> Result<(), Diag> {
        let mut errors = Vec::new();
        for sig in sigs.into_iter() {
            if let Some(other) = self.sigs.insert(sig.ident, self.intern(sig)) {
                let ident = sig.ident.to_string(self);
                errors.push(
                    self.report_error(sig.span, format!("`{}` is already defined", ident))
                        .msg(Msg::error(
                            &self.source_map,
                            other.span,
                            "previously defined",
                        )),
                );
            }
        }

        if !errors.is_empty() {
            Err(Diag::bundle(errors))
        } else {
            Ok(())
        }
    }

    pub fn store_impl_sigs(&mut self, ty: Ty, sigs: Vec<Sig<'a>>) -> Result<(), Diag> {
        for sig in sigs.into_iter() {
            if let Some(other) = self.impl_sigs.insert((ty, sig.ident), self.intern(sig)) {
                return Err(self
                    .report_error(
                        sig.span,
                        format!("`{}` is already defined", sig.ident.to_string(self)),
                    )
                    .msg(Msg::help(
                        &self.source_map,
                        other.span,
                        "previously defined here",
                    )));
            }
        }

        Ok(())
    }

    pub fn get_sig(&self, ident: Symbol) -> Option<&'a Sig<'a>> {
        self.sigs.get(&ident).copied()
    }

    pub fn get_method_sig(&self, ty: Ty, ident: Symbol) -> Option<&'a Sig<'a>> {
        self.impl_sigs.get(&(ty, ident)).copied()
    }

    pub fn store_const(&mut self, konst: Const<'a>) {
        self.const_map.insert(konst.name.sym, konst);
    }

    pub fn get_const(&self, id: Symbol) -> Option<&Const> {
        self.const_map.get(&id)
    }

    pub fn consts(&self) -> impl Iterator<Item = &Const<'a>> {
        self.const_map.values()
    }

    pub fn token_ident(&self, token: impl Borrow<TokenId>) -> Ident {
        let token = token.borrow();
        Ident {
            sym: Symbol::intern(self.as_str(token)),
            span: self.span(token),
        }
    }
}

#[allow(unused)]
impl<'a> Ctx<'a> {
    #[track_caller]
    pub fn report_error<S: SpannedCtx>(&self, s: S, err: impl Into<Cow<'static, str>>) -> Diag {
        let span = s.ctx_span(self);
        Diag::new(
            Level::Error,
            self.source_map.source(span.source as usize),
            span,
            err.into(),
            Vec::new(),
        )
    }

    #[track_caller]
    pub fn report_warn<S: SpannedCtx>(&self, s: S, err: impl Into<Cow<'static, str>>) -> Diag {
        let span = s.ctx_span(self);
        Diag::new(
            Level::Warning,
            self.source_map.source(span.source as usize),
            span,
            err.into(),
            Vec::new(),
        )
    }

    #[track_caller]
    pub fn report_note<S: SpannedCtx>(&self, s: S, err: impl Into<Cow<'static, str>>) -> Diag {
        let span = s.ctx_span(self);
        Diag::new(
            Level::Note,
            self.source_map.source(span.source as usize),
            span,
            err.into(),
            Vec::new(),
        )
    }

    #[track_caller]
    pub fn report_help<S: SpannedCtx>(&self, s: S, err: impl Into<Cow<'static, str>>) -> Diag {
        let span = s.ctx_span(self);
        Diag::new(
            Level::Help,
            self.source_map.source(span.source as usize),
            span,
            err.into(),
            Vec::new(),
        )
    }

    #[track_caller]
    pub fn mismatch<E: CtxFmt, G: CtxFmt>(&self, span: Span, expected: E, got: G) -> Diag {
        self.report_error(
            span,
            format!(
                "mismatched types: expected `{}`, got `{}`",
                expected.to_string(self),
                got.to_string(self)
            ),
        )
    }

    #[track_caller]
    pub fn undeclared<U: SpannedCtxFmt>(&self, u: U) -> Diag {
        let (span, str) = u.spanned_ctx_fmt(self);
        self.report_error(span, format!("`{}` is not declared", str))
    }
}

impl<'a> Ctx<'a> {
    pub fn intern<T>(&self, item: T) -> &'a T
    where
        T: Copy,
    {
        self.arena.alloc(item)
    }

    pub fn intern_str(&self, str: &str) -> &'a str {
        self.arena.alloc_str(str)
    }

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
}

impl<'a> TokenQuery<'a> for Ctx<'a> {
    fn kind(&self, token: impl Borrow<TokenId>) -> TokenKind {
        let token = *token.borrow();
        let buf = self.source_map.buffer(token.source as usize);
        buf.kind(token)
    }

    fn span(&self, token: impl Borrow<TokenId>) -> Span {
        let token = *token.borrow();
        let buf = self.source_map.buffer(token.source as usize);
        buf.span(token)
    }

    fn as_str(&'a self, token: impl Borrow<TokenId>) -> &'a str {
        let token = *token.borrow();
        let buf = self.source_map.buffer(token.source as usize);
        buf.as_str(token)
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

impl CtxFmt for TyKind {
    fn ctx_fmt<'a>(&'a self, ctx: &'a Ctx<'a>, buf: &mut String) {
        buf.push_str(&self.to_string(ctx));
    }
}

impl CtxFmt for Ty {
    fn ctx_fmt<'a>(&self, ctx: &'a Ctx<'a>, buf: &mut String) {
        self.0.ctx_fmt(ctx, buf);
    }
}

impl CtxFmt for Ident {
    fn ctx_fmt<'a>(&'a self, _: &'a Ctx<'a>, buf: &mut String) {
        buf.push_str(self.as_str());
    }
}

impl CtxFmt for Symbol {
    fn ctx_fmt<'a>(&'a self, _: &'a Ctx<'a>, buf: &mut String) {
        buf.push_str(self.as_str());
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
