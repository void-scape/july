#![allow(unused)]
use crate::lex::buffer::Span;
use crate::lex::source::{Source, SourceMap};
use annotate_snippets::{Level, Renderer, Snippet};
use std::borrow::Cow;
use std::collections::HashSet;
use std::ffi::OsStr;
use std::marker::PhantomData;
use std::panic::Location;
use std::sync::Arc;

/// Compiler diagnostic.
///
/// `Diag` can be one or many [`RawDiag`]s. Each [`RawDiag`] represents a single diagnostic
/// message with a title, origin, and span. [`RawDiag`]s can hold additional [`Msg`]s that
/// will appear within the same report.
///
/// `Diag`s are rendered with [`annotate_snippets`] in [`Diag::report`].
#[derive(Debug)]
pub enum Diag {
    Single(RawDiag),
    Bundle(Vec<RawDiag>),
}

impl Diag {
    #[track_caller]
    pub fn new(
        level: Level,
        source: Arc<Source>,
        span: Span,
        title: impl Into<Cow<'static, str>>,
        msgs: Vec<Msg>,
    ) -> Self {
        Self::Single(RawDiag {
            level,
            source,
            span,
            title: title.into(),
            msgs,
            loc: Location::caller(),
        })
    }

    /// Panics when `bundle` is empty
    #[track_caller]
    pub fn bundle(bundle: impl IntoIterator<Item = Diag>) -> Self {
        let mut buf = Vec::new();
        for diag in bundle.into_iter() {
            match diag {
                Diag::Single(raw) => buf.push(raw),
                Diag::Bundle(bundle) => buf.extend(bundle),
            }
        }
        assert!(!buf.is_empty());
        Self::Bundle(buf)
    }

    /// Append `msg` to the most recent `RawDiag`.
    pub fn msg(mut self, msg: Msg) -> Self {
        match &mut self {
            Self::Single(raw) => raw.msgs.push(msg),
            // cannot have an empty bundle
            Self::Bundle(bundle) => bundle.last_mut().unwrap().msgs.push(msg),
        }

        self
    }

    /// Append `msgs` to the most recent `RawDiag`.
    pub fn msgs(mut self, msgs: impl IntoIterator<Item = Msg>) -> Self {
        match &mut self {
            Self::Single(raw) => {
                raw.msgs.extend(msgs);
            }
            // cannot have an empty bundle
            Self::Bundle(bundle) => bundle.last_mut().unwrap().msgs.extend(msgs),
        }

        self
    }

    /// Concat two sets of `RawDiag` together.
    ///
    /// Useful to `bundle` diagnostics together for simple `Result<T, Diag>` types.
    pub fn join(self, other: Self) -> Self {
        match self {
            Self::Single(raw) => match other {
                Self::Single(other_raw) => Self::Bundle(vec![raw, other_raw]),
                Self::Bundle(mut bundle) => {
                    bundle.push(raw);
                    Self::Bundle(bundle)
                }
            },
            Self::Bundle(mut bundle) => match other {
                Self::Single(raw) => {
                    bundle.push(raw);
                    Self::Bundle(bundle)
                }
                Self::Bundle(other_bundle) => {
                    bundle.extend(other_bundle);
                    Self::Bundle(bundle)
                }
            },
        }
    }

    /// Write diagnostics in `self` to stdout.
    #[track_caller]
    pub fn report(self) {
        match self {
            Self::Single(raw) => raw.report(),
            Self::Bundle(bundle) => bundle.into_iter().for_each(RawDiag::report),
        }
    }
}

/// Generic message associated with a [`RawDiag`].
///
/// Use [`Diag::msg`] or [`Diag::msgs`] to associate a message or collection of messages with a [`Diag`].
#[derive(Debug)]
pub struct Msg {
    level: Level,
    // TODO: this is dangerous, it may have a different source than the diag
    span: Span,
    msg: Cow<'static, str>,
}

#[allow(unused)]
impl Msg {
    pub fn new(level: Level, span: Span, msg: impl Into<Cow<'static, str>>) -> Self {
        Self {
            level,
            span,
            msg: msg.into(),
        }
    }

    pub fn spanned(level: Level, span: Span) -> Self {
        Self::new(level, span, "")
    }

    pub fn error(span: Span, msg: impl Into<Cow<'static, str>>) -> Self {
        Self::new(Level::Error, span, msg)
    }

    pub fn warning(span: Span, msg: impl Into<Cow<'static, str>>) -> Self {
        Self::new(Level::Warning, span, msg)
    }

    pub fn info(span: Span, msg: impl Into<Cow<'static, str>>) -> Self {
        Self::new(Level::Info, span, msg)
    }

    pub fn note(span: Span, msg: impl Into<Cow<'static, str>>) -> Self {
        Self::new(Level::Note, span, msg)
    }

    pub fn help(span: Span, msg: impl Into<Cow<'static, str>>) -> Self {
        Self::new(Level::Help, span, msg)
    }

    pub fn error_span(span: Span) -> Self {
        Self::spanned(Level::Error, span)
    }

    pub fn warning_span(span: Span) -> Self {
        Self::spanned(Level::Warning, span)
    }

    pub fn info_span(span: Span) -> Self {
        Self::spanned(Level::Info, span)
    }

    pub fn note_span(span: Span) -> Self {
        Self::spanned(Level::Note, span)
    }

    pub fn help_span(span: Span) -> Self {
        Self::spanned(Level::Help, span)
    }
}

#[derive(Debug)]
pub struct RawDiag {
    level: Level,
    source: Arc<Source>,
    span: Span,
    title: Cow<'static, str>,
    msgs: Vec<Msg>,
    loc: &'static Location<'static>,
}

impl RawDiag {
    #[track_caller]
    pub fn report(mut self) {
        assert!(
            self.msgs
                .iter()
                .all(|msg| msg.span.source == self.source.id as u32),
            "source: {}\n{:#?}",
            self.source.id,
            self.msgs
        );

        let origin = self.source.origin.to_string_lossy();
        self.msgs.push(Msg::error_span(self.span));
        let message = self.level.title(&self.title).snippet(
            Snippet::source(&self.source.source)
                .origin(&origin)
                .fold(true)
                .annotations(
                    self.msgs
                        .iter()
                        .map(|msg| msg.level.span(msg.span.range()).label(&msg.msg)),
                ),
        );

        let renderer = Renderer::styled();
        println!("{}", renderer.render(message));
        println!("generated: {}", self.loc);
    }
}
