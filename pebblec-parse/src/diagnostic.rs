#![allow(unused)]
use crate::lex::buffer::Span;
use crate::lex::source::{Source, SourceMap};
use annotate_snippets::{Level, Renderer, Snippet};
use indexmap::IndexMap;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
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
    source: Arc<Source>,
    span: Span,
    msg: Cow<'static, str>,
}

#[allow(unused)]
impl Msg {
    pub fn new(
        level: Level,
        source: Arc<Source>,
        span: Span,
        msg: impl Into<Cow<'static, str>>,
    ) -> Self {
        Self {
            level,
            source,
            span,
            msg: msg.into(),
        }
    }

    pub fn spanned(level: Level, source_map: &SourceMap, span: Span) -> Self {
        Self::new(level, source_map.source(span.source as usize), span, "")
    }

    pub fn error(source_map: &SourceMap, span: Span, msg: impl Into<Cow<'static, str>>) -> Self {
        Self::new(
            Level::Error,
            source_map.source(span.source as usize),
            span,
            msg,
        )
    }

    pub fn warning(source_map: &SourceMap, span: Span, msg: impl Into<Cow<'static, str>>) -> Self {
        Self::new(
            Level::Warning,
            source_map.source(span.source as usize),
            span,
            msg,
        )
    }

    pub fn info(source_map: &SourceMap, span: Span, msg: impl Into<Cow<'static, str>>) -> Self {
        Self::new(
            Level::Info,
            source_map.source(span.source as usize),
            span,
            msg,
        )
    }

    pub fn note(source_map: &SourceMap, span: Span, msg: impl Into<Cow<'static, str>>) -> Self {
        Self::new(
            Level::Note,
            source_map.source(span.source as usize),
            span,
            msg,
        )
    }

    pub fn help(source_map: &SourceMap, span: Span, msg: impl Into<Cow<'static, str>>) -> Self {
        Self::new(
            Level::Help,
            source_map.source(span.source as usize),
            span,
            msg,
        )
    }

    pub fn error_span(source_map: &SourceMap, span: Span) -> Self {
        Self::spanned(Level::Error, source_map, span)
    }

    pub fn warning_span(source_map: &SourceMap, span: Span) -> Self {
        Self::spanned(Level::Warning, source_map, span)
    }

    pub fn info_span(source_map: &SourceMap, span: Span) -> Self {
        Self::spanned(Level::Info, source_map, span)
    }

    pub fn note_span(source_map: &SourceMap, span: Span) -> Self {
        Self::spanned(Level::Note, source_map, span)
    }

    pub fn help_span(source_map: &SourceMap, span: Span) -> Self {
        Self::spanned(Level::Help, source_map, span)
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
        let origin = self.source.origin.to_string_lossy();

        let mut common_msgs: IndexMap<u32, Vec<Msg>> = IndexMap::new();
        common_msgs
            .entry(self.span.source)
            .or_default()
            .push(Msg::new(
                Level::Error,
                self.source.clone(),
                self.span,
                self.title,
            ));
        for msg in self.msgs.drain(..) {
            common_msgs.entry(msg.span.source).or_default().push(msg);
        }

        for (_, msgs) in common_msgs.into_iter() {
            let first = msgs.first().unwrap();

            let message = first.level.title(&first.msg).snippet(
                Snippet::source(&first.source.source)
                    .origin(&origin)
                    .fold(true)
                    .annotations(msgs.iter().enumerate().map(|(i, msg)| {
                        if i == 0 {
                            msg.level.span(msg.span.range())
                        } else {
                            msg.level.span(msg.span.range()).label(&msg.msg)
                        }
                    })),
            );

            let renderer = Renderer::styled();
            println!("{}", renderer.render(message));
            println!("generated: {}", self.loc);
        }
    }
}
