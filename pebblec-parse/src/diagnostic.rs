#![allow(unused)]
use crate::lex::buffer::Span;
use crate::lex::source::Source;
use annotate_snippets::{Level, Renderer, Snippet};
use std::borrow::Cow;
use std::panic::Location;

#[derive(Debug)]
pub struct Diag<'a> {
    inner: DiagInnerPtr<'a>,
}

#[derive(Debug)]
pub enum DiagInnerPtr<'a> {
    One(Box<DiagInner<'a>>),
    Many(Vec<DiagInner<'a>>),
}

impl<'a> Diag<'a> {
    pub fn new(title: impl Into<String>, diag: impl Diagnostic<'a>) -> Self {
        Self {
            inner: DiagInnerPtr::One(Box::new(diag.into_diagnostic(title.into()))),
        }
    }

    pub fn sourced(title: impl Into<String>, source: &'a Source, msg: Msg) -> Self {
        Self {
            inner: DiagInnerPtr::One(Box::new(
                Sourced::new(source, msg).into_diagnostic(title.into()),
            )),
        }
    }

    pub fn wrap(self, other: Self) -> Self {
        let mut new = self.into_inner();
        new.extend(other.into_inner());
        Self {
            inner: DiagInnerPtr::Many(new),
        }
    }

    /// Panics if bundle is empty
    #[track_caller]
    pub fn bundle(bundle: impl IntoIterator<Item = Self>) -> Self {
        let mut iter = bundle.into_iter();
        let Some(mut first) = iter.next() else {
            panic!("bundle cannot be empty");
        };

        for diag in iter {
            first = first.wrap(diag);
        }

        first
    }

    pub fn level(mut self, level: Level) -> Self {
        match &mut self.inner {
            DiagInnerPtr::One(diag) => {
                diag.level = level;
            }
            DiagInnerPtr::Many(diags) => {
                diags.last_mut().map(|diag| diag.level = level);
            }
        }
        self
    }

    pub fn msg(mut self, msg: Msg) -> Self {
        match &mut self.inner {
            DiagInnerPtr::One(diag) => {
                diag.msgs.push(msg);
            }
            DiagInnerPtr::Many(diags) => {
                diags.last_mut().map(|diag| diag.msgs.push(msg));
            }
        }
        self
    }

    pub fn loc(mut self, loc: &Location) -> Self {
        match &mut self.inner {
            DiagInnerPtr::One(diag) => {
                diag.compiler_loc = loc.to_string();
            }
            DiagInnerPtr::Many(diags) => {
                diags
                    .last_mut()
                    .map(|diag| diag.compiler_loc = loc.to_string());
            }
        }
        self
    }

    fn into_inner(self) -> Vec<DiagInner<'a>> {
        match self.inner {
            DiagInnerPtr::One(diag) => vec![*diag],
            DiagInnerPtr::Many(diags) => diags,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DiagInner<'a> {
    title: String,
    source: &'a str,
    origin: Cow<'a, str>,
    level: Level,
    msgs: Vec<Msg>,
    compiler_loc: String,
    reported: bool,
}

impl Drop for DiagInner<'_> {
    fn drop(&mut self) {
        if !self.reported {
            //panic!("unreported error: {:#?}", self);
        }

        let _ = std::mem::take(&mut self.msgs);
        let _ = std::mem::replace(&mut self.origin, Cow::Borrowed(""));
    }
}

#[derive(Debug, Clone)]
pub struct Msg {
    pub level: Level,
    pub span: Span,
    pub label: String,
}

impl Msg {
    pub fn new(level: Level, span: Span, label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            level,
            span,
        }
    }

    pub fn empty() -> Self {
        Self::new(Level::Error, Span::empty(), "")
    }

    pub fn error(span: Span, label: impl Into<String>) -> Self {
        Self::new(Level::Error, span, label)
    }

    pub fn warn(span: Span, label: impl Into<String>) -> Self {
        Self::new(Level::Warning, span, label)
    }

    pub fn warn_span(span: Span) -> Self {
        Self::new(Level::Warning, span, "")
    }

    pub fn error_span(span: Span) -> Self {
        Self::new(Level::Error, span, "")
    }

    pub fn note(span: Span, label: impl Into<String>) -> Self {
        Self::new(Level::Note, span, label)
    }

    pub fn note_span(span: Span) -> Self {
        Self::new(Level::Note, span, "")
    }

    pub fn info(span: Span, label: impl Into<String>) -> Self {
        Self::new(Level::Info, span, label)
    }

    pub fn info_span(span: Span) -> Self {
        Self::new(Level::Info, span, "")
    }

    pub fn help(span: Span, label: impl Into<String>) -> Self {
        Self::new(Level::Help, span, label)
    }

    pub fn help_span(span: Span) -> Self {
        Self::new(Level::Help, span, "")
    }
}

pub trait Diagnostic<'a> {
    fn into_diagnostic(self, title: String) -> DiagInner<'a>;
}

pub struct Label<'a> {
    label: Cow<'a, str>,
    level: Level,
}

impl<'a> Label<'a> {
    pub fn new(label: &'a str, level: Level) -> Self {
        Self {
            label: Cow::Borrowed(label),
            level,
        }
    }

    pub fn new_owned(label: String, level: Level) -> Self {
        Self {
            label: Cow::Owned(label),
            level,
        }
    }

    pub fn error(label: impl Into<Cow<'a, str>>) -> Self {
        Self {
            label: label.into(),
            level: Level::Error,
        }
    }

    pub fn note(label: impl Into<Cow<'a, str>>) -> Self {
        Self {
            label: label.into(),
            level: Level::Note,
        }
    }
}

pub struct Spanned<T> {
    span: Span,
    inner: T,
}

impl<T> Spanned<T> {
    pub fn new(span: Span, inner: T) -> Self {
        Self { span, inner }
    }
}

pub struct Sourced<'a, T> {
    source: &'a Source,
    inner: T,
}

impl<'a, T> Sourced<'a, T> {
    pub fn new(source: &'a Source, inner: T) -> Self {
        Self { source, inner }
    }
}

impl<'a> Diagnostic<'a> for Sourced<'a, Spanned<Label<'a>>> {
    fn into_diagnostic(self, title: String) -> DiagInner<'a> {
        DiagInner {
            title,
            source: self.source.raw(),
            origin: self.source.origin().to_string_lossy(),
            level: Level::Error,
            reported: false,
            compiler_loc: String::new(),

            msgs: vec![Msg::new(
                self.inner.inner.level,
                self.inner.span,
                self.inner.inner.label.to_string(),
            )],
        }
    }
}

impl<'a> Diagnostic<'a> for Sourced<'a, Msg> {
    fn into_diagnostic(self, title: String) -> DiagInner<'a> {
        DiagInner {
            title,
            source: self.source.raw(),
            origin: self.source.origin().to_string_lossy(),
            level: Level::Error,
            msgs: vec![self.inner],
            compiler_loc: String::new(),
            reported: false,
        }
    }
}

impl<'a> Diagnostic<'a> for Sourced<'a, Level> {
    fn into_diagnostic(self, title: String) -> DiagInner<'a> {
        DiagInner {
            title,
            source: self.source.raw(),
            origin: self.source.origin().to_string_lossy(),
            level: self.inner,
            msgs: Vec::new(),
            compiler_loc: String::new(),
            reported: false,
        }
    }
}

pub fn report(diag: Diag) {
    for mut diag in diag.into_inner() {
        let message = diag.level.title(&diag.title).snippet(
            Snippet::source(&diag.source)
                .origin(&diag.origin)
                .fold(true)
                .annotations(
                    diag.msgs
                        .iter()
                        .map(|msg| msg.level.span(msg.span.range()).label(&msg.label)),
                ),
        );

        let renderer = Renderer::styled();
        println!("{}", renderer.render(message));
        println!("generated: {}", diag.compiler_loc);
        diag.reported = true;
    }
}

pub fn report_set<'a>(set: impl IntoIterator<Item = Diag<'a>>) {
    for diag in set.into_iter() {
        report(diag);
    }
}
