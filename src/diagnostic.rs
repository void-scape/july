use crate::lex::buffer::Span;
use crate::source::Source;
use annotate_snippets::{Level, Renderer, Snippet};
use std::borrow::Cow;

#[derive(Debug)]
pub struct Diag<'a> {
    inner: Box<DiagInner<'a>>,
}

impl<'a> Diag<'a> {
    pub fn new(title: &'static str, diag: impl Diagnostic<'a>) -> Self {
        Self {
            inner: Box::new(diag.into_diagnostic(title)),
        }
    }

    pub fn sourced(title: &'static str, source: &'a Source, msg: Msg) -> Self {
        Self {
            inner: Box::new(Sourced::new(source, msg).into_diagnostic(title)),
        }
    }

    pub fn level(mut self, level: Level) -> Self {
        self.inner.level = level;
        self
    }

    pub fn msg(&mut self, msg: Msg) {
        self.inner.msgs.push(msg);
    }
}

#[derive(Debug, Clone)]
pub struct DiagInner<'a> {
    title: &'static str,
    source: &'a str,
    origin: &'a str,
    level: Level,
    msgs: Vec<Msg>,
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

    pub fn error(span: Span, label: impl Into<String>) -> Self {
        Self::new(Level::Error, span, label)
    }

    pub fn note(span: Span, label: impl Into<String>) -> Self {
        Self::new(Level::Note, span, label)
    }

    pub fn help(span: Span, label: impl Into<String>) -> Self {
        Self::new(Level::Help, span, label)
    }
}

pub trait Diagnostic<'a> {
    fn into_diagnostic(self, title: &'static str) -> DiagInner<'a>;
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
    fn into_diagnostic(self, title: &'static str) -> DiagInner<'a> {
        DiagInner {
            title,
            source: self.source.as_str(),
            origin: self.source.origin(),
            level: Level::Error,

            msgs: vec![Msg::new(
                self.inner.inner.level,
                self.inner.span,
                self.inner.inner.label.to_string(),
            )],
        }
    }
}

impl<'a> Diagnostic<'a> for Sourced<'a, Msg> {
    fn into_diagnostic(self, title: &'static str) -> DiagInner<'a> {
        DiagInner {
            title,
            source: self.source.as_str(),
            origin: self.source.origin(),
            level: Level::Error,
            msgs: vec![self.inner],
        }
    }
}

pub fn report(diag: Diag) {
    let message = diag.inner.level.title(diag.inner.title).snippet(
        Snippet::source(&diag.inner.source)
            .origin(diag.inner.origin)
            .fold(true)
            .annotations(
                diag.inner
                    .msgs
                    .iter()
                    .map(|msg| msg.level.span(msg.span.range()).label(&msg.label)),
            ),
    );

    let renderer = Renderer::styled();
    anstream::println!("{}", renderer.render(message))
}
