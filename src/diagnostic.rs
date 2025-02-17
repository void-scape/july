use crate::lex::buffer::Span;
use crate::unit::source::Source;
use annotate_snippets::{Level, Renderer, Snippet};
use std::borrow::Cow;

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
    pub fn new(title: &'static str, diag: impl Diagnostic<'a>) -> Self {
        Self {
            inner: DiagInnerPtr::One(Box::new(diag.into_diagnostic(title))),
        }
    }

    pub fn sourced(title: &'static str, source: &'a Source, msg: Msg) -> Self {
        Self {
            inner: DiagInnerPtr::One(Box::new(Sourced::new(source, msg).into_diagnostic(title))),
        }
    }

    pub fn wrap(self, other: Self) -> Self {
        let mut new = other.into_inner();
        new.extend(self.into_inner());
        Self {
            inner: DiagInnerPtr::Many(new),
        }
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

    fn into_inner(self) -> Vec<DiagInner<'a>> {
        match self.inner {
            DiagInnerPtr::One(diag) => vec![*diag],
            DiagInnerPtr::Many(diags) => diags,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DiagInner<'a> {
    title: &'static str,
    source: &'a str,
    origin: Cow<'a, str>,
    level: Level,
    msgs: Vec<Msg>,
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
            source: self.source.raw(),
            origin: self.source.origin().to_string_lossy(),
            level: Level::Error,
            reported: false,

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
            source: self.source.raw(),
            origin: self.source.origin().to_string_lossy(),
            level: Level::Error,
            msgs: vec![self.inner],
            reported: false,
        }
    }
}

impl<'a> Diagnostic<'a> for Sourced<'a, Level> {
    fn into_diagnostic(self, title: &'static str) -> DiagInner<'a> {
        DiagInner {
            title,
            source: self.source.raw(),
            origin: self.source.origin().to_string_lossy(),
            level: self.inner,
            msgs: Vec::new(),
            reported: false,
        }
    }
}

pub fn report(diag: Diag) {
    for mut diag in diag.into_inner() {
        let message = diag.level.title(diag.title).snippet(
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
        anstream::println!("{}", renderer.render(message));
        diag.reported = true;
    }
}
