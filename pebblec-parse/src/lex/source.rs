use super::buffer::TokenBuffer;
use super::{Lexer, io};
use std::borrow::Borrow;
use std::collections::HashMap;
use std::ffi::{OsStr, OsString};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use thiserror::Error;

static SOURCE_ID: AtomicUsize = AtomicUsize::new(0);

#[derive(Error, Debug)]
pub enum SourceError {
    #[error("could not open `{file}`: {io}")]
    Io { file: String, io: std::io::Error },
    #[error("encountered unparsable symbol in `{0}`")]
    Lex(String),
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct SourceMap<'a> {
    tokens: HashMap<usize, TokenBuffer<'a>>,
}

impl<'a> SourceMap<'a> {
    /// Create a `SourceMap` from a collection of file paths.
    pub fn from_paths<'p, P: AsRef<Path>>(
        sources: impl IntoIterator<Item = P>,
    ) -> Result<Self, SourceError> {
        let mut srcs = Vec::new();
        for source in sources.into_iter() {
            srcs.push(Source::new(&source).map_err(|io| SourceError::Io {
                file: source.as_ref().to_string_lossy().to_string(),
                io,
            })?);
        }

        let mut tokens = HashMap::with_capacity(srcs.len());
        for src in srcs.into_iter() {
            let origin = src.origin.to_string_lossy().to_string();
            match Lexer::new(src).lex() {
                Ok(buf) => {
                    tokens.insert(buf.source_id(), buf);
                }
                Err(err) => {
                    return Err(SourceError::Lex(origin));
                }
            }
        }

        Ok(Self { tokens })
    }

    /// Create a `SourceMap` from a collection of strings.
    ///
    /// `Origin` is the file location that diagnostics will refer to.
    pub fn from_strings<'s, Origin: Borrow<&'s str>, Src: Into<String>>(
        sources: impl IntoIterator<Item = (Origin, Src)>,
    ) -> Option<Self> {
        let mut srcs = Vec::new();
        for (origin, source) in sources.into_iter() {
            srcs.push(Source::from_string(origin.borrow(), source.into()));
        }

        let mut tokens = HashMap::with_capacity(srcs.len());
        for src in srcs.into_iter() {
            match Lexer::new(src).lex() {
                Ok(buf) => {
                    tokens.insert(buf.source_id(), buf);
                }
                // TODO: report error and return `None`
                Err(err) => panic!("encountered unrecognized symbol"),
            }
        }

        Some(Self { tokens })
    }

    pub fn insert(&mut self, buffer: TokenBuffer<'a>) {
        self.tokens.insert(buffer.source_id(), buffer);
    }

    pub fn buffers(&self) -> impl Iterator<Item = &TokenBuffer<'a>> {
        self.tokens.values()
    }

    #[track_caller]
    pub fn buffer(&self, id: usize) -> &TokenBuffer<'a> {
        self.tokens.get(&id).unwrap()
    }

    #[track_caller]
    pub fn source(&self, id: usize) -> &Source {
        self.buffer(id).source()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Source {
    //pub id: usize,
    pub source: String,
    pub origin: OsString,
    pub id: usize,
}

impl Source {
    /// Panics if any [`SourceData`] are alive.
    #[track_caller]
    pub fn new<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let origin = path.as_ref().as_os_str().to_owned();
        let source = io::read_string(path)?;
        Ok(Self::from_string(origin, source))
    }

    /// Panics if any [`SourceData`] are alive.
    #[track_caller]
    pub fn from_string<O: AsRef<OsStr>>(origin: O, str: String) -> Self {
        Self {
            source: str,
            origin: origin.as_ref().to_os_string(),
            id: SOURCE_ID.fetch_add(1, Ordering::SeqCst),
        }
    }
}
