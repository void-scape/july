use super::buffer::{TokenBuffer, TokenQuery};
use super::{Lexer, io};
use crate::diagnostic::Diag;
use crate::{Item, ItemKind};
use annotate_snippets::Level;
use std::collections::{HashMap, HashSet};
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use thiserror::Error;

static SOURCE_ID: AtomicUsize = AtomicUsize::new(0);

pub fn core_parent_path() -> PathBuf {
    let mut path = std::env::current_exe().unwrap();
    while !path.ends_with("pebble") {
        path.pop();
    }
    path
}

#[derive(Error, Debug)]
pub enum SourceError {
    #[error("expected `.peb` file, got `{0}`")]
    InvalidExtension(String),
    #[error("could not open `{file}`: {io}")]
    Io { file: String, io: std::io::Error },
    #[error("encountered unparsable symbol in `{0}`")]
    Lex(String),
    #[error("failed to parse `{0}`")]
    Parse(String),
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct SourceMap {
    tokens: HashMap<usize, TokenBuffer>,
}

impl SourceMap {
    #[track_caller]
    pub fn origin(&self) -> &OsStr {
        &self.tokens.values().next().unwrap().source_ref().origin
    }

    /// Create a `SourceMap` from a file path.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, SourceError> {
        if !path.as_ref().to_string_lossy().ends_with(".peb") {
            return Err(SourceError::InvalidExtension(
                path.as_ref().to_string_lossy().to_string(),
            ));
        }

        let mut tokens = HashMap::new();
        let src = Source::new(path.as_ref()).map_err(|io| SourceError::Io {
            file: path.as_ref().to_string_lossy().to_string(),
            io,
        })?;
        let origin = src.origin.to_string_lossy().to_string();
        match Lexer::new(src).lex() {
            Ok(buf) => {
                tokens.insert(buf.source_id(), buf);
            }
            Err(_) => {
                return Err(SourceError::Lex(origin));
            }
        }

        Ok(Self { tokens })
    }

    /// Create a `SourceMap` from a source string.
    ///
    /// `Origin` is the file location that diagnostics will refer to.
    pub fn from_string<Origin: AsRef<OsStr>>(
        origin: Origin,
        src: String,
    ) -> Result<Self, SourceError> {
        if !origin.as_ref().to_string_lossy().ends_with(".peb") {
            return Err(SourceError::InvalidExtension(
                origin.as_ref().to_string_lossy().to_string(),
            ));
        }

        let mut tokens = HashMap::new();
        let src = Source::from_string(origin, src);
        let origin = src.origin.to_string_lossy().to_string();
        match Lexer::new(src).lex() {
            Ok(buf) => {
                tokens.insert(buf.source_id(), buf);
            }
            Err(_) => {
                return Err(SourceError::Lex(origin));
            }
        }

        Ok(Self { tokens })
    }

    pub fn insert(&mut self, buffer: TokenBuffer) {
        self.tokens.insert(buffer.source_id(), buffer);
    }

    pub fn buffers(&self) -> impl Iterator<Item = &TokenBuffer> {
        self.tokens.values()
    }

    #[track_caller]
    pub fn buffer(&self, id: usize) -> &TokenBuffer {
        self.tokens.get(&id).unwrap()
    }

    #[track_caller]
    pub fn source(&self, id: usize) -> Arc<Source> {
        self.buffer(id).source()
    }

    // TODO: this should be simpler
    pub fn parse(&mut self) -> Result<Vec<Item>, SourceError> {
        let origin = self.origin().to_owned();
        let mut visited = HashSet::new();

        let path = PathBuf::from(origin.clone());
        // HACK: need some proper way to define what the root of a project is.
        //
        // If you try to compile a core file, and it recursively imports the origin core file, then
        // it won't skip it because the origin core won't be properly defined in the visited set.
        if path
            .components()
            .any(|c| c.as_os_str().to_str().is_some_and(|str| str == "core"))
        {
            visited.insert(vec![
                String::from("core"),
                path.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .strip_suffix(".peb")
                    .unwrap()
                    .to_string(),
            ]);
        } else {
            visited.insert(vec![
                path.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .strip_suffix(".peb")
                    .unwrap()
                    .to_string(),
            ]);
        }

        let mut err = false;
        let mut items = self
            .buffers()
            .filter_map(|buf| match crate::parse(buf) {
                Ok(items) => Some(items),
                Err(diag) => {
                    err = true;
                    diag.report();
                    None
                }
            })
            .flatten()
            .collect::<Vec<_>>();
        self.parse_uses(&origin, &mut visited, &mut items)?;

        if err {
            Err(SourceError::Parse(origin.to_string_lossy().to_string()))
        } else {
            Ok(items)
        }
    }

    fn parse_uses(
        &mut self,
        origin: &OsStr,
        visited: &mut HashSet<Vec<String>>,
        items: &mut Vec<Item>,
    ) -> Result<(), SourceError> {
        let uses = items
            .iter()
            .filter_map(|item| match &item.kind {
                ItemKind::Use(uze) => Some((
                    uze.span,
                    uze.path
                        .iter()
                        .map(|step| self.buffer(item.source).as_str(step).to_string())
                        .collect::<Vec<_>>(),
                )),
                _ => None,
            })
            .collect::<Vec<_>>();

        let new_sources = uses
            .iter()
            .filter(|(_, uze)| visited.insert(uze.to_vec()))
            .collect::<Vec<_>>();

        if new_sources.is_empty() {
            return Ok(());
        }

        let mut new_items = Vec::new();
        for (span, source) in new_sources {
            assert!(!source.is_empty());
            let mut path = if source.iter().next().is_some_and(|p| p == "core") {
                core_parent_path()
            } else {
                let mut path = PathBuf::from(origin);
                path.pop();
                path
            };
            for (i, step) in source.iter().enumerate() {
                if i == source.len() - 1 {
                    path.push(format!("{}.peb", step));
                } else {
                    path.push(step);
                }
            }

            match Source::new(&path) {
                Err(io) => {
                    Diag::new(
                        Level::Error,
                        self.source(span.source as usize),
                        *span,
                        "could not resolve path",
                        Vec::new(),
                    )
                    .report();

                    return Err(SourceError::Io {
                        file: path.to_string_lossy().to_string(),
                        io,
                    });
                }
                Ok(source) => {
                    let src = source.origin.to_string_lossy().to_string();
                    let buffer = Lexer::new(source).lex().unwrap();
                    let mut err = false;
                    match crate::parse(&buffer) {
                        Ok(items) => new_items.extend(items),
                        Err(diag) => {
                            err = true;
                            diag.report()
                        }
                    }
                    if err {
                        return Err(SourceError::Parse(src));
                    }
                    self.insert(buffer);
                }
            }
        }

        self.parse_uses(origin, visited, &mut new_items)?;
        items.extend(new_items);
        Ok(())
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
