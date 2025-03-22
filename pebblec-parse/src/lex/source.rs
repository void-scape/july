use super::io;
use std::ffi::{OsStr, OsString};
use std::path::Path;

#[derive(Debug, PartialEq, Eq)]
pub struct Source {
    source: String,
    origin: OsString,
}

impl Source {
    pub fn new<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let origin = path.as_ref().as_os_str().to_owned();
        let source = io::read_string(path)?;
        Ok(Self::from_string(origin, source))
    }

    pub fn from_string<O: AsRef<OsStr>>(origin: O, str: String) -> Self {
        Self {
            source: str,
            origin: origin.as_ref().to_owned(),
        }
    }

    pub fn raw(&self) -> &str {
        &self.source
    }

    pub fn origin(&self) -> &OsStr {
        &self.origin
    }
}
