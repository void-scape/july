use crate::air::ByteCode;
use crate::{air, ice, ir};
use pebblec_parse::lex::source::{SourceError, SourceMap};
use std::ffi::OsStr;
use std::panic::{AssertUnwindSafe, UnwindSafe};
use std::path::Path;

pub const COMP_PANIC_NOTE: &str = "compiler encountered unexpected panic, this is a bug";

#[derive(Debug, Clone)]
pub struct Config {
    pub log: bool,
    pub no_capture: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            log: false,
            no_capture: false,
        }
    }
}

impl Config {
    pub fn log(mut self, log: bool) -> Self {
        self.log = log;
        self
    }

    pub fn no_capture(mut self, no_capture: bool) -> Self {
        self.no_capture = no_capture;
        self
    }
}

#[derive(Debug, Default, Clone)]
pub struct CompUnit {
    config: Config,
}

impl CompUnit {
    pub fn new(config: impl Into<Config>) -> Self {
        Self {
            config: config.into(),
        }
    }

    pub fn compile<'a, P: AsRef<Path> + UnwindSafe>(
        &mut self,
        path: P,
    ) -> Result<ByteCode<'a>, CompErr> {
        let capture = !self.config.no_capture;
        let mut unit = AssertUnwindSafe(self);
        ice::reported_panic(capture, COMP_PANIC_NOTE, move || {
            unit.panicking_compile(path)
        })
        .ok_or(CompErr::Panic)?
    }

    pub fn compile_string<'a, Origin: AsRef<OsStr> + UnwindSafe>(
        &mut self,
        origin: Origin,
        src: String,
    ) -> Result<ByteCode<'a>, CompErr> {
        let capture = !self.config.no_capture;
        let mut unit = AssertUnwindSafe(self);
        ice::reported_panic(capture, COMP_PANIC_NOTE, move || {
            unit.panicking_compile_string(origin, src)
        })
        .ok_or(CompErr::Panic)?
    }

    pub fn panicking_compile_string<'a, Origin: AsRef<OsStr>>(
        &mut self,
        origin: Origin,
        string: String,
    ) -> Result<ByteCode<'a>, CompErr> {
        let (source_init, source_map) =
            Self::record_time_result(|| SourceMap::from_string(origin, string))?;
        let lines = source_map
            .buffers()
            .map(|b| b.source().source.lines().count())
            .sum::<usize>();

        let (parse_dur, ir) = Self::record_time_result(|| ir::lower(source_map))?;
        let (bytecode_dur, bytecode) = Self::record_time(|| air::lower(ir));

        self.log_report(Report {
            lines,
            timing: CompTiming {
                parse: source_init + parse_dur,
                bytecode: bytecode_dur,
            },
        });

        Ok(bytecode)
    }

    pub fn panicking_compile<'a, P: AsRef<Path>>(
        &mut self,
        path: P,
    ) -> Result<ByteCode<'a>, CompErr> {
        let (source_init, source_map) = Self::record_time_result(|| SourceMap::from_path(path))?;
        let lines = source_map
            .buffers()
            .map(|b| b.source().source.lines().count())
            .sum::<usize>();

        let (parse_dur, ir) = Self::record_time_result(|| ir::lower(source_map))?;
        let (bytecode_dur, bytecode) = Self::record_time(|| air::lower(ir));

        self.log_report(Report {
            lines,
            timing: CompTiming {
                parse: source_init + parse_dur,
                bytecode: bytecode_dur,
            },
        });

        Ok(bytecode)
    }

    fn record_time_result<R, E>(f: impl FnOnce() -> Result<R, E>) -> Result<(f32, R), E> {
        let start = std::time::Instant::now();
        let result = f();
        let end = std::time::Instant::now()
            .duration_since(start)
            .as_secs_f32();
        result.map(|r| (end, r))
    }

    fn record_time<R>(f: impl FnOnce() -> R) -> (f32, R) {
        let start = std::time::Instant::now();
        let result = f();
        let end = std::time::Instant::now()
            .duration_since(start)
            .as_secs_f32();
        (end, result)
    }

    fn log_report(&self, report: Report) {
        println!("{} lines", report.lines);
        self.report_time("parse", report.timing.parse);
        self.report_time("bytecode", report.timing.bytecode);
        self.report_time("total", report.timing.parse + report.timing.bytecode);
        println!();
    }

    fn report_time(&self, title: &'static str, time: f32) {
        println!("  {:.4}s ... {}", time, title);
    }
}

#[derive(Debug)]
pub enum CompErr {
    Source(SourceError),
    Panic,
    Ir,
}

impl From<SourceError> for CompErr {
    fn from(value: SourceError) -> Self {
        CompErr::Source(value)
    }
}

pub struct Report {
    pub lines: usize,
    pub timing: CompTiming,
}

pub struct CompTiming {
    pub parse: f32,
    pub bytecode: f32,
}
