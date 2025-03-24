use crate::air::ctx::AirCtx;
use crate::air::{Air, AirFunc};
use crate::ir::ctx::Ctx;
use crate::{air, interp, ir};
use pebblec_parse::diagnostic::{Diag, Msg};
use pebblec_parse::lex::buffer::{TokenBuffer, TokenQuery};
use pebblec_parse::lex::source::{Source, SourceMap};
use pebblec_parse::lex::{self, Lexer};
use pebblec_parse::{Item, ItemKind, diagnostic};
use std::collections::HashSet;
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};

#[derive(Debug, Default, Clone)]
pub struct CompUnit {
    paths: Vec<OsString>,
    sources: Vec<Source>,
}

impl CompUnit {
    pub fn path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.paths.push(path.as_ref().as_os_str().to_owned());
        self
    }

    pub fn source(mut self, source: Source) -> Self {
        self.sources.push(source);
        self
    }

    pub fn compile<'a>(&mut self, log: bool) -> i32 {
        let (lex_dur, source_map) = Self::record_time(|| {
            let sources = self.sources.drain(..).collect::<Vec<_>>();
            SourceMap::from_paths(&self.paths).map(|mut map| {
                sources
                    .into_iter()
                    .for_each(|s| map.insert(lex::Lexer::new(s).lex().unwrap()));
                map
            })
        });
        let mut source_map = match source_map {
            Ok(source_map) => source_map,
            Err(e) => {
                println!("{e}");
                return 1;
            }
        };

        let lines = source_map
            .buffers()
            .map(|b| b.source().source.lines().count())
            .sum::<usize>();

        let origin = &source_map.buffers().next().unwrap().source().origin.clone();
        let (parse_dur, items) = Self::record_time(|| Self::parse(origin, &mut source_map));
        let mut items = match items {
            Ok(items) => items,
            Err(_) => {
                return 2;
            }
        };
        let mut ctx = Ctx::new(source_map);

        let (lower_dur, result) = Self::record_time(|| ir::lower(&mut ctx, &mut items));
        let (key, const_eval_order) = match result {
            Ok(result) => result,
            Err(_) => return 3,
        };

        let (air_dur, (air_funcs, consts)) =
            Self::record_time(|| air::lower(&ctx, &key, const_eval_order));

        self.log_report(Report {
            lines,
            timing: CompTiming {
                lex: lex_dur,
                parse: parse_dur,
                lower: lower_dur,
                air: air_dur,
            },
        });

        interp::run(&ctx, &air_funcs, &consts, log)
    }

    // TODO: this should be simpler
    pub fn parse<'a>(origin: &OsStr, source_map: &mut SourceMap<'a>) -> Result<Vec<Item>, ()> {
        let mut visited = HashSet::new();

        let mut items = source_map
            .buffers()
            .filter_map(|buf| match pebblec_parse::parse(buf) {
                Ok(items) => Some(items),
                Err(diag) => {
                    diagnostic::report(source_map, diag);
                    None
                }
            })
            .flatten()
            .collect::<Vec<_>>();
        Self::parse_uses(origin, source_map, &mut visited, &mut items)?;

        Ok(items)
    }

    fn parse_uses<'a>(
        origin: &OsStr,
        source_map: &mut SourceMap,
        visited: &mut HashSet<Vec<String>>,
        items: &mut Vec<Item>,
    ) -> Result<(), ()> {
        let uses = items
            .iter()
            .filter_map(|item| match &item.kind {
                ItemKind::Use(uze) => Some((
                    uze.span,
                    uze.path
                        .iter()
                        .map(|step| source_map.buffer(item.source).as_str(step).to_string())
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
                crate::core_path()
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

            match Source::new(path) {
                Err(_) => {
                    diagnostic::report(
                        source_map,
                        Diag::sourced("could not resolve path", Msg::error_span(*span)),
                    );
                    return Err(());
                }
                Ok(source) => {
                    let buffer = Lexer::new(source).lex().unwrap();
                    new_items.extend(pebblec_parse::parse(&buffer).unwrap());
                    source_map.insert(buffer);
                }
            }
        }

        Self::parse_uses(origin, source_map, visited, &mut new_items)?;
        items.extend(new_items);
        Ok(())
    }

    fn record_time<'a, R>(f: impl FnOnce() -> R) -> (f32, R) {
        let start = std::time::Instant::now();
        let result = f();
        let end = std::time::Instant::now()
            .duration_since(start)
            .as_secs_f32();
        (end, result)
    }

    fn log_report(&self, report: Report) {
        println!("{} lines", report.lines);
        self.report_time("lex", report.timing.lex);
        self.report_time("parse", report.timing.parse);
        self.report_time("lower", report.timing.lower);
        self.report_time("air", report.timing.air);
        self.report_time(
            "total",
            report.timing.lex + report.timing.parse + report.timing.lower + report.timing.air,
        );
        println!();
    }

    fn report_time(&self, title: &'static str, time: f32) {
        println!("{:>5} ... {:.4}s", title, time);
    }
}

pub struct Report {
    pub lines: usize,
    pub timing: CompTiming,
}

pub struct CompTiming {
    pub lex: f32,
    pub parse: f32,
    pub lower: f32,
    pub air: f32,
}
