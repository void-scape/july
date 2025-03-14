use super::source::Source;
use crate::air::ctx::AirCtx;
use crate::lex::buffer::TokenBuffer;
use crate::lex::Lexer;
use crate::parse::Parser;
use crate::{air, interp, ir};
use std::path::Path;

pub struct CompUnit {
    source: Source,
}

impl CompUnit {
    pub fn new<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        Ok(Self {
            source: Source::new(path)?,
        })
    }

    pub fn compile(&self, log: bool) -> Result<i32, ()> {
        let (lex_dur, buf) =
            self.record_time::<TokenBuffer>(|unit| Lexer::new(&unit.source).lex().unwrap());
        let (parse_dur, items) = self.record_time(|_| Parser::parse(&buf));
        let items = items?;
        let (lower_dur, ir) = self.record_time(|_| ir::lower(&buf, &items));
        let (ctx, key, const_eval_order) = ir?;

        let mut air_ctx = AirCtx::new(&ctx, &key);
        let (air_dur, (air_funcs, consts)) = self.record_time(|_| {
            let consts = const_eval_order
                .into_iter()
                .flat_map(|id| air::lower_const(&mut air_ctx, ctx.tys.get_const(id).unwrap()))
                .collect::<Vec<_>>();
            let air_funcs = ctx
                .funcs
                .iter()
                .map(|func| air::lower_func(&mut air_ctx, func))
                .collect::<Vec<_>>();
            (air_funcs, consts)
        });

        let lines = buf.source().raw().lines().count();
        println!("{lines} lines");
        self.report_time("lex", lex_dur);
        self.report_time("parse", parse_dur);
        self.report_time("lower", lower_dur);
        self.report_time("air", air_dur);
        self.report_time("total", lex_dur + parse_dur + lower_dur + air_dur);
        println!();

        Ok(interp::run(&ctx, &air_funcs, &consts, log))
    }

    fn record_time<'a, R>(&'a self, f: impl FnOnce(&'a Self) -> R) -> (f32, R) {
        let start = std::time::Instant::now();
        let result = f(self);
        let end = std::time::Instant::now()
            .duration_since(start)
            .as_secs_f32();
        (end, result)
    }

    fn report_time(&self, title: &'static str, time: f32) {
        println!("{:>5} ... {:.4}s", title, time);
    }
}
