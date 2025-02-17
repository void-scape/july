use super::source::Source;
use crate::diagnostic::Diag;
use crate::lex::buffer::TokenBuffer;
use crate::lex::Lexer;
use crate::parse::Parser;
use crate::unit::io;
use crate::{air, codegen, diagnostic, interp, ir};
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

    pub fn compile<'a>(&'a self) -> Result<(), ()> {
        fn compile<'a>(buf: &'a TokenBuffer<'a>) -> Result<(), Vec<Diag<'a>>> {
            let items = Parser::parse(&buf)?;
            let ctx = ir::lower(&buf, &items).map_err(|_| Vec::new())?;
            let key = ir::resolve_types(&ctx)?;
            let air_funcs = ctx
                .funcs
                .iter()
                .map(|func| air::lower_func(&ctx, &key, func))
                .collect::<Vec<_>>();
            let exit = interp::run(&ctx, &air_funcs).unwrap();
            println!("exit: {exit}");

            //io::write("out.o", &codegen::codegen(&ctx, &key, &air_funcs)).unwrap();
            Ok(())
        }

        let buf = Lexer::new(&self.source).lex().unwrap();
        if let Err(diags) = compile(&buf) {
            for diag in diags.into_iter() {
                diagnostic::report(diag);
            }
            return Err(());
        }

        Ok(())
    }
}
