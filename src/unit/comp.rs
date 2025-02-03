use super::source::Source;
use crate::ir::{block, expr};
use crate::lex::Lexer;
use crate::parse::Parser;
use crate::unit::io;
use crate::{codegen, diagnostic, ir};
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

    pub fn compile(self) {
        let buf = Lexer::new(&self.source).lex().unwrap();
        match Parser::parse(&buf) {
            Ok(items) => {
                let ctx = ir::lower(&buf, &items);
                //if let Err(diag) = expr::validate_bin_ops(&ctx) {
                //    diagnostic::report(diag);
                //    return;
                //}

                //println!("{:#?}", ctx);
                match ctx.key() {
                    Ok(key) => {
                        if let Err(diag) = block::validate_end_exprs(&ctx, &key) {
                            diagnostic::report(diag);
                        } else {
                            io::write("out.o", &codegen::codegen(&ctx, &key)).unwrap();
                        }
                    }
                    Err(diag) => {
                        diagnostic::report(diag);
                    }
                }
            }
            Err(diags) => {
                for diag in diags.into_iter() {
                    diagnostic::report(diag);
                }
            }
        }
    }
}
