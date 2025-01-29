use self::ir::prelude::Ctx;
use self::lex::Lexer;
use self::parse::rule::*;
use self::parse::*;
use self::source::Source;

mod diagnostic;
mod ir;
mod lex;
mod parse;
mod source;

fn main() {
    let file = std::env::args().nth(1).expect("no input file given");
    let source = std::fs::read_to_string(&file).expect("failed to read file");
    //let source = String::from("let x");
    let source = Source::new(source, file);
    let buf = Lexer::new(&source).lex().unwrap();
    //println!(
    //    "{:#?}",
    //    buf.tokens().map(|t| buf.kind(t)).collect::<Vec<_>>()
    //);

    let mut ctx = Ctx::new(&buf);
    match Parser::new(&buf, &mut ctx) {
        Ok(parser) => {
            for _ in parser.post_order.iter() {
                //println!("visited node: {:#?}", buf.kind(*token));
            }
        }
        Err(diag) => {
            diagnostic::report(diag);
        }
    }
}
