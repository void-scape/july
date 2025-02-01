use self::lex::Lexer;
use self::parse::*;
use self::source::Source;

mod diagnostic;
mod ir;
mod lex;
mod parse;
mod recon;
mod source;

fn main() {
    let file = std::env::args().nth(1).expect("no input file given");
    let source = std::fs::read_to_string(&file).expect("failed to read file");
    let source = Source::new(source, file);
    let buf = Lexer::new(&source).lex().unwrap();

    match Parser::parse(&buf) {
        Ok(_) => {}
        Err(diags) => {
            for diag in diags.into_iter() {
                diagnostic::report(diag);
            }
        }
    }
}
