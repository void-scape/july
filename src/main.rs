#![feature(maybe_uninit_slice)]
#![feature(str_from_raw_parts)]

use self::unit::comp::CompUnit;
use clap::Parser;

mod air;
mod arena;
mod codegen;
mod diagnostic;
mod interp;
mod ir;
mod lex;
mod parse;
mod unit;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Name of the `.jy` file to compile
    #[arg(short, long)]
    file: String,

    /// Log the interpretor
    #[arg(short, long, default_value_t = false)]
    log: bool,
}

#[allow(non_snake_case)]
fn main() {
    let args = Args::parse();
    let unit = CompUnit::new(args.file).expect("invalid file path");
    unit.compile(args.log);
}
