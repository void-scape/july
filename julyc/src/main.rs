use clap::Parser;
use julyc::comp::CompUnit;
use std::process::ExitCode;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Name of the `.jy` file to compile
    #[arg(short, long)]
    file: String,

    /// Log the interpreter
    #[arg(short, long, default_value_t = false)]
    log: bool,
}

#[allow(non_snake_case)]
fn main() -> ExitCode {
    let args = Args::parse();
    match CompUnit::new(args.file) {
        Ok(unit) => match unit.compile(args.log) {
            Ok(exit) => ExitCode::from(exit as u8),
            Err(_) => ExitCode::FAILURE,
        },
        Err(e) => {
            println!("invalid file path: {e}");
            ExitCode::FAILURE
        }
    }
}
