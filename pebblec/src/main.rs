use clap::Parser;
use pebblec::comp::{CompErr, CompUnit, Config};
use pebblec::interp::InterpInstance;
use std::process::ExitCode;

/// Pebble Compiler
#[derive(Parser, Debug)]
#[command(about, long_about = None)]
struct Args {
    /// path to a `.peb` file
    file: String,

    /// do not capture stdout during compilation
    #[arg(short, long, default_value_t = false)]
    no_capture: bool,

    /// log the interpreter
    #[arg(short, long, default_value_t = false)]
    log: bool,
}

impl Args {
    pub fn config(&self) -> Config {
        Config {
            log: self.log,
            no_capture: self.no_capture,
        }
    }
}

fn main() -> ExitCode {
    let args = Args::parse();
    match CompUnit::new(args.config()).compile(args.file) {
        Ok(bytecode) => {
            ExitCode::from(InterpInstance::new(&bytecode).run(args.log) as u8)
        }
        Err(err) => {
            match err {
                CompErr::Source(err) => {
                    println!("{err}");
                }
                CompErr::Ir | CompErr::Panic => {}
            }
            ExitCode::FAILURE
        }
    }
}
