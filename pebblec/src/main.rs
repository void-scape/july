use clap::Parser;
use pebblec::comp::{CompErr, CompUnit, Config};
use pebblec::interp::InterpInstance;
use std::process::ExitCode;

/// Pebble Compiler
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// path to a `.peb` file
    #[arg(short, long, default_value_t = String::from("hi"))]
    file: String,

    /// log the interpreter
    #[arg(short, long, default_value_t = false)]
    log: bool,
}

impl Args {
    pub fn config(&self) -> Config {
        Config { log: self.log }
    }
}

fn main() -> ExitCode {
    let args = Args::parse();
    match CompUnit::new(args.config()).compile(args.file) {
        Ok(bytecode) => {
            let exit_code = InterpInstance::new(&bytecode).run(args.log);
            ExitCode::from(exit_code as u8)
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
