use clap::Parser;
use pebblec::comp::{CompErr, CompUnit, Config};
use pebblec::interp::InterpInstance;
use std::process::ExitCode;

/// Pebble Compiler
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// path to a `.peb` file
    #[arg(short, long)]
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
            // TODO: if the interpreter panics, then we should report it (ICE). Buuut, the user code
            // could also crash the interpreter... annoying
            ExitCode::from(InterpInstance::new(&bytecode).run(args.log) as u8)
            //match ice::reported_panic(false, move || InterpInstance::new(&bytecode).run(args.log)) {
            //    Some(exit_code) => ExitCode::from(exit_code as u8),
            //    None => ExitCode::FAILURE,
            //}
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
