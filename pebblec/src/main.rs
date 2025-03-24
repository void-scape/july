#![feature(internal_output_capture)]

use clap::Parser;
use pebblec::comp::CompUnit;
use pebblec_parse::lex::io;
use std::process::{ExitCode, Termination};
use std::sync::Arc;
use std::time::UNIX_EPOCH;

/// Compiler for the Pebble programming language
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Name of the `.peb` file to compile
    #[arg(short, long)]
    file: String,

    /// Do not capture output to stdout
    #[arg(short, long, default_value_t = false)]
    no_capture: bool,

    /// Log the interpreter
    #[arg(short, long, default_value_t = false)]
    log: bool,
}

#[allow(non_snake_case)]
fn main() -> ExitCode {
    let args = Args::parse();

    let (result, captured) = if args.no_capture {
        (
            std::panic::catch_unwind(|| {
                let mut unit = CompUnit::default().path(&args.file);
                unit.compile(args.log)
            }),
            String::new(),
        )
    } else {
        // TODO: don't capture interpreter output
        std::io::set_output_capture(Some(Default::default()));
        let result = std::panic::catch_unwind(|| {
            let mut unit = CompUnit::default().path(&args.file);
            unit.compile(args.log)
        });
        let capture = std::io::set_output_capture(None);
        let captured = capture.unwrap();
        let captured = Arc::try_unwrap(captured).unwrap();
        let captured = captured.into_inner().unwrap();
        (result, String::from_utf8(captured).unwrap())
    };

    print!("{captured}");
    match result {
        Ok(exit_code) => ExitCode::from(exit_code as u8),
        Err(_) => {
            match io::write(
                format!(
                    "ICE-{}",
                    std::time::SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs()
                ),
                captured.as_bytes(),
            ) {
                Ok(_) => ExitCode::FAILURE,
                Err(io_err) => {
                    println!("failed to report ICE: {io_err}");
                    ExitCode::FAILURE
                }
            }
        }
    }
}
