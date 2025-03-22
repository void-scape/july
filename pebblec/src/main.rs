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

    /// Log the interpreter
    #[arg(short, long, default_value_t = false)]
    log: bool,
}

#[allow(non_snake_case)]
fn main() -> ExitCode {
    let args = Args::parse();
    std::io::set_output_capture(Some(Default::default()));
    let result =
        std::panic::catch_unwind(|| CompUnit::new(&args.file).map(|unit| unit.compile(args.log)));
    let capture = std::io::set_output_capture(None);
    let captured = capture.unwrap();
    let captured = Arc::try_unwrap(captured).unwrap();
    let captured = captured.into_inner().unwrap();
    let captured = String::from_utf8(captured).unwrap();

    let unit_result = match result {
        Ok(result) => result,
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
                Ok(_) => return ExitCode::FAILURE,
                Err(io_err) => {
                    println!("failed to report ICE: {io_err}");
                    println!("ICE: {captured}");
                    return ExitCode::FAILURE;
                }
            }
        }
    };

    let compiler_result = match unit_result {
        Ok(result) => result,
        Err(io_err) => {
            println!("failed to open `{}`: {io_err}", args.file);
            return ExitCode::FAILURE;
        }
    };

    match compiler_result {
        Ok(exit_code) => ExitCode::from(exit_code as u8),
        Err(_) => {
            print!("{captured}");
            ExitCode::FAILURE
        }
    }
}
