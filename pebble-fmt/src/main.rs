#![feature(internal_output_capture)]

use clap::Parser;
use pebblec_parse::lex::io;
use std::panic;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::UNIX_EPOCH;

mod fmt;
#[cfg(test)]
mod fmt_tests;
mod node;

/// Formatter for the Pebble programming language
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Name of the `.peb` file to format
    #[arg(short, long)]
    file: String,
}

fn main() -> ExitCode {
    let args = Args::parse();

    //std::io::set_output_capture(Some(Default::default()));
    let result = panic::catch_unwind(|| fmt::fmt(&args.file));
    let capture = std::io::set_output_capture(None);

    let fmt_result = match result {
        Ok(result) => result,
        Err(_) => {
            let captured = capture.unwrap();
            let captured = Arc::try_unwrap(captured).unwrap();
            let captured = captured.into_inner().unwrap();
            let captured = String::from_utf8(captured).unwrap();

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

    match fmt_result {
        Ok(fmted) => match fmted {
            Some(fmted) => match io::write(&args.file, fmted.as_bytes()) {
                Ok(_) => {
                    return ExitCode::SUCCESS;
                }
                Err(e) => {
                    println!("Failed to write output: {e}");
                    return ExitCode::FAILURE;
                }
            },
            None => {
                println!("failed to format `{}`", args.file);
                return ExitCode::FAILURE;
            }
        },
        Err(e) => {
            println!("failed to load `{}`: {e}", args.file);
            return ExitCode::FAILURE;
        }
    }
}
