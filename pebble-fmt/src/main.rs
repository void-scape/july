#![warn(clippy::pedantic)]
#![feature(internal_output_capture)]

use clap::Parser;
use pebble_fmt::fmt;
use pebblec_parse::lex::io;
use std::panic;
use std::process::ExitCode;
use std::sync::Arc;

/// Pebble Formatter
#[derive(Parser, Debug)]
#[command(about, long_about = None)]
struct Args {
    /// path to a `.peb` file
    file: String,
}

fn main() -> ExitCode {
    let args = Args::parse();

    std::io::set_output_capture(Some(Default::default()));
    let result = panic::catch_unwind(|| fmt::fmt(&args.file));
    let capture = std::io::set_output_capture(None);

    let fmt_result = match result {
        Ok(result) => result,
        Err(_) => {
            let captured = capture.unwrap();
            let captured = Arc::try_unwrap(captured).unwrap();
            let captured = captured.into_inner().unwrap();
            let captured = String::from_utf8(captured).unwrap();
            println!("{}", captured);
            println!("note: unexpected panic, this is a bug");
            return ExitCode::FAILURE;
        }
    };

    match fmt_result {
        Ok(fmted) => match fmted {
            Some(fmted) => {
                // if it read the file, I assume we can write to it
                io::write(&args.file, fmted.as_bytes()).unwrap();
                ExitCode::SUCCESS
            }
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
