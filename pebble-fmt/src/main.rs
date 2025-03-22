use clap::Parser;
use pebblec_parse::lex::io;

mod fmt;
mod node;
#[cfg(test)]
mod fmt_tests;

/// Formatter for the July programming language
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Name of the `.jy` file to format
    #[arg(short, long)]
    file: String,
}

fn main() {
    let args = Args::parse();
    match fmt::fmt(&args.file) {
        Ok(fmted) => {
            let tmp_name = format!("{}_tmp", args.file);
            match fmted {
                Some(fmted) => match io::write(&tmp_name, fmted.as_bytes()) {
                    Ok(_) => {
                        std::fs::rename(&tmp_name, &args.file).unwrap_or_else(|e| panic!("{e}"))
                    }
                    Err(e) => println!("Failed to write output: {e}"),
                },
                None => println!("failed to format `{}`", args.file),
            }
        }
        Err(e) => println!("failed to load `{}`: {e}", args.file),
    }
}
