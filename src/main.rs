use self::unit::comp::CompUnit;
use std::process::Command;

mod codegen;
mod diagnostic;
mod ir;
mod lex;
mod parse;
mod unit;

fn main() {
    let file = std::env::args().nth(1).expect("no input file given");
    let unit = CompUnit::new(file).unwrap();
    unit.compile();

    run();
}

fn run() {
    //otool -x -v out.o
    let output = Command::new("otool")
        .arg("-x")
        .arg("-v")
        .arg("out.o")
        .output()
        .unwrap();
    println!(
        "{}",
        output.stdout.iter().map(|c| *c as char).collect::<String>()
    );
    println!(
        "{}",
        output.stderr.iter().map(|c| *c as char).collect::<String>()
    );

    //ld out.o -o out -macosx_version_min 11.0 -L /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib -lSystem
    let output = Command::new("ld")
        .arg("out.o")
        .arg("-o")
        .arg("out")
        .arg("-macosx_version_min")
        .arg("11.0")
        .arg("-L")
        .arg("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib")
        .arg("-l")
        .arg("System")
        .output()
        .unwrap();
    if !output.stdout.is_empty() {
        println!(
            "ld: {}",
            output.stdout.iter().map(|c| *c as char).collect::<String>()
        );
    }
    if !output.stderr.is_empty() {
        println!(
            "ld: {}",
            output.stderr.iter().map(|c| *c as char).collect::<String>()
        );
    }

    let output = Command::new("./out").output().unwrap();
    println!("out: {}", output.status);
}
