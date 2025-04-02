use pebblec::comp::{CompUnit, Config};
use pebblec_parse::lex::io;

mod hosted;

const INVADERS: &str = "../demo/invaders/invaders.peb";

#[test]
fn deterministic() {
    let first = CompUnit::new(Config::default().no_capture(true))
        .compile(INVADERS)
        .unwrap();
    let second = CompUnit::new(Config::default().no_capture(true))
        .compile(INVADERS)
        .unwrap();

    // TyStore stores span information
    assert_eq!(first.extern_sigs, second.extern_sigs);
    assert_eq!(first.consts, second.consts);
    assert_eq!(first.funcs, second.funcs);
}

#[test]
fn core() {
    for entry in walkdir::WalkDir::new("../core") {
        let path = entry.as_ref().unwrap().path();
        if path.is_dir() {
            continue;
        }

        let mut file_str = io::read_string(path).unwrap();

        // add a dummy main so that file can be compiled as a unit
        file_str.push_str("main: () {}");

        assert!(
            CompUnit::new(Config::default().no_capture(true))
                .compile_string(path, file_str)
                .is_ok()
        )
    }
}
