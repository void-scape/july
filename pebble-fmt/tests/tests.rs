use pebble_fmt::fmt;
use pebblec::comp::{CompUnit, Config};
use pebblec::interp::InterpInstance;

const INVADERS: &str = "tests/invaders_unformatted.peb";
const TESTS: &str = "tests/tests_unformatted.peb";

#[test]
fn deterministic() {
    let outputs = (0..100)
        .map(|_| fmt::fmt(INVADERS).unwrap().unwrap())
        .collect::<Vec<_>>();
    let first = outputs.first().unwrap();
    assert!(outputs.iter().all(|o| o == first));
}

#[test]
fn fmt_the_fmt() {
    let first = fmt::fmt(INVADERS).unwrap().unwrap();
    let second = fmt::fmt_string(first.clone()).unwrap();
    let third = fmt::fmt_string(second.clone()).unwrap();
    assert_eq!(first, second);
    assert_eq!(second, third);
}

#[test]
fn codegen() {
    codegen_with(INVADERS);
    codegen_with(TESTS);
}

#[test]
fn language_tests() {
    assert_eq!(
        0,
        InterpInstance::new(
            &CompUnit::new(Config::default().no_capture(true))
                .compile(TESTS)
                .unwrap()
        )
        .run(false)
    );
    assert_eq!(
        0,
        InterpInstance::new(
            &CompUnit::new(Config::default().no_capture(true))
                .compile_string(TESTS, fmt::fmt(TESTS).unwrap().unwrap())
                .unwrap()
        )
        .run(false)
    );
}

fn codegen_with(path: &str) {
    let unfmt = CompUnit::new(Config::default().no_capture(true))
        .compile(path)
        .unwrap();
    let fmt = CompUnit::new(Config::default().no_capture(true))
        .compile_string(TESTS, fmt::fmt(path).unwrap().unwrap())
        .unwrap();

    // TyStore stores span information
    assert_eq!(unfmt.extern_sigs, fmt.extern_sigs);
    assert_eq!(unfmt.consts, fmt.consts);
    assert_eq!(unfmt.funcs, fmt.funcs);
}
