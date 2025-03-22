use super::*;
use julyc::comp::CompUnit;
use julyc_parse::lex::source::Source;

const INVADERS: &str = "res/invaders_unformatted.jy";
const TESTS: &str = "res/tests_unformatted.jy";

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

// codegen appears to be non deterministic?
#[test]
fn codegen() {}

#[test]
fn language_tests() {
    assert_eq!(0, CompUnit::new(TESTS).unwrap().compile(false).unwrap());
    assert_eq!(
        0,
        CompUnit::with_source(Source::from_string(
            "tests",
            fmt::fmt(TESTS).unwrap().unwrap()
        ))
        .compile(false)
        .unwrap()
    );
}
