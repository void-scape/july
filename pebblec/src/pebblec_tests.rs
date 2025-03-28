use crate::comp::CompUnit;
use crate::interp::InterpInstance;

const TESTS: &str = "../demo/tests.peb";

#[test]
fn language_tests() {
    for _ in 0..100 {
        assert_eq!(
            0,
            InterpInstance::new(&CompUnit::default().panicking_compile(TESTS).unwrap()).run(false)
        );
    }
}

#[test]
fn deterministic() {
    let first = CompUnit::default().panicking_compile(TESTS).unwrap();
    let second = CompUnit::default().panicking_compile(TESTS).unwrap();

    // TyStore stores span information
    assert_eq!(first.extern_sigs, second.extern_sigs);
    assert_eq!(first.consts, second.consts);
    assert_eq!(first.funcs, second.funcs);
}
