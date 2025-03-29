use crate::comp::{CompUnit, Config};
use crate::interp::InterpInstance;

const TESTS: &str = "../demo/tests.peb";

#[test]
fn language_tests() {
    for _ in 0..100 {
        assert_eq!(
            0,
            InterpInstance::new(
                &CompUnit::new(Config::default().no_capture(true))
                    .compile(TESTS)
                    .unwrap()
            )
            .run(false)
        );
    }
}

#[test]
fn deterministic() {
    let first = CompUnit::new(Config::default().no_capture(true))
        .compile(TESTS)
        .unwrap();
    let second = CompUnit::new(Config::default().no_capture(true))
        .compile(TESTS)
        .unwrap();

    // TyStore stores span information
    assert_eq!(first.extern_sigs, second.extern_sigs);
    assert_eq!(first.consts, second.consts);
    assert_eq!(first.funcs, second.funcs);
}
