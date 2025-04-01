#[test]
fn cast() {
    assert_eq!(
        0, pebblec::interp::InterpInstance::new(&
        pebblec::comp::CompUnit::new(pebblec::comp::Config::default().no_capture(true))
        .compile("/Users/nicolasball/dev/pebble/pebblec/tests/hosted/cast.peb").unwrap())
        .run(false)
    );
}
#[test]
fn general() {
    assert_eq!(
        0, pebblec::interp::InterpInstance::new(&
        pebblec::comp::CompUnit::new(pebblec::comp::Config::default().no_capture(true))
        .compile("/Users/nicolasball/dev/pebble/pebblec/tests/hosted/general.peb")
        .unwrap()).run(false)
    );
}
#[test]
fn slice() {
    assert_eq!(
        0, pebblec::interp::InterpInstance::new(&
        pebblec::comp::CompUnit::new(pebblec::comp::Config::default().no_capture(true))
        .compile("/Users/nicolasball/dev/pebble/pebblec/tests/hosted/slice.peb")
        .unwrap()).run(false)
    );
}
