#[test]
fn cast() {
    assert_eq!(
        0, pebblec::interp::InterpInstance::new(&
        pebblec::comp::CompUnit::new(pebblec::comp::Config::default().no_capture(true))
        .compile("/Users/nicolasball/dev/pebble/pebblec/tests/hosted/cast.peb").unwrap())
        .run(true)
    );
}
#[test]
fn methods() {
    assert_eq!(
        0, pebblec::interp::InterpInstance::new(&
        pebblec::comp::CompUnit::new(pebblec::comp::Config::default().no_capture(true))
        .compile("/Users/nicolasball/dev/pebble/pebblec/tests/hosted/methods.peb")
        .unwrap()).run(true)
    );
}
#[test]
fn general() {
    assert_eq!(
        0, pebblec::interp::InterpInstance::new(&
        pebblec::comp::CompUnit::new(pebblec::comp::Config::default().no_capture(true))
        .compile("/Users/nicolasball/dev/pebble/pebblec/tests/hosted/general.peb")
        .unwrap()).run(true)
    );
}
#[test]
fn slice() {
    assert_eq!(
        0, pebblec::interp::InterpInstance::new(&
        pebblec::comp::CompUnit::new(pebblec::comp::Config::default().no_capture(true))
        .compile("/Users/nicolasball/dev/pebble/pebblec/tests/hosted/slice.peb")
        .unwrap()).run(true)
    );
}
