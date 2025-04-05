#[test]
fn cast() {
    assert_eq!(
        0, pebblec::interp::InterpInstance::new(&
        pebblec::comp::CompUnit::new(pebblec::comp::Config::default().no_capture(true))
        .compile_string("/Users/nicolasball/dev/pebble/pebble-fmt/../pebblec/tests/hosted/cast.peb",
        pebble_fmt::fmt::fmt("/Users/nicolasball/dev/pebble/pebble-fmt/../pebblec/tests/hosted/cast.peb")
        .unwrap().unwrap()).unwrap()).run(false)
    );
    assert_eq!(
        0, pebblec::interp::InterpInstance::new(&
        pebblec::comp::CompUnit::new(pebblec::comp::Config::default().no_capture(true))
        .compile_string("/Users/nicolasball/dev/pebble/pebble-fmt/../pebblec/tests/hosted/cast.peb",
        pebble_fmt::fmt::fmt_string(pebble_fmt::fmt::fmt("/Users/nicolasball/dev/pebble/pebble-fmt/../pebblec/tests/hosted/cast.peb")
        .unwrap().unwrap()).unwrap()).unwrap()).run(false)
    );
}
#[test]
fn methods() {
    assert_eq!(
        0, pebblec::interp::InterpInstance::new(&
        pebblec::comp::CompUnit::new(pebblec::comp::Config::default().no_capture(true))
        .compile_string("/Users/nicolasball/dev/pebble/pebble-fmt/../pebblec/tests/hosted/methods.peb",
        pebble_fmt::fmt::fmt("/Users/nicolasball/dev/pebble/pebble-fmt/../pebblec/tests/hosted/methods.peb")
        .unwrap().unwrap()).unwrap()).run(false)
    );
    assert_eq!(
        0, pebblec::interp::InterpInstance::new(&
        pebblec::comp::CompUnit::new(pebblec::comp::Config::default().no_capture(true))
        .compile_string("/Users/nicolasball/dev/pebble/pebble-fmt/../pebblec/tests/hosted/methods.peb",
        pebble_fmt::fmt::fmt_string(pebble_fmt::fmt::fmt("/Users/nicolasball/dev/pebble/pebble-fmt/../pebblec/tests/hosted/methods.peb")
        .unwrap().unwrap()).unwrap()).unwrap()).run(false)
    );
}
#[test]
fn general() {
    assert_eq!(
        0, pebblec::interp::InterpInstance::new(&
        pebblec::comp::CompUnit::new(pebblec::comp::Config::default().no_capture(true))
        .compile_string("/Users/nicolasball/dev/pebble/pebble-fmt/../pebblec/tests/hosted/general.peb",
        pebble_fmt::fmt::fmt("/Users/nicolasball/dev/pebble/pebble-fmt/../pebblec/tests/hosted/general.peb")
        .unwrap().unwrap()).unwrap()).run(false)
    );
    assert_eq!(
        0, pebblec::interp::InterpInstance::new(&
        pebblec::comp::CompUnit::new(pebblec::comp::Config::default().no_capture(true))
        .compile_string("/Users/nicolasball/dev/pebble/pebble-fmt/../pebblec/tests/hosted/general.peb",
        pebble_fmt::fmt::fmt_string(pebble_fmt::fmt::fmt("/Users/nicolasball/dev/pebble/pebble-fmt/../pebblec/tests/hosted/general.peb")
        .unwrap().unwrap()).unwrap()).unwrap()).run(false)
    );
}
#[test]
fn slice() {
    assert_eq!(
        0, pebblec::interp::InterpInstance::new(&
        pebblec::comp::CompUnit::new(pebblec::comp::Config::default().no_capture(true))
        .compile_string("/Users/nicolasball/dev/pebble/pebble-fmt/../pebblec/tests/hosted/slice.peb",
        pebble_fmt::fmt::fmt("/Users/nicolasball/dev/pebble/pebble-fmt/../pebblec/tests/hosted/slice.peb")
        .unwrap().unwrap()).unwrap()).run(false)
    );
    assert_eq!(
        0, pebblec::interp::InterpInstance::new(&
        pebblec::comp::CompUnit::new(pebblec::comp::Config::default().no_capture(true))
        .compile_string("/Users/nicolasball/dev/pebble/pebble-fmt/../pebblec/tests/hosted/slice.peb",
        pebble_fmt::fmt::fmt_string(pebble_fmt::fmt::fmt("/Users/nicolasball/dev/pebble/pebble-fmt/../pebblec/tests/hosted/slice.peb")
        .unwrap().unwrap()).unwrap()).unwrap()).run(false)
    );
}
