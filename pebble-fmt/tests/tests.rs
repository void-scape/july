use pebble_fmt::fmt;
use pebblec::comp::{CompUnit, Config};

const INVADERS: &str = "../demo/invaders/invaders.peb";

#[test]
fn invaders_codegen() {
    let unfmt = CompUnit::new(Config::default().no_capture(true))
        .compile(INVADERS)
        .unwrap();
    let fmt = CompUnit::new(Config::default().no_capture(true))
        .compile_string(INVADERS, fmt::fmt(INVADERS).unwrap().unwrap())
        .unwrap();

    // TyStore stores span information
    assert_eq!(unfmt.extern_sigs, fmt.extern_sigs);
    assert_eq!(unfmt.consts, fmt.consts);
    assert_eq!(unfmt.funcs, fmt.funcs);
}
