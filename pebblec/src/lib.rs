#![feature(str_from_raw_parts)]

use std::path::PathBuf;
use std::str::FromStr;

pub mod air;
pub mod comp;
pub mod interp;
pub mod ir;
#[cfg(test)]
mod pebblec_tests;

pub fn core_path() -> PathBuf {
    let mut path = std::env::current_exe().unwrap();
    while !path.ends_with("pebble") {
        path.pop();
    }
    //path.push("core");
    path
}
