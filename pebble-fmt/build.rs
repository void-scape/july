use quote::{format_ident, quote};
use std::env::current_dir;
use std::ffi::OsString;
use std::fs;
use walkdir::WalkDir;

fn main() {
    println!("cargo::rerun-if-changed=../pebblec/tests/hosted");
    build_hosted_tests();
}

fn build_hosted_tests() {
    let mut path = current_dir().unwrap();
    path.push("../pebblec/tests/hosted");

    let mut tests = Vec::new();
    for entry in WalkDir::new(&path).into_iter() {
        let entry = entry.unwrap();
        if entry.path().is_dir()
            || entry
                .path()
                .extension()
                .is_none_or(|ext| ext != OsString::from("peb"))
        {
            continue;
        }

        let path = entry.path().to_string_lossy().to_string();
        let path_ident = format_ident!(
            "{}",
            &entry
                .path()
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .to_string()
        );
        let path = quote! { #path };

        tests.push(quote! {
            #[test]
            fn #path_ident() {
                assert_eq!(
                    0,
                    pebblec::interp::InterpInstance::new(
                        &pebblec::comp::CompUnit::new(pebblec::comp::Config::default().no_capture(true))
                            .compile_string(#path, pebble_fmt::fmt::fmt(#path).unwrap().unwrap())
                            .unwrap()
                    )
                    .run(false)
                );

                // fmt the fmt
                assert_eq!(
                    0,
                    pebblec::interp::InterpInstance::new(
                        &pebblec::comp::CompUnit::new(pebblec::comp::Config::default().no_capture(true))
                            .compile_string(#path, 
                                pebble_fmt::fmt::fmt_string(pebble_fmt::fmt::fmt(#path).unwrap().unwrap()).unwrap()
                            ).unwrap()
                    )
                    .run(false)
                );
            }
        });
    }

    fs::write(
        "tests/hosted.rs",
        prettyplease::unparse(&syn::parse_file(&quote!(#(#tests)*).to_string()).unwrap()),
    )
    .unwrap();
}
