#![feature(maybe_uninit_slice)]

use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};

use self::unit::comp::CompUnit;
use clap::Parser;
use libffi::low::CodePtr;
use libffi::middle::{Arg, Cif, Type};

mod air;
mod arena;
mod codegen;
mod diagnostic;
mod interp;
mod ir;
mod lex;
mod parse;
mod unit;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Name of the `.jy` file to compile
    #[arg(short, long)]
    file: String,

    /// Log the interpretor
    #[arg(short, long, default_value_t = false)]
    log: bool,
}

#[repr(C)]
struct SDL_Window {
    id: u32,
    title: *mut c_char,
}

#[repr(C)]
struct SDL_Renderer {
    id: u32,
    title: *mut c_char,
}

#[repr(C)]
struct SDL_Event {
    data: [u8; 128],
}

#[allow(non_snake_case)]
fn main() {
    //unsafe {
    //    let lib = libloading::Library::new("/Users/nicolasball/SDL/SDL3.framework/SDL3").unwrap();
    //
    //    const SDL_INIT_VIDEO: u32 = 0x00000020;
    //    let SDL_Init: libloading::Symbol<*mut c_void> = lib.get(b"SDL_Init").unwrap();
    //    let cif = Cif::new([Type::u32()], Type::u8());
    //    let result = cif.call::<u8>(
    //        CodePtr(SDL_Init.into_raw().as_raw_ptr()),
    //        &[Arg::new(&SDL_INIT_VIDEO)],
    //    );
    //    assert!(std::mem::transmute::<u8, bool>(result));
    //
    //    let SDL_CreateWindow: libloading::Symbol<
    //        unsafe extern "C" fn(*const c_char, c_int, c_int, u64) -> *mut SDL_Window,
    //    > = lib.get(b"SDL_CreateWindow").unwrap();
    //    let cif = Cif::new(
    //        [Type::u8(), Type::c_int(), Type::c_int(), Type::u64()],
    //        Type::pointer(),
    //    );
    //    let result = cif.call::<*mut c_void>(
    //        CodePtr(SDL_CreateWindow.into_raw().as_raw_ptr()),
    //        &[Arg::new(&0), Arg::new(&640), Arg::new(&480), Arg::new(&0)],
    //    );
    //    let window = result as *mut SDL_Window;
    //    assert!(!window.is_null());
    //    println!("{:?}", window.addr());
    //
    //    let SDL_CreateRenderer: libloading::Symbol<
    //        unsafe extern "C" fn(*mut SDL_Window, *const c_char) -> *mut SDL_Renderer,
    //    > = lib.get(b"SDL_CreateRenderer").unwrap();
    //    let renderer = SDL_CreateRenderer(window, std::ptr::null_mut());
    //    assert!(!renderer.is_null());
    //
    //    //let SDL_GetError: libloading::Symbol<unsafe extern "C" fn() -> *const c_char> =
    //    //    lib.get(b"SDL_GetError").unwrap();
    //    //
    //    //let str = CStr::from_ptr(SDL_GetError());
    //    //if !str.is_empty() {
    //    //    println!("{:?}", str);
    //    //}
    //
    //    let SDL_PollEvent: libloading::Symbol<unsafe extern "C" fn(*mut SDL_Event) -> bool> =
    //        lib.get(b"SDL_PollEvent").unwrap();
    //
    //    //let SDL_RenderClear: libloading::Symbol<unsafe extern "C" fn(*mut SDL_Renderer) -> bool> =
    //    //    lib.get(b"SDL_RenderClear").unwrap();
    //    //let SDL_RenderPresent: libloading::Symbol<unsafe extern "C" fn(*mut SDL_Renderer) -> bool> =
    //    //    lib.get(b"SDL_RenderPresent").unwrap();
    //    let SDL_Delay: libloading::Symbol<unsafe extern "C" fn(u32)> =
    //        lib.get(b"SDL_Delay").unwrap();
    //
    //    let mut event = SDL_Event { data: [0; 128] };
    //
    //    loop {
    //        SDL_PollEvent(&mut event);
    //
    //        let ty =
    //            std::mem::transmute::<[u8; 4], u32>(event.data[..4].try_into().unwrap()) as u32;
    //        if ty == 0x100 {
    //            break;
    //        }
    //
    //        //assert!(SDL_RenderClear(renderer));
    //        //assert!(SDL_RenderPresent(renderer));
    //
    //        SDL_Delay(16);
    //    }
    //
    //    // SDL_Renderer * SDL_CreateRenderer(SDL_Window *window, const char *name);
    //
    //    //loop {}
    //}

    let args = Args::parse();
    let unit = CompUnit::new(args.file).expect("invalid file path");
    unit.compile(args.log);
}

//fn run() {
//    //otool -x -v out.o
//    //let output = Command::new("otool")
//    //    .arg("-x")
//    //    .arg("-v")
//    //    .arg("out.o")
//    //    .output()
//    //    .unwrap();
//    //println!(
//    //    "{}",
//    //    output.stdout.iter().map(|c| *c as char).collect::<String>()
//    //);
//    //println!(
//    //    "{}",
//    //    output.stderr.iter().map(|c| *c as char).collect::<String>()
//    //);
//
//    //ld out.o -o out -macosx_version_min 11.0 -L /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib -lSystem
//    //let output = Command::new("ld")
//    //    .arg("out.o")
//    //    .arg("-o")
//    //    .arg("out")
//    //    .arg("-macosx_version_min")
//    //    .arg("11.0")
//    //    .arg("-L")
//    //    .arg("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib")
//    //    .arg("-l")
//    //    .arg("System")
//    //    .output()
//    //    .unwrap();
//    //if !output.stdout.is_empty() {
//    //    println!(
//    //        "ld: {}",
//    //        output.stdout.iter().map(|c| *c as char).collect::<String>()
//    //    );
//    //}
//    //if !output.stderr.is_empty() {
//    //    println!(
//    //        "ld: {}",
//    //        output.stderr.iter().map(|c| *c as char).collect::<String>()
//    //    );
//    //}
//
//    let output = Command::new("clang")
//        .arg("out.o")
//        .arg("-o")
//        .arg("out")
//        .output()
//        .unwrap();
//    if !output.stdout.is_empty() {
//        println!(
//            "ld: {}",
//            output.stdout.iter().map(|c| *c as char).collect::<String>()
//        );
//    }
//    if !output.stderr.is_empty() {
//        println!(
//            "ld: {}",
//            output.stderr.iter().map(|c| *c as char).collect::<String>()
//        );
//    }
//
//    let output = Command::new("./out").output().unwrap();
//    println!("out: {}", output.status);
//}
