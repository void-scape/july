use self::ctx::InterpCtx;
use crate::air::{Air, AirFunc, Bits, ConstData, IntKind, OffsetVar};
use crate::ir::ctx::Ctx;
use crate::ir::sig::{Linkage, Sig};
use crate::ir::ty::store::TyId;
use crate::ir::ty::{FloatTy, Ty, Width};
use anstream::println;
use core::str;
use libffi::low::CodePtr;
use libffi::middle::{Arg, Cif, Type};
use std::collections::{HashMap, HashSet};
use std::ffi::{c_char, c_void};
use std::ops::BitXor;

mod ctx;
mod stack;

pub fn run<'a>(ctx: &'a Ctx<'a>, air_funcs: &[AirFunc], consts: &[Air<'a>], log: bool) -> i32 {
    let libs = load_libraries(ctx.sigs.values().copied());

    let main = air_funcs
        .iter()
        .find(|f| ctx.expect_ident(f.func.sig.ident) == "main")
        .unwrap();
    entry(ctx, main, air_funcs, &libs, consts, log)
}

fn load_libraries<'a>(
    sigs: impl Iterator<Item = &'a Sig<'a>>,
) -> HashMap<&'a str, libloading::Library> {
    let mut links = HashSet::<&str>::default();
    for sig in sigs {
        match sig.linkage {
            Linkage::External { link } => {
                links.insert(link);
            }
            _ => {}
        }
    }

    links
        .into_iter()
        .map(|l| (l, unsafe { libloading::Library::new(l).unwrap() }))
        .collect()
}

fn entry(
    ctx: &Ctx,
    main: &AirFunc,
    air_funcs: &[AirFunc],
    libs: &HashMap<&str, libloading::Library>,
    consts: &[Air],
    log: bool,
) -> i32 {
    let mut ctx = InterpCtx::new(ctx);
    ctx.consts(consts);
    loop {
        match execute(&mut ctx, air_funcs, libs, log) {
            InstrResult::Break => break,
            InstrResult::Continue => continue,
            InstrResult::Ok => {}
        }

        ctx.incr_instr();
    }

    ctx.start_func(main);
    loop {
        match execute(&mut ctx, air_funcs, libs, log) {
            InstrResult::Break => break,
            InstrResult::Continue => continue,
            InstrResult::Ok => {}
        }

        ctx.incr_instr();
    }

    println!("\nstack: {:?} bytes", ctx.stack.len() * 8);
    ctx.a.r() as i32
}

macro_rules! float_op {
    ($ctx:expr, $width:expr, $op:tt) => {
        match $width {
            Width::W32 => {
                $ctx.a.w(
                    (f32::from_bits($ctx.a.r() as u32) $op f32::from_bits($ctx.b.r() as u32)).to_bits()
                        as u64,
                );
            }
            Width::W64 => {
                $ctx.a
                    .w((f64::from_bits($ctx.a.r()) $op f64::from_bits($ctx.b.r())).to_bits());
            }
            _ => unreachable!(),
        }
    };

    (Cmp, $ctx:expr, $width:expr, $op:tt) => {
        match $width {
            Width::W32 => {
                $ctx.a.w(
                    (f32::from_bits($ctx.a.r() as u32) $op f32::from_bits($ctx.b.r() as u32)) as u64,
                );
            }
            Width::W64 => {
                $ctx.a
                    .w((f64::from_bits($ctx.a.r()) $op f64::from_bits($ctx.b.r())) as u64);
            }
            _ => unreachable!(),
        }
    };
}

macro_rules! int_op {
    ($ctx:expr, $width:expr, $op:tt) => {
        match $width {
            Width::W8 => $ctx.a.w(($ctx.a.r() as u8 $op $ctx.b.r() as u8) as u64),
            Width::W16 => $ctx.a.w(($ctx.a.r() as u16 $op $ctx.b.r() as u16) as u64),
            Width::W32 => $ctx.a.w(($ctx.a.r() as u32 $op $ctx.b.r() as u32) as u64),
            Width::W64 => $ctx.a.w(($ctx.a.r() $op $ctx.b.r()) as u64),
        }
    };
}

enum InstrResult {
    Break,
    Continue,
    Ok,
}

fn execute<'a>(
    ctx: &mut InterpCtx<'a>,
    air_funcs: &'a [AirFunc],
    libs: &HashMap<&str, libloading::Library>,
    log: bool,
) -> InstrResult {
    let instr = match ctx.next() {
        Some(instr) => instr,
        None => return InstrResult::Break,
    };

    if log {
        ctx.log(instr);
    }

    match instr {
        Air::Ret => return ctx.pop_frame(),
        Air::Call(sig, args) => {
            match sig.linkage {
                Linkage::Local => {
                    let func = air_funcs
                        .iter()
                        .find(|f| f.func.sig.ident == sig.ident)
                        .unwrap_or_else(|| panic!("invalid func"));
                    ctx.start_func(func);

                    if ctx.expect_ident(sig.ident) == "printf" {
                        let (ty, fmt) = args.vars.first().unwrap();
                        assert_eq!(*ty, TyId::STR_LIT);
                        let fmt = OffsetVar::zero(*fmt);
                        let len = ctx.stack.read_var::<u64>(fmt);
                        let addr = ctx.stack.read_var::<u64>(fmt.add(Width::W64));
                        let str = unsafe { str::from_raw_parts(addr as *const u8, len as usize) };

                        enum Entry {
                            Char(char),
                            Arg,
                        }

                        let mut args = args.vars.iter();
                        let _fmt = args.next();
                        let mut buf = String::with_capacity(str.len());
                        for entry in
                            str.chars()
                                .map(|c| if c == '%' { Entry::Arg } else { Entry::Char(c) })
                        {
                            match entry {
                                Entry::Char(c) => buf.push(c),
                                Entry::Arg => {
                                    print!("{}", buf);
                                    buf.clear();
                                    match args.next() {
                                        Some((ty, var)) => {
                                            let var = OffsetVar::zero(*var);
                                            match ctx.tys.ty(*ty) {
                                                Ty::Int(ty) => match ty.kind() {
                                                    IntKind::U8 => {
                                                        print!("{}", ctx.stack.read_var::<u8>(var))
                                                    }
                                                    IntKind::U16 => {
                                                        print!("{}", ctx.stack.read_var::<u16>(var))
                                                    }
                                                    IntKind::U32 => {
                                                        print!("{}", ctx.stack.read_var::<u32>(var))
                                                    }
                                                    IntKind::U64 => {
                                                        print!("{}", ctx.stack.read_var::<u64>(var))
                                                    }
                                                    IntKind::I8 => {
                                                        print!("{}", ctx.stack.read_var::<i8>(var))
                                                    }
                                                    IntKind::I16 => {
                                                        print!("{}", ctx.stack.read_var::<i16>(var))
                                                    }
                                                    IntKind::I32 => {
                                                        print!("{}", ctx.stack.read_var::<i32>(var))
                                                    }
                                                    IntKind::I64 => {
                                                        print!("{}", ctx.stack.read_var::<i64>(var))
                                                    }
                                                },
                                                Ty::Float(float) => match float {
                                                    FloatTy::F32 => {
                                                        print!("{}", ctx.stack.read_var::<f32>(var))
                                                    }
                                                    FloatTy::F64 => {
                                                        print!("{}", ctx.stack.read_var::<f64>(var))
                                                    }
                                                },
                                                Ty::Bool => {
                                                    print!("{}", ctx.stack.read_var::<u8>(var) == 1)
                                                }
                                                Ty::Ref(&Ty::Str) => todo!(),
                                                Ty::Ref(_) => {
                                                    print!("{:#x}", ctx.stack.read_var::<u64>(var))
                                                }
                                                ty => unimplemented!("printf arg: {ty:?}"),
                                            }
                                        }
                                        None => panic!("expected more args in printf"),
                                    }
                                }
                            }
                        }

                        assert_eq!(12.4, f32::from_le_bytes((12.4f32).to_le_bytes()));

                        if args.next().is_some() {
                            panic!("too many args in printf");
                        }

                        print!("{}", buf);
                        buf.clear();
                    }

                    return InstrResult::Continue;
                }
                Linkage::External { link } => {
                    // TODO: build these once

                    let lib = libs.get(link).unwrap();
                    let func: libloading::Symbol<*mut c_void> =
                        unsafe { lib.get(ctx.expect_ident(sig.ident).as_bytes()).unwrap() };

                    let mut params = Vec::with_capacity(sig.params.len());
                    for param in sig.params.iter() {
                        params.push(ctx.tys.ty(param.ty).libffi_type(ctx))
                    }

                    let ty = if sig.ty == ctx.tys.unit() {
                        Type::void()
                    } else {
                        ctx.tys.ty(sig.ty).libffi_type(ctx)
                    };
                    let cif = Cif::new(params.into_iter(), ty);

                    let size = if sig.ty == ctx.tys.unit() {
                        0
                    } else {
                        ctx.tys.ty(sig.ty).size(ctx)
                    };

                    let args = args
                        .vars
                        .iter()
                        .map(|(_, v)| {
                            Arg::new(unsafe { &*ctx.stack.var_ptr_mut(OffsetVar::zero(*v)) })
                        })
                        .collect::<Vec<_>>();

                    unsafe {
                        match size {
                            0 => cif.call::<()>(CodePtr(func.into_raw().as_raw_ptr()), &args),
                            1 => {
                                let result =
                                    cif.call::<u8>(CodePtr(func.into_raw().as_raw_ptr()), &args);

                                match ctx.tys.ty(sig.ty) {
                                    Ty::Bool => {
                                        ctx.a.w(result as u64);
                                    }
                                    Ty::Ref(Ty::Str) => todo!(),
                                    Ty::Ref(_) => ctx.a.w(result as u64),
                                    _ => todo!(),
                                }
                            }
                            2 => {
                                let result =
                                    cif.call::<u16>(CodePtr(func.into_raw().as_raw_ptr()), &args);

                                match ctx.tys.ty(sig.ty) {
                                    Ty::Bool => {
                                        ctx.a.w(result as u64);
                                    }
                                    Ty::Ref(Ty::Str) => todo!(),
                                    Ty::Ref(_) => ctx.a.w(result as u64),
                                    _ => todo!(),
                                }
                            }
                            4 => {
                                let result =
                                    cif.call::<u32>(CodePtr(func.into_raw().as_raw_ptr()), &args);

                                match ctx.tys.ty(sig.ty) {
                                    Ty::Bool => {
                                        ctx.a.w(result as u64);
                                    }
                                    Ty::Ref(Ty::Str) => todo!(),
                                    Ty::Ref(_) => ctx.a.w(result as u64),
                                    _ => todo!(),
                                }
                            }
                            8 => {
                                let result =
                                    cif.call::<u64>(CodePtr(func.into_raw().as_raw_ptr()), &args);

                                match ctx.tys.ty(sig.ty) {
                                    Ty::Bool => {
                                        ctx.a.w(result);
                                    }
                                    Ty::Ref(Ty::Str) => todo!(),
                                    Ty::Ref(_) => ctx.a.w(result),
                                    Ty::Struct(id) => {
                                        let bytes = ctx.tys.struct_layout(id).size;
                                        let addr = ctx.stack.anon_alloc(bytes);
                                        ctx.stack.write_bits(Bits::from_u64(result), addr as usize);
                                        ctx.a.w(addr as u64);
                                    }
                                    _ => todo!(),
                                }
                            }
                            _ => todo!(),
                        }
                    };
                }
            }
        }

        Air::IfElse {
            condition,
            then,
            otherwise,
        } => {
            if ctx.r(*condition) == 1 {
                ctx.start_block(*then);
            } else {
                ctx.start_block(*otherwise);
            }
            return InstrResult::Continue;
        }
        Air::Jmp(block) => {
            ctx.start_block(*block);
            return InstrResult::Continue;
        }

        Air::SAlloc(var, bytes) => {
            ctx.stack.alloc(*var, *bytes);
        }

        Air::ReadSP(var) => {
            ctx.stack
                .push_some_bits(*var, Bits::from_u64(ctx.stack.sp() as u64));
        }
        Air::WriteSP(var) => {
            *ctx.stack.sp_mut() = ctx.stack.read_var::<u64>(*var) as usize;
        }

        Air::Addr(reg, var) => {
            let addr = ctx.stack.var_addr(*var) as u64;
            ctx.w(*reg, addr);
        }
        Air::MemCpy { dst, src, bytes } => {
            let dst = ctx.r(*dst) as usize;
            let src = ctx.r(*src) as usize;
            unsafe { ctx.stack.memcpy(dst, src, *bytes) };
        }

        Air::PushIConst(var, data) => match data {
            ConstData::Bits(bits) => {
                ctx.stack.push_some_bits(*var, *bits);
            }
            ConstData::Ptr(entry) => {
                ctx.stack.push_var::<u64>(*var, entry.addr() as u64);
            }
        },
        Air::PushIReg { dst, width, src } => {
            let bits = Bits::from_width(ctx.r(*src), *width);
            ctx.stack.push_some_bits(*dst, bits);
        }
        Air::PushIVar { dst, width, src } => {
            let bits = ctx.stack.read_some_bits(*src, *width);
            ctx.stack.push_some_bits(*dst, bits);
        }

        Air::Read { dst, addr, width } => {
            let addr = ctx.r(*addr);
            ctx.w(
                *dst,
                ctx.stack
                    .read_some_bits_with_addr(addr as usize, *width)
                    .to_u64(),
            );
        }
        Air::Write { addr, data, width } => {
            let addr = ctx.r(*addr);
            let data_bits = ctx.r(*data);
            ctx.stack
                .write_bits(Bits::from_width(data_bits, *width), addr as usize);
        }

        Air::SwapReg => {
            let tmp = ctx.a.r();
            ctx.a.w(ctx.b.r());
            ctx.b.w(tmp);
        }
        Air::MovIConst(reg, data) => {
            let data = match data {
                ConstData::Bits(bits) => bits.to_u64(),
                ConstData::Ptr(data) => data.addr() as u64,
            };

            ctx.w(*reg, data);
        }
        Air::MovIVar(reg, var, width) => {
            let bits = ctx.stack.read_some_bits(*var, *width).to_u64();
            ctx.w(*reg, bits);
        }

        Air::AddAB(width) => int_op!(ctx, width, +),
        Air::SubAB(width) => int_op!(ctx, width, -),
        Air::MulAB(width) => int_op!(ctx, width, *),
        Air::DivAB(width) => int_op!(ctx, width, /),
        Air::XorAB(width) => int_op!(ctx, width, ^),

        Air::EqAB(width) => int_op!(ctx, width, ==),
        Air::NEqAB(width) => int_op!(ctx, width, !=),

        Air::FAddAB(width) => float_op!(ctx, width, +),
        Air::FSubAB(width) => float_op!(ctx, width, -),
        Air::FMulAB(width) => float_op!(ctx, width, *),
        Air::FDivAB(width) => float_op!(ctx, width, /),

        Air::FEqAB(width) => float_op!(Cmp, ctx, width, ==),
        Air::NFEqAB(width) => float_op!(Cmp, ctx, width, !=),

        Air::FSqrt(ty) => match ty {
            FloatTy::F32 => {
                ctx.a
                    .w(f32::from_bits(ctx.a.r() as u32).sqrt().to_bits() as u64);
            }
            FloatTy::F64 => {
                ctx.a.w(f64::from_bits(ctx.a.r()).sqrt().to_bits());
            }
        },

        Air::XOR(mask) => {
            ctx.a.w(mask.bitxor(ctx.a.r()));
        }

        Air::Fu32 => {
            ctx.a.w(f32::from_bits(ctx.a.r() as u32) as u32 as u64);
        }

        Air::Exit => {
            return InstrResult::Break;
        }
        Air::PrintCStr => unsafe {
            libc::printf(ctx.a.r() as *const c_char);
            libc::printf("\n\0".as_ptr() as *const c_char);
        },
    }

    InstrResult::Ok
}

impl Ty<'_> {
    pub fn libffi_type(&self, ctx: &Ctx) -> Type {
        match self {
            Ty::Int(ty) => match ty.kind() {
                IntKind::I8 => Type::i8(),
                IntKind::U8 => Type::u8(),
                IntKind::I16 => Type::i16(),
                IntKind::U16 => Type::u16(),
                IntKind::I32 => Type::i32(),
                IntKind::U32 => Type::u32(),
                IntKind::I64 => Type::i64(),
                IntKind::U64 => Type::u64(),
            },
            Ty::Float(ty) => match ty {
                FloatTy::F32 => Type::f32(),
                FloatTy::F64 => Type::f64(),
            },
            Ty::Str => Type::pointer(),
            Ty::Bool => Type::u8(),
            Ty::Ref(Ty::Str) => Type::structure([Type::u64(), Type::pointer()]),
            Ty::Ref(_) => Type::pointer(),
            Ty::Struct(id) => Type::structure(
                ctx.tys
                    .strukt(*id)
                    .fields
                    .iter()
                    .map(|f| ctx.tys.ty(f.ty).libffi_type(ctx)),
            ),
            ty => todo!("{ty:?}"),
        }
    }
}
