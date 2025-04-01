use self::ctx::InterpCtx;
use crate::air::{
    Air, AirFunc, AirLinkage, AirSig, Bits, ByteCode, ConstData, IntKind, OffsetVar, Prim,
};
use crate::ir::ty::store::TyStore;
use crate::ir::ty::{FloatTy, Sign, Ty, TyKind, Width};
use core::str;
use libffi::low::CodePtr;
use libffi::middle::{Arg, Cif, Type};
use std::collections::{HashMap, HashSet};
use std::ffi::{c_char, c_void};

mod ctx;
mod stack;

pub struct InterpInstance<'a> {
    bytecode: &'a ByteCode<'a>,
}

impl<'a> InterpInstance<'a> {
    pub fn new(bytecode: &'a ByteCode<'a>) -> Self {
        Self { bytecode }
    }

    pub fn run(&self, log: bool) -> i32 {
        let libs = load_libraries(self.bytecode.extern_sigs.values().copied());
        let main = self
            .bytecode
            .funcs
            .iter()
            .find(|f| f.sig.ident == "main")
            .unwrap();
        entry(main, self.bytecode, &libs, log)
    }
}

fn load_libraries<'a>(
    sigs: impl Iterator<Item = &'a AirSig<'a>>,
) -> HashMap<&'a str, libloading::Library> {
    let mut links = HashSet::<&str>::default();
    for sig in sigs {
        match sig.linkage {
            AirLinkage::External { link } => {
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
    main: &AirFunc,
    bytecode: &ByteCode,
    libs: &HashMap<&str, libloading::Library>,
    log: bool,
) -> i32 {
    let mut ctx = InterpCtx::new(&bytecode.tys, &bytecode.bss);
    ctx.consts(&bytecode.consts);
    loop {
        match execute(&mut ctx, &bytecode.funcs, libs, log) {
            InstrResult::Break => break,
            InstrResult::Continue => continue,
            InstrResult::Ok => {}
        }

        ctx.incr_instr();
    }

    ctx.start_func(main);
    loop {
        match execute(&mut ctx, &bytecode.funcs, libs, log) {
            InstrResult::Break => break,
            InstrResult::Continue => continue,
            InstrResult::Ok => {}
        }

        ctx.incr_instr();
    }
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
    ($ctx:expr, $width:expr, $sign:expr, $op:tt) => {
        match $width {
            Width::W8 => match $sign {
                Sign::U => $ctx
                    .a
                    .w((($ctx.a.r() as u8).$op(($ctx.b.r() as u8))) as u64),
                Sign::I => unsafe {
                    $ctx.a.w(std::mem::transmute(
                        (($ctx.a.r() as i8).$op(($ctx.b.r() as i8))) as i64,
                    ))
                },
            },
            Width::W16 => match $sign {
                Sign::U => $ctx
                    .a
                    .w((($ctx.a.r() as u16).$op(($ctx.b.r() as u16))) as u64),
                Sign::I => unsafe {
                    $ctx.a.w(std::mem::transmute(
                        (($ctx.a.r() as i16).$op(($ctx.b.r() as i16))) as i64,
                    ))
                },
            },
            Width::W32 => match $sign {
                Sign::U => $ctx
                    .a
                    .w((($ctx.a.r() as u32).$op(($ctx.b.r() as u32))) as u64),
                Sign::I => unsafe {
                    $ctx.a.w(std::mem::transmute(
                        (($ctx.a.r() as i32).$op(($ctx.b.r() as i32))) as i64,
                    ))
                },
            },
            Width::W64 => match $sign {
                Sign::U => $ctx
                    .a
                    .w((($ctx.a.r() as u64).$op(($ctx.b.r() as u64))) as u64),
                Sign::I => unsafe {
                    $ctx.a.w(std::mem::transmute(
                        (($ctx.a.r() as i64).$op(($ctx.b.r() as i64))) as i64,
                    ))
                },
            },
        }
    };
}

macro_rules! shift_op {
    ($ctx:expr, $width:expr, $sign:expr, $op:tt) => {
        match $width {
            Width::W8 => match $sign {
                Sign::U => $ctx
                    .a
                    .w((($ctx.a.r() as u8).$op(($ctx.b.r() as u32))) as u64),
                Sign::I => unsafe {
                    $ctx.a.w(std::mem::transmute(
                        (($ctx.a.r() as i8).$op(($ctx.b.r() as u32))) as i64,
                    ))
                },
            },
            Width::W16 => match $sign {
                Sign::U => $ctx
                    .a
                    .w((($ctx.a.r() as u16).$op(($ctx.b.r() as u32))) as u64),
                Sign::I => unsafe {
                    $ctx.a.w(std::mem::transmute(
                        (($ctx.a.r() as i16).$op(($ctx.b.r() as u32))) as i64,
                    ))
                },
            },
            Width::W32 => match $sign {
                Sign::U => $ctx
                    .a
                    .w((($ctx.a.r() as u32).$op(($ctx.b.r() as u32))) as u64),
                Sign::I => unsafe {
                    $ctx.a.w(std::mem::transmute(
                        (($ctx.a.r() as i32).$op(($ctx.b.r() as u32))) as i64,
                    ))
                },
            },
            Width::W64 => match $sign {
                Sign::U => $ctx
                    .a
                    .w((($ctx.a.r() as u64).$op(($ctx.b.r() as u32))) as u64),
                Sign::I => unsafe {
                    $ctx.a.w(std::mem::transmute(
                        (($ctx.a.r() as i64).$op(($ctx.b.r() as u32))) as i64,
                    ))
                },
            },
        }
    };
}

macro_rules! cmp_op {
    ($ctx:expr, $width:expr, $sign:expr, $op:tt) => {
        match $width {
            Width::W8 => match $sign {
                Sign::U => $ctx
                    .a
                    .w((($ctx.a.r() as u8).$op(&($ctx.b.r() as u8))) as u64),
                Sign::I => unsafe {
                    $ctx.a.w(std::mem::transmute(
                        (($ctx.a.r() as i8).$op(&($ctx.b.r() as i8))) as i64,
                    ))
                },
            },
            Width::W16 => match $sign {
                Sign::U => $ctx
                    .a
                    .w((($ctx.a.r() as u16).$op(&($ctx.b.r() as u16))) as u64),
                Sign::I => unsafe {
                    $ctx.a.w(std::mem::transmute(
                        (($ctx.a.r() as i16).$op(&($ctx.b.r() as i16))) as i64,
                    ))
                },
            },
            Width::W32 => match $sign {
                Sign::U => $ctx
                    .a
                    .w((($ctx.a.r() as u32).$op(&($ctx.b.r() as u32))) as u64),
                Sign::I => unsafe {
                    $ctx.a.w(std::mem::transmute(
                        (($ctx.a.r() as i32).$op(&($ctx.b.r() as i32))) as i64,
                    ))
                },
            },
            Width::W64 => match $sign {
                Sign::U => $ctx
                    .a
                    .w((($ctx.a.r() as u64).$op(&($ctx.b.r() as u64))) as u64),
                Sign::I => unsafe {
                    $ctx.a.w(std::mem::transmute(
                        (($ctx.a.r() as i64).$op(&($ctx.b.r() as i64))) as i64,
                    ))
                },
            },
        }
    };
}

macro_rules! bit_op {
    ($ctx:expr, $width:tt, $op:tt) => {
        match $width {
            Width::W8 => {
                $ctx.a.w((($ctx.a.r() as u8) $op ($ctx.b.r() as u8)) as u64)
            }
            Width::W16 => {
                $ctx.a.w((($ctx.a.r() as u16) $op ($ctx.b.r() as u16)) as u64)
            }
            Width::W32 => {
                $ctx.a.w((($ctx.a.r() as u32) $op ($ctx.b.r() as u32)) as u64)
            }
            Width::W64 => {
                $ctx.a.w($ctx.a.r() $op $ctx.b.r())
            }
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
                AirLinkage::Local => {
                    let func = air_funcs
                        .iter()
                        .find(|f| f.sig == *sig)
                        .unwrap_or_else(|| panic!("invalid func"));
                    ctx.start_func(func);

                    if sig.ident == "print" || sig.ident == "println" {
                        let (ty, fmt) = args.vars.first().unwrap();
                        assert_eq!(*ty, Ty::STR_LIT);
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
                        for entry in str
                            .chars()
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
                                            match ty.0 {
                                                TyKind::Int(ty) => match ty.kind() {
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
                                                TyKind::Float(float) => match float {
                                                    FloatTy::F32 => {
                                                        print!("{}", ctx.stack.read_var::<f32>(var))
                                                    }
                                                    FloatTy::F64 => {
                                                        print!("{}", ctx.stack.read_var::<f64>(var))
                                                    }
                                                },
                                                TyKind::Bool => {
                                                    print!("{}", ctx.stack.read_var::<u8>(var) == 1)
                                                }
                                                TyKind::Ref(TyKind::Str) => {
                                                    let len = ctx
                                                        .stack
                                                        .read_some_bits(var, Width::SIZE)
                                                        .to_u64();
                                                    let ptr = ctx
                                                        .stack
                                                        .read_some_bits(
                                                            var.add(Width::SIZE),
                                                            Width::SIZE,
                                                        )
                                                        .to_u64();
                                                    print!("{}", unsafe {
                                                        str::from_raw_parts(
                                                            ptr as *const u8,
                                                            len as usize,
                                                        )
                                                    })
                                                }
                                                TyKind::Ref(_) => {
                                                    print!("{:#x}", ctx.stack.read_var::<u64>(var))
                                                }
                                                ty => unimplemented!("print arg: {ty:?}"),
                                            }
                                        }
                                        None => panic!("expected more args in print"),
                                    }
                                }
                            }
                        }

                        assert_eq!(12.4, f32::from_le_bytes((12.4f32).to_le_bytes()));

                        if args.next().is_some() {
                            panic!("too many args in printf");
                        }

                        print!("{}", buf);
                        if sig.ident == "println" {
                            println!();
                        }
                        buf.clear();
                    }

                    return InstrResult::Continue;
                }
                AirLinkage::External { link } => {
                    // TODO: build these once

                    let lib = libs.get(link).unwrap();
                    let func: libloading::Symbol<*mut c_void> =
                        unsafe { lib.get(sig.ident.as_bytes()).unwrap() };

                    let mut params = Vec::with_capacity(sig.params.len());
                    for param in sig.params.iter() {
                        params.push(param.libffi_type(ctx.tys))
                    }

                    let ty = if sig.ty == Ty::UNIT {
                        Type::void()
                    } else {
                        sig.ty.libffi_type(ctx.tys)
                    };
                    let cif = Cif::new(params.into_iter(), ty);

                    let size = if sig.ty == Ty::UNIT {
                        0
                    } else {
                        sig.ty.size(&ctx.tys)
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
                                match sig.ty.0 {
                                    TyKind::Bool => {
                                        ctx.a.w(result as u64);
                                    }
                                    TyKind::Ref(TyKind::Str) => todo!(),
                                    TyKind::Ref(_) => ctx.a.w(result as u64),
                                    _ => todo!(),
                                }
                            }
                            2 => {
                                let result =
                                    cif.call::<u16>(CodePtr(func.into_raw().as_raw_ptr()), &args);
                                match sig.ty.0 {
                                    TyKind::Bool => {
                                        ctx.a.w(result as u64);
                                    }
                                    TyKind::Ref(TyKind::Str) => todo!(),
                                    TyKind::Ref(_) => ctx.a.w(result as u64),
                                    _ => todo!(),
                                }
                            }
                            4 => {
                                let result =
                                    cif.call::<u32>(CodePtr(func.into_raw().as_raw_ptr()), &args);
                                match sig.ty.0 {
                                    TyKind::Bool => {
                                        ctx.a.w(result as u64);
                                    }
                                    TyKind::Ref(TyKind::Str) => todo!(),
                                    TyKind::Ref(_) => ctx.a.w(result as u64),
                                    _ => todo!(),
                                }
                            }
                            8 => {
                                let result =
                                    cif.call::<u64>(CodePtr(func.into_raw().as_raw_ptr()), &args);

                                match sig.ty.0 {
                                    TyKind::Bool => {
                                        ctx.a.w(result);
                                    }
                                    TyKind::Ref(TyKind::Str) => todo!(),
                                    TyKind::Ref(_) => ctx.a.w(result),
                                    TyKind::Struct(id) => {
                                        let bytes = ctx.tys.struct_layout(*id).size;
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
        Air::Deref { dst, addr } => {
            let addr = ctx.r(*addr);
            ctx.stack.point(dst.var, dst.offset + addr as usize);
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

        Air::MulAB(width, sign) => int_op!(ctx, width, sign, wrapping_mul),
        Air::DivAB(width, sign) => int_op!(ctx, width, sign, wrapping_div),
        Air::RemAB(width, sign) => int_op!(ctx, width, sign, wrapping_rem),

        Air::AddAB(width, sign) => int_op!(ctx, width, sign, wrapping_add),
        Air::SubAB(width, sign) => int_op!(ctx, width, sign, wrapping_sub),

        Air::ShlAB(width, sign) => shift_op!(ctx, width, sign, wrapping_shl),
        Air::ShrAB(width, sign) => shift_op!(ctx, width, sign, wrapping_shr),

        Air::BandAB(width) => bit_op!(ctx, width, &),
        Air::XorAB(width) => bit_op!(ctx, width, ^),
        Air::BorAB(width) => bit_op!(ctx, width, |),

        Air::EqAB(width, sign) => cmp_op!(ctx, width, sign, eq),
        Air::NEqAB(width, sign) => cmp_op!(ctx, width, sign, ne),
        Air::LtAB(width, sign) => cmp_op!(ctx, width, sign, lt),
        Air::GtAB(width, sign) => cmp_op!(ctx, width, sign, gt),
        Air::LeAB(width, sign) => cmp_op!(ctx, width, sign, le),
        Air::GeAB(width, sign) => cmp_op!(ctx, width, sign, ge),

        Air::FMulAB(width) => float_op!(ctx, width, *),
        Air::FDivAB(width) => float_op!(ctx, width, /),
        Air::FRemAB(width) => float_op!(ctx, width, %),

        Air::FAddAB(width) => float_op!(ctx, width, +),
        Air::FSubAB(width) => float_op!(ctx, width, -),

        Air::FEqAB(width) => float_op!(Cmp, ctx, width, ==),
        Air::NFEqAB(width) => float_op!(Cmp, ctx, width, !=),
        Air::FLtAB(width) => float_op!(Cmp, ctx, width, <),
        Air::FGtAB(width) => float_op!(Cmp, ctx, width, >),
        Air::FLeAB(width) => float_op!(Cmp, ctx, width, <=),
        Air::FGeAB(width) => float_op!(Cmp, ctx, width, >=),

        Air::FSqrt(ty) => match ty {
            FloatTy::F32 => {
                ctx.a
                    .w(f32::from_bits(ctx.a.r() as u32).sqrt().to_bits() as u64);
            }
            FloatTy::F64 => {
                ctx.a.w(f64::from_bits(ctx.a.r()).sqrt().to_bits());
            }
        },

        Air::CastA {
            from: (from, from_width),
            to: (to, to_width),
        } => {
            let bytes = ctx.a.r().to_le_bytes();

            match from {
                Prim::Bool => match to {
                    Prim::UInt | Prim::Bool => {}
                    Prim::Int => {
                        let b = ctx.a.r();
                        ctx.a.w(if b == 1 { 1u64 as u64 } else { 0u64 as u64 });
                    }
                    Prim::Float => unreachable!(),
                },
                Prim::Float => match to {
                    Prim::Float => {}
                    Prim::UInt => match from_width {
                        Width::W32 => {
                            ctx.a.w(f32::from_le_bytes(
                                bytes[..std::mem::size_of::<f32>()].try_into().unwrap(),
                            ) as u32 as u64);
                        }
                        Width::W64 => {
                            ctx.a.w(f64::from_le_bytes(
                                bytes[..std::mem::size_of::<f64>()].try_into().unwrap(),
                            ) as u64);
                        }
                        _ => unreachable!(),
                    },
                    Prim::Int => match from_width {
                        Width::W32 => {
                            unsafe {
                                ctx.a.w(std::mem::transmute(f32::from_le_bytes(
                                    bytes[..std::mem::size_of::<f32>()].try_into().unwrap(),
                                ) as i32
                                    as i64))
                            };
                        }
                        Width::W64 => {
                            unsafe {
                                ctx.a.w(std::mem::transmute(f64::from_le_bytes(
                                    bytes[..std::mem::size_of::<f64>()].try_into().unwrap(),
                                )
                                    as i64))
                            };
                        }
                        _ => unreachable!(),
                    },
                    Prim::Bool => unreachable!(),
                },
                Prim::UInt => match to {
                    Prim::UInt => {}
                    Prim::Bool => unreachable!(),
                    Prim::Int => match from_width {
                        Width::W8 => {
                            unsafe {
                                ctx.a.w(std::mem::transmute(u8::from_le_bytes(
                                    bytes[..std::mem::size_of::<u8>()].try_into().unwrap(),
                                ) as i8
                                    as i64))
                            };
                        }
                        Width::W16 => {
                            unsafe {
                                ctx.a.w(std::mem::transmute(u16::from_le_bytes(
                                    bytes[..std::mem::size_of::<u16>()].try_into().unwrap(),
                                ) as i16
                                    as i64))
                            };
                        }
                        Width::W32 => {
                            unsafe {
                                ctx.a.w(std::mem::transmute(u32::from_le_bytes(
                                    bytes[..std::mem::size_of::<u32>()].try_into().unwrap(),
                                ) as i32
                                    as i64))
                            };
                        }
                        Width::W64 => {
                            unsafe {
                                ctx.a.w(std::mem::transmute(u64::from_le_bytes(
                                    bytes[..std::mem::size_of::<u64>()].try_into().unwrap(),
                                )
                                    as i64))
                            };
                        }
                    },
                    Prim::Float => match from_width {
                        Width::W32 => {
                            let f = u32::from_le_bytes(
                                bytes[..std::mem::size_of::<f32>()].try_into().unwrap(),
                            ) as f32;

                            match to_width {
                                Width::W32 => {
                                    ctx.a.w(f.to_bits() as u64);
                                }
                                Width::W64 => {
                                    ctx.a.w((f as f64).to_bits() as u64);
                                }
                                _ => unreachable!(),
                            }
                        }
                        Width::W64 => {
                            let f = u64::from_le_bytes(
                                bytes[..std::mem::size_of::<f64>()].try_into().unwrap(),
                            ) as f64;

                            match to_width {
                                Width::W32 => {
                                    ctx.a.w((f as f32).to_bits() as u64);
                                }
                                Width::W64 => {
                                    ctx.a.w(f.to_bits() as u64);
                                }
                                _ => unreachable!(),
                            }
                        }
                        _ => unreachable!(),
                    },
                },
                Prim::Int => match to {
                    Prim::Int => {}
                    Prim::Bool => unreachable!(),
                    Prim::UInt => match from_width {
                        Width::W8 => {
                            ctx.a.w(i8::from_le_bytes(
                                bytes[..std::mem::size_of::<u8>()].try_into().unwrap(),
                            ) as u8 as u64);
                        }
                        Width::W16 => {
                            ctx.a.w(i16::from_le_bytes(
                                bytes[..std::mem::size_of::<u16>()].try_into().unwrap(),
                            ) as u16 as u64);
                        }
                        Width::W32 => {
                            ctx.a.w(i32::from_le_bytes(
                                bytes[..std::mem::size_of::<u32>()].try_into().unwrap(),
                            ) as u32 as u64);
                        }
                        Width::W64 => {
                            ctx.a.w(i64::from_le_bytes(
                                bytes[..std::mem::size_of::<u64>()].try_into().unwrap(),
                            ) as u64);
                        }
                    },
                    Prim::Float => match from_width {
                        Width::W32 => {
                            let f = i32::from_le_bytes(
                                bytes[..std::mem::size_of::<f32>()].try_into().unwrap(),
                            ) as f32;

                            match to_width {
                                Width::W32 => {
                                    ctx.a.w(f.to_bits() as u64);
                                }
                                Width::W64 => {
                                    ctx.a.w((f as f64).to_bits() as u64);
                                }
                                _ => unreachable!(),
                            }
                        }
                        Width::W64 => {
                            let f = i64::from_le_bytes(
                                bytes[..std::mem::size_of::<f64>()].try_into().unwrap(),
                            ) as f64;

                            match to_width {
                                Width::W32 => {
                                    ctx.a.w((f as f32).to_bits() as u64);
                                }
                                Width::W64 => {
                                    ctx.a.w(f.to_bits() as u64);
                                }
                                _ => unreachable!(),
                            }
                        }
                        _ => unreachable!(),
                    },
                },
            }
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

impl TyKind {
    pub fn libffi_type(&self, tys: &TyStore) -> Type {
        match self {
            TyKind::Int(ty) => match ty.kind() {
                IntKind::I8 => Type::i8(),
                IntKind::U8 => Type::u8(),
                IntKind::I16 => Type::i16(),
                IntKind::U16 => Type::u16(),
                IntKind::I32 => Type::i32(),
                IntKind::U32 => Type::u32(),
                IntKind::I64 => Type::i64(),
                IntKind::U64 => Type::u64(),
            },
            TyKind::Float(ty) => match ty {
                FloatTy::F32 => Type::f32(),
                FloatTy::F64 => Type::f64(),
            },
            TyKind::Str => Type::pointer(),
            TyKind::Bool => Type::u8(),
            TyKind::Ref(TyKind::Str) => Type::structure([Type::u64(), Type::pointer()]),
            TyKind::Ref(_) => Type::pointer(),
            TyKind::Struct(id) => {
                Type::structure(tys.strukt(*id).fields.iter().map(|f| f.ty.libffi_type(tys)))
            }
            ty => todo!("{ty:?}"),
        }
    }
}
