use crate::air::{Air, AirFunc, Bits, BlockId, ConstData, IntKind, OffsetVar, Reg};
use crate::ir::ctx::Ctx;
use crate::ir::sig::{Linkage, Sig};
use crate::ir::ty::store::TyId;
use crate::ir::ty::{FloatTy, Ty, Width};
use anstream::println;
use core::str;
use libffi::low::CodePtr;
use libffi::middle::{Arg, Cif, Type};
use stack::*;
use std::collections::{HashMap, HashSet};
use std::ffi::{c_char, c_void};
use std::ops::BitXor;

mod stack;

#[derive(Debug)]
pub enum InterpErr {
    NoEntry,
    InvalidFunc,
    NoReturn,
    DryStream,
}

pub fn run<'a>(
    ctx: &'a Ctx<'a>,
    air_funcs: &[AirFunc],
    consts: &[Air<'a>],
    log: bool,
) -> Result<i32, InterpErr> {
    let libs = load_libraries(ctx.sigs.values().copied());

    let main = air_funcs
        .iter()
        .find(|f| ctx.expect_ident(f.func.sig.ident) == "main")
        .ok_or_else(|| InterpErr::NoEntry)?;
    let mut stack = Stack::default();
    eval_consts(consts, &mut stack)?;
    let result = entry(ctx, main, air_funcs, &mut stack, &libs, log);
    println!("stack_len: {:?}", stack.len());
    result
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

#[derive(Default)]
pub struct CallStack<'a> {
    stack: Vec<(LR<'a>, usize, BlockId)>,
}

impl<'a> CallStack<'a> {
    pub fn push_lr(&mut self, lr: LR<'a>, block_id: BlockId) {
        self.stack.push((lr, 0, block_id));
    }

    pub fn grow_stack(&mut self, bytes: usize) {
        if let Some((_, stack, _)) = self.stack.last_mut() {
            *stack += bytes;
        }
    }
}

pub struct LR<'a> {
    func: &'a AirFunc<'a>,
    instr: usize,
}

#[derive(Default)]
pub struct Rega(u64);

impl Rega {
    pub fn w(&mut self, val: u64) {
        self.0 = val;
    }

    pub fn r(&self) -> u64 {
        self.0
    }
}

fn entry(
    ctx: &Ctx,
    main: &AirFunc,
    air_funcs: &[AirFunc],
    stack: &mut Stack,
    libs: &HashMap<&str, libloading::Library>,
    log: bool,
) -> Result<i32, InterpErr> {
    let mut current_func = main;
    let mut instrs = main.start().iter();
    let mut current_block = Some(main.start_block());
    let mut call_stack = CallStack::default();
    let mut rega = Rega::default();
    let mut regb = Rega::default();

    let mut total_instr_num = 0;
    let mut instr_num = 0;

    loop {
        let Some(instr) = instrs.next() else {
            //if let Some(block) = current_block {
            //    if let Some(next_block) = current_func.next_block(block) {
            //        //println!("{:#?}", next_block);
            //        //instrs = next_block.iter();
            //        //continue;
            //    }
            //}

            return Err(InterpErr::DryStream);
        };

        if log {
            log_instr(
                ctx,
                current_func,
                stack,
                &call_stack,
                &rega,
                &regb,
                instr,
                instr_num,
                total_instr_num,
                current_block.unwrap(),
            );
        }

        match instr {
            Air::Ret => match call_stack.stack.last() {
                Some((LR { func, instr }, stack_frame_size, block)) => {
                    current_func = func;
                    instrs = func.block(*block).iter();
                    for _ in 0..*instr {
                        instrs.next();
                    }
                    instr_num = *instr;
                    *stack.sp_mut() -= stack_frame_size;
                    current_block = Some(*block);
                    call_stack.stack.pop();
                    continue;
                }
                None => {
                    if ctx.expect_ident(current_func.func.sig.ident) == "main" {
                        //rega.set(0);
                        break;
                    } else {
                        return Err(InterpErr::NoReturn);
                    }
                }
            },
            Air::Call(sig, args) => {
                match sig.linkage {
                    Linkage::Local => {
                        call_stack.push_lr(
                            LR {
                                func: current_func,
                                instr: instr_num + 1,
                            },
                            current_block.expect("no current block"),
                        );
                        instr_num = 0;

                        // TODO: hashmap?
                        let func = air_funcs
                            .iter()
                            .find(|f| f.func.sig.ident == sig.ident)
                            .ok_or_else(|| InterpErr::InvalidFunc)?;
                        current_func = func;
                        instrs = func.start().iter();
                        current_block = Some(func.start_block());

                        //&*stack.var_ptr_mut(args.vars.first().unwrap().1),
                        if ctx.expect_ident(sig.ident) == "printf" {
                            let (ty, fmt) = args.vars.first().unwrap();
                            assert_eq!(*ty, TyId::STR_LIT);
                            let fmt = OffsetVar::zero(*fmt);
                            let len = stack.read_var::<u64>(fmt);
                            let addr = stack.read_var::<u64>(fmt.add(Width::W64));
                            let str =
                                unsafe { str::from_raw_parts(addr as *const u8, len as usize) };

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
                                                            print!("{}", stack.read_var::<u8>(var))
                                                        }
                                                        IntKind::U16 => {
                                                            print!("{}", stack.read_var::<u16>(var))
                                                        }
                                                        IntKind::U32 => {
                                                            print!("{}", stack.read_var::<u32>(var))
                                                        }
                                                        IntKind::U64 => {
                                                            print!("{}", stack.read_var::<u64>(var))
                                                        }
                                                        IntKind::I8 => {
                                                            print!("{}", stack.read_var::<i8>(var))
                                                        }
                                                        IntKind::I16 => {
                                                            print!("{}", stack.read_var::<i16>(var))
                                                        }
                                                        IntKind::I32 => {
                                                            print!("{}", stack.read_var::<i32>(var))
                                                        }
                                                        IntKind::I64 => {
                                                            print!("{}", stack.read_var::<i64>(var))
                                                        }
                                                    },
                                                    Ty::Float(float) => match float {
                                                        FloatTy::F32 => {
                                                            print!("{}", stack.read_var::<f32>(var))
                                                        }
                                                        FloatTy::F64 => {
                                                            print!("{}", stack.read_var::<f64>(var))
                                                        }
                                                    },
                                                    Ty::Bool => {
                                                        print!("{}", stack.read_var::<u8>(var) == 1)
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

                        continue;
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
                                Arg::new(unsafe { &*stack.var_ptr_mut(OffsetVar::zero(*v)) })
                            })
                            .collect::<Vec<_>>();

                        unsafe {
                            match size {
                                0 => cif.call::<()>(CodePtr(func.into_raw().as_raw_ptr()), &args),
                                1 => {
                                    let result = cif
                                        .call::<u8>(CodePtr(func.into_raw().as_raw_ptr()), &args);

                                    match ctx.tys.ty(sig.ty) {
                                        Ty::Bool => {
                                            rega.w(result as u64);
                                        }
                                        Ty::Ref(Ty::Str) => todo!(),
                                        Ty::Ref(_) => rega.w(result as u64),
                                        _ => todo!(),
                                    }
                                }
                                2 => {
                                    let result = cif
                                        .call::<u16>(CodePtr(func.into_raw().as_raw_ptr()), &args);

                                    match ctx.tys.ty(sig.ty) {
                                        Ty::Bool => {
                                            rega.w(result as u64);
                                        }
                                        Ty::Ref(Ty::Str) => todo!(),
                                        Ty::Ref(_) => rega.w(result as u64),
                                        _ => todo!(),
                                    }
                                }
                                4 => {
                                    let result = cif
                                        .call::<u32>(CodePtr(func.into_raw().as_raw_ptr()), &args);

                                    match ctx.tys.ty(sig.ty) {
                                        Ty::Bool => {
                                            rega.w(result as u64);
                                        }
                                        Ty::Ref(Ty::Str) => todo!(),
                                        Ty::Ref(_) => rega.w(result as u64),
                                        _ => todo!(),
                                    }
                                }
                                8 => {
                                    let result = cif
                                        .call::<u64>(CodePtr(func.into_raw().as_raw_ptr()), &args);

                                    match ctx.tys.ty(sig.ty) {
                                        Ty::Bool => {
                                            rega.w(result);
                                        }
                                        Ty::Ref(Ty::Str) => todo!(),
                                        Ty::Ref(_) => rega.w(result),
                                        Ty::Struct(id) => {
                                            let bytes = ctx.tys.struct_layout(id).size;
                                            let addr = stack.anon_alloc(bytes);
                                            stack.memcpy(
                                                addr as usize,
                                                &result as *const u64 as usize,
                                                bytes,
                                            );
                                            rega.w(addr as u64);
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
                if match condition {
                    Reg::A => rega.r() == 1,
                    Reg::B => regb.r() == 1,
                } {
                    current_block = Some(*then);
                    instrs = current_func.block(*then).iter();
                    instr_num = 0;
                } else {
                    current_block = Some(*otherwise);
                    instrs = current_func.block(*otherwise).iter();
                    instr_num = 0;
                }
                continue;
            }
            Air::Jmp(block) => {
                current_block = Some(*block);
                instrs = current_func.block(*block).iter();
                instr_num = 0;
                continue;
            }

            Air::SAlloc(var, bytes) => {
                if !stack.allocated(*var) {
                    stack.alloc(*var, *bytes);
                    call_stack.grow_stack(*bytes);
                }
            }

            Air::Addr(reg, var) => {
                let addr = unsafe { stack.var_ptr_mut(*var).addr() as u64 };
                match reg {
                    Reg::A => rega.w(addr),
                    Reg::B => regb.w(addr),
                }
            }
            Air::MemCpy { dst, src, bytes } => {
                let dst = match dst {
                    Reg::A => rega.r(),
                    Reg::B => regb.r(),
                } as usize;
                let src = match src {
                    Reg::A => rega.r(),
                    Reg::B => regb.r(),
                } as usize;
                unsafe { stack.memcpy(dst, src, *bytes) };
            }

            Air::PushIConst(var, data) => match data {
                ConstData::Bits(bits) => {
                    stack.push_some_bits(*var, *bits);
                }
                ConstData::Ptr(entry) => {
                    stack.push_var::<u64>(*var, entry.addr() as u64);
                }
            },
            Air::PushIReg { dst, width, src } => {
                let reg = match src {
                    Reg::A => &mut rega,
                    Reg::B => &mut regb,
                };
                let bits = Bits::from_width(reg.r(), *width);
                stack.push_some_bits(*dst, bits);
            }
            Air::PushIVar { dst, width, src } => {
                let bits = stack.read_some_bits(*src, *width);
                stack.push_some_bits(*dst, bits);
            }

            Air::SwapReg => {
                let tmp = rega.r();
                rega.w(regb.r());
                regb.w(tmp);
            }
            Air::MovIConst(reg, data) => {
                let data = match data {
                    ConstData::Bits(bits) => bits.to_u64(),
                    ConstData::Ptr(data) => data.addr() as u64,
                };

                match reg {
                    Reg::A => rega.w(data),
                    Reg::B => regb.w(data),
                }
            }
            Air::MovIVar(reg, var, width) => {
                let int = stack.read_some_bits(*var, *width).to_u64();
                match reg {
                    Reg::A => {
                        rega.w(int);
                    }
                    Reg::B => {
                        regb.w(int);
                    }
                }
            }

            Air::AddAB(width) => match width {
                Width::W8 => rega.w((rega.r() as u8 + regb.r() as u8) as u64),
                Width::W16 => rega.w((rega.r() as u16 + regb.r() as u16) as u64),
                Width::W32 => rega.w((rega.r() as u32 + regb.r() as u32) as u64),
                Width::W64 => rega.w(rega.r() + regb.r()),
            },
            Air::SubAB(width) => match width {
                Width::W8 => rega.w((rega.r() as u8 - regb.r() as u8) as u64),
                Width::W16 => rega.w((rega.r() as u16 - regb.r() as u16) as u64),
                Width::W32 => rega.w((rega.r() as u32 - regb.r() as u32) as u64),
                Width::W64 => rega.w(rega.r() - regb.r()),
            },
            Air::MulAB(width) => match width {
                Width::W8 => rega.w((rega.r() as u8 * regb.r() as u8) as u64),
                Width::W16 => rega.w((rega.r() as u16 * regb.r() as u16) as u64),
                Width::W32 => rega.w((rega.r() as u32 * regb.r() as u32) as u64),
                Width::W64 => rega.w(rega.r() * regb.r()),
            },
            Air::EqAB(width) => match width {
                Width::W8 => rega.w((rega.r() as u8 == regb.r() as u8) as u64),
                Width::W16 => rega.w((rega.r() as u16 == regb.r() as u16) as u64),
                Width::W32 => rega.w((rega.r() as u32 == regb.r() as u32) as u64),
                Width::W64 => rega.w((rega.r() == regb.r()) as u64),
            },

            Air::FAddAB(width) => match width {
                Width::W32 => {
                    rega.w(
                        (f32::from_bits(rega.r() as u32) + f32::from_bits(regb.r() as u32))
                            .to_bits() as u64,
                    );
                }
                Width::W64 => {
                    rega.w((f64::from_bits(rega.r()) + f64::from_bits(regb.r())).to_bits());
                }
                _ => unreachable!(),
            },
            Air::FSubAB(width) => match width {
                Width::W32 => {
                    rega.w(
                        (f32::from_bits(rega.r() as u32) - f32::from_bits(regb.r() as u32))
                            .to_bits() as u64,
                    );
                }
                Width::W64 => {
                    rega.w((f64::from_bits(rega.r()) - f64::from_bits(regb.r())).to_bits());
                }
                _ => unreachable!(),
            },
            Air::FMulAB(width) => match width {
                Width::W32 => {
                    rega.w(
                        (f32::from_bits(rega.r() as u32) * f32::from_bits(regb.r() as u32))
                            .to_bits() as u64,
                    );
                }
                Width::W64 => {
                    rega.w((f64::from_bits(rega.r()) * f64::from_bits(regb.r())).to_bits());
                }
                _ => unreachable!(),
            },
            Air::FEqAB(width) => match width {
                Width::W32 => {
                    rega.w(
                        (f32::from_bits(rega.r() as u32) == f32::from_bits(regb.r() as u32)) as u64,
                    );
                }
                Width::W64 => {
                    rega.w((f64::from_bits(rega.r()) == f64::from_bits(regb.r())) as u64);
                }
                _ => unreachable!(),
            },

            Air::XOR(mask) => {
                rega.w(mask.bitxor(rega.r()));
            }

            Air::Exit => {
                break;
            }
            Air::PrintCStr => unsafe {
                libc::printf(rega.r() as *const c_char);
                libc::printf("\n\0".as_ptr() as *const c_char);
            },
        }

        instr_num += 1;
        total_instr_num += 1;
    }

    Ok(rega.r() as i32)
}

fn eval_consts(consts: &[Air], stack: &mut Stack) -> Result<(), InterpErr> {
    let mut instrs = consts.iter();
    let mut rega = Rega::default();
    let mut regb = Rega::default();

    loop {
        let Some(instr) = instrs.next() else {
            return Ok(());
        };

        match instr {
            Air::SAlloc(var, bytes) => {
                if !stack.allocated(*var) {
                    stack.alloc(*var, *bytes);
                }
            }

            Air::Addr(reg, var) => {
                let addr = unsafe { stack.var_ptr_mut(*var).addr() as u64 };
                match reg {
                    Reg::A => rega.w(addr),
                    Reg::B => regb.w(addr),
                }
            }

            Air::MemCpy { dst, src, bytes } => {
                let dst = match dst {
                    Reg::A => rega.r(),
                    Reg::B => regb.r(),
                } as usize;
                let src = match src {
                    Reg::A => rega.r(),
                    Reg::B => regb.r(),
                } as usize;
                unsafe { stack.memcpy(dst, src, *bytes) };
            }

            Air::PushIConst(var, data) => match data {
                ConstData::Bits(bits) => {
                    stack.push_some_bits(*var, *bits);
                }
                ConstData::Ptr(entry) => {
                    stack.push_var::<u64>(*var, entry.addr() as u64);
                }
            },
            Air::PushIReg { dst, width, src } => {
                let reg = match src {
                    Reg::A => &mut rega,
                    Reg::B => &mut regb,
                };
                let bits = Bits::from_width(reg.r(), *width);
                stack.push_some_bits(*dst, bits);
            }
            Air::PushIVar { dst, width, src } => {
                let bits = stack.read_some_bits(*src, *width);
                stack.push_some_bits(*dst, bits);
            }

            Air::SwapReg => {
                let tmp = rega.r();
                rega.w(regb.r());
                regb.w(tmp);
            }
            Air::MovIConst(reg, data) => {
                let data = match data {
                    ConstData::Bits(bits) => bits.to_u64(),
                    ConstData::Ptr(data) => data.addr() as u64,
                };

                match reg {
                    Reg::A => rega.w(data),
                    Reg::B => regb.w(data),
                }
            }
            Air::MovIVar(reg, var, width) => {
                let int = stack.read_some_bits(*var, *width).to_u64();
                match reg {
                    Reg::A => {
                        rega.w(int);
                    }
                    Reg::B => {
                        regb.w(int);
                    }
                }
            }

            Air::AddAB(width) => match width {
                Width::W8 => rega.w((rega.r() as u8 + regb.r() as u8) as u64),
                Width::W16 => rega.w((rega.r() as u16 + regb.r() as u16) as u64),
                Width::W32 => rega.w((rega.r() as u32 + regb.r() as u32) as u64),
                Width::W64 => rega.w(rega.r() + regb.r()),
            },
            Air::SubAB(width) => match width {
                Width::W8 => rega.w((rega.r() as u8 - regb.r() as u8) as u64),
                Width::W16 => rega.w((rega.r() as u16 - regb.r() as u16) as u64),
                Width::W32 => rega.w((rega.r() as u32 - regb.r() as u32) as u64),
                Width::W64 => rega.w(rega.r() - regb.r()),
            },
            Air::MulAB(width) => match width {
                Width::W8 => rega.w((rega.r() as u8 + regb.r() as u8) as u64),
                Width::W16 => rega.w((rega.r() as u16 + regb.r() as u16) as u64),
                Width::W32 => rega.w((rega.r() as u32 + regb.r() as u32) as u64),
                Width::W64 => rega.w(rega.r() + regb.r()),
            },
            Air::EqAB(_) => {
                rega.w((rega.r() == regb.r()) as u64);
            }

            instr => panic!("cannot {:?} functions in const eval", instr),
        }
    }
}

fn log_instr(
    ctx: &Ctx,
    current_func: &AirFunc,
    stack: &mut Stack,
    call_stack: &CallStack,
    rega: &Rega,
    regb: &Rega,
    instr: &Air,
    instr_num: usize,
    total_instr_num: usize,
    current_block: BlockId,
) {
    println!(
        "[\u{1b}[32m{}\u{1b}[39m:{:?}:{total_instr_num}]\u{1b}[1m\u{1b}[35m {instr:?}\u{1b}[39m\u{1b}[22m",
        ctx.expect_ident(current_func.func.sig.ident),
        current_block,
    );

    match instr {
        Air::PushIConst(_, _) | Air::Exit | Air::Jmp(_) | Air::SwapReg | Air::MovIConst(_, _) => {}
        Air::XOR(mask) => {
            println!(
                " | Reg(A) <- {:#x} <- {:#x} XOR {:#x}",
                mask.bitxor(rega.r()),
                rega.r(),
                mask
            );
        }
        Air::PushIVar { dst, width, src } => {
            println!(" | {dst:?} <- {:?}", stack.read_some_bits(*src, *width));
        }
        Air::PushIReg { dst, width, src } => {
            println!(
                " | {dst:?} <- {:#x} @ {width:?}",
                match src {
                    Reg::A => rega.r(),
                    Reg::B => regb.r(),
                },
            );
        }
        Air::PrintCStr => {
            println!(" | print_c_str @ `{}`", rega.r());
        }
        Air::IfElse {
            condition,
            then,
            otherwise,
        } => {
            println!(
                " | if ({:#x}), then {:?}, otherwise {:?}",
                match condition {
                    Reg::A => rega.r(),
                    Reg::B => regb.r(),
                },
                then,
                otherwise
            );
        }
        Air::SAlloc(var, bytes) => {
            if !stack.allocated(*var) {
                println!(" | Addr({:?}) <- {:#x}", var, stack.sp());
                println!(" | SP({:?}) += {}", stack.sp(), bytes);
            } else {
                println!(" | Already allocated, skipping");
            }
        }
        Air::MemCpy { dst, src, bytes } => {
            println!(
                " | memcpy {:#x} <- {:#x} @ {bytes} bytes",
                match dst {
                    Reg::A => rega.r(),
                    Reg::B => regb.r(),
                },
                match src {
                    Reg::A => rega.r(),
                    Reg::B => regb.r(),
                }
            );
        }
        Air::Addr(_, var) => {
            let addr = unsafe { stack.var_ptr_mut(*var).addr() };
            println!(" | Addr({:?}) = {:#x}", var.var, addr);
            println!(" | Offset({})", var.offset);
        }
        Air::MovIVar(reg, var, width) => {
            let int = stack.read_some_bits(*var, *width);
            println!(" | {reg:?} <- {int:?}");
        }
        Air::Ret => {
            match call_stack.stack.last() {
                Some((LR { func, instr }, _, _)) => {
                    println!(
                        " | proc {} -> proc {} @ instr #{}",
                        ctx.expect_ident(current_func.func.sig.ident),
                        ctx.expect_ident(func.func.sig.ident),
                        instr
                    );
                }
                None => {
                    println!(
                        " | proc {} -> ???",
                        ctx.expect_ident(current_func.func.sig.ident),
                    );
                }
            }
            println!(" | A = {:#x}", rega.r());
        }
        Air::Call(sig, args) => {
            println!(
                " | call proc [{}({:?})]",
                ctx.expect_ident(sig.ident),
                &args
                    .vars
                    .iter()
                    .map(|(ty, var)| (ctx.ty_str(*ty), var))
                    .collect::<Vec<_>>(),
            );
        }
        Air::AddAB(_) => {
            let result = rega.r() + regb.r();
            println!(" | A <- {} <- A({}) + B({})", result, rega.r(), regb.r());
        }
        Air::SubAB(_) => {
            let result = rega.r() - regb.r();
            println!(" | A <- {} <- A({}) - B({})", result, rega.r(), regb.r());
        }
        Air::MulAB(_) => {
            let result = rega.r() * regb.r();
            println!(" | A <- {} <- A({}) * B({})", result, rega.r(), regb.r());
        }
        Air::EqAB(_) => {
            let result = rega.r() == regb.r();
            println!(" | A <- {} <- A({}) == B({})", result, rega.r(), regb.r());
        }

        Air::FAddAB(_) => {
            let lhs = rega.r();
            let rhs = regb.r();
            let result = f64::from_bits(lhs) + f64::from_bits(rhs);
            println!(" | A <- {} <- A({}) + B({})", result, lhs, rhs);
        }
        Air::FSubAB(_) => {
            let lhs = rega.r();
            let rhs = regb.r();
            let result = f64::from_bits(lhs) - f64::from_bits(rhs);
            println!(" | A <- {} <- A({}) - B({})", result, lhs, rhs);
        }
        Air::FMulAB(_) => {
            let lhs = rega.r();
            let rhs = regb.r();
            let result = f64::from_bits(lhs) * f64::from_bits(rhs);
            println!(" | A <- {} <- A({}) * B({})", result, lhs, rhs);
        }
        Air::FEqAB(_) => {
            let lhs = rega.r();
            let rhs = regb.r();
            let result = f64::from_bits(lhs) == f64::from_bits(rhs);
            println!(" | A <- {} <- A({}) == B({})", result, lhs, rhs);
        }
    }
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
