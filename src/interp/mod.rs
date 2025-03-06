use crate::air::{Air, AirFunc, BlockId, ConstData, IntKind, Reg};
use crate::ir::ctx::Ctx;
use crate::ir::sig::{Linkage, Sig};
use crate::ir::ty::Ty;
use anstream::println;
use libffi::low::CodePtr;
use libffi::middle::{Arg, Cif, Type};
use stack::*;
use std::collections::{HashMap, HashSet};
use std::ffi::{c_char, c_void};

mod stack;

#[derive(Debug)]
pub enum InterpErr {
    NoEntry,
    InvalidFunc,
    NoReturn,
    DryStream,
}

pub fn run<'a>(ctx: &'a Ctx<'a>, air_funcs: &[AirFunc], log: bool) -> Result<i32, InterpErr> {
    let libs = load_libraries(ctx.sigs.values().copied());

    let main = air_funcs
        .iter()
        .find(|f| ctx.expect_ident(f.func.sig.ident) == "main")
        .ok_or_else(|| InterpErr::NoEntry)?;
    let mut stack = Stack::default();
    let result = entry(ctx, main, air_funcs, &mut stack, libs, log);
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
pub struct Rega(i64);

impl Rega {
    pub fn w(&mut self, val: i64) {
        self.0 = val;
    }

    pub fn r(&self) -> i64 {
        self.0
    }
}

fn entry(
    ctx: &Ctx,
    main: &AirFunc,
    air_funcs: &[AirFunc],
    stack: &mut Stack,
    libs: HashMap<&str, libloading::Library>,
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
            if let Some(block) = current_block {
                if let Some(next_block) = current_func.next_block(block) {
                    instrs = next_block.iter();
                    continue;
                }
            }

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

                        unsafe {
                            match size {
                                0 => cif.call::<()>(
                                    CodePtr(func.into_raw().as_raw_ptr()),
                                    &args
                                        .vars
                                        .iter()
                                        .map(|v| Arg::new(&mut *stack.var_ptr_mut(*v)))
                                        .collect::<Vec<_>>(),
                                ),
                                1 => {
                                    let result = cif.call::<u8>(
                                        CodePtr(func.into_raw().as_raw_ptr()),
                                        &args
                                            .vars
                                            .iter()
                                            .map(|v| Arg::new(&mut *stack.var_ptr_mut(*v)))
                                            .collect::<Vec<_>>(),
                                    );

                                    match ctx.tys.ty(sig.ty) {
                                        Ty::Bool => {
                                            rega.w(result as i64);
                                        }
                                        Ty::Ref(Ty::Str) => todo!(),
                                        Ty::Ref(_) => rega.w(result as i64),
                                        _ => todo!(),
                                    }
                                }
                                2 => {
                                    let result = cif.call::<u16>(
                                        CodePtr(func.into_raw().as_raw_ptr()),
                                        &args
                                            .vars
                                            .iter()
                                            .map(|v| Arg::new(&mut *stack.var_ptr_mut(*v)))
                                            .collect::<Vec<_>>(),
                                    );

                                    match ctx.tys.ty(sig.ty) {
                                        Ty::Bool => {
                                            rega.w(result as i64);
                                        }
                                        Ty::Ref(Ty::Str) => todo!(),
                                        Ty::Ref(_) => rega.w(result as i64),
                                        _ => todo!(),
                                    }
                                }
                                4 => {
                                    let result = cif.call::<u32>(
                                        CodePtr(func.into_raw().as_raw_ptr()),
                                        &args
                                            .vars
                                            .iter()
                                            .map(|v| Arg::new(&mut *stack.var_ptr_mut(*v)))
                                            .collect::<Vec<_>>(),
                                    );

                                    match ctx.tys.ty(sig.ty) {
                                        Ty::Bool => {
                                            rega.w(result as i64);
                                        }
                                        Ty::Ref(Ty::Str) => todo!(),
                                        Ty::Ref(_) => rega.w(result as i64),
                                        _ => todo!(),
                                    }
                                }
                                8 => {
                                    let result = cif.call::<u64>(
                                        CodePtr(func.into_raw().as_raw_ptr()),
                                        &args
                                            .vars
                                            .iter()
                                            .map(|v| Arg::new(&mut *stack.var_ptr_mut(*v)))
                                            .collect::<Vec<_>>(),
                                    );

                                    match ctx.tys.ty(sig.ty) {
                                        Ty::Bool => {
                                            rega.w(result as i64);
                                        }
                                        Ty::Ref(Ty::Str) => todo!(),
                                        Ty::Ref(_) => rega.w(result as i64),
                                        Ty::Struct(id) => {
                                            let bytes = ctx.tys.struct_layout(id).size;
                                            let addr = stack.anon_alloc(bytes);
                                            stack.memcpy(addr as usize, result as usize, bytes);
                                            rega.w(addr as i64);
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
                let addr = unsafe { stack.var_ptr_mut(*var).addr() as i64 };
                match reg {
                    Reg::A => rega.w(addr as i64),
                    Reg::B => regb.w(addr as i64),
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

            Air::PushIConst(var, kind, data) => match data {
                ConstData::Int(int) => {
                    stack.push_some_int(*var, *kind, *int);
                }
                ConstData::Ptr(entry) => {
                    stack.push_some_int(*var, *kind, entry.addr() as i64);
                }
            },
            Air::PushIReg { dst, kind, src } => {
                let reg = match src {
                    Reg::A => &mut rega,
                    Reg::B => &mut regb,
                };
                stack.push_some_int(*dst, *kind, reg.r());
            }
            Air::PushIVar { dst, kind, src } => {
                let int = stack.read_some_int(*src, *kind);
                stack.push_some_int(*dst, *kind, int);
            }

            Air::SwapReg => {
                let tmp = rega.r();
                rega.w(regb.r());
                regb.w(tmp);
            }
            Air::MovIConst(reg, int) => match reg {
                Reg::A => rega.w(*int),
                Reg::B => regb.w(*int),
            },
            Air::MovIVar(reg, var, kind) => {
                let int = stack.read_some_int(*var, *kind);
                match reg {
                    Reg::A => {
                        rega.w(int);
                    }
                    Reg::B => {
                        regb.w(int);
                    }
                }
            }

            Air::AddAB => {
                rega.w(rega.r() + regb.r());
            }
            Air::SubAB => {
                rega.w(rega.r() - regb.r());
            }
            Air::MulAB => {
                rega.w(rega.r() * regb.r());
            }
            Air::EqAB => {
                rega.w((rega.r() == regb.r()) as i64);
            }

            Air::Exit => {
                break;
            }
            Air::Printf => unsafe {
                libc::write(0, rega.r() as *const c_void, regb.r() as usize);
            },
            Air::PrintCStr => unsafe {
                libc::printf(rega.r() as *const c_char);
            },
        }

        instr_num += 1;
        total_instr_num += 1;
    }

    Ok(rega.r() as i32)
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
) {
    println!(
        "[\u{1b}[32m{}\u{1b}[39m:{instr_num}:{total_instr_num}]\u{1b}[1m\u{1b}[35m {instr:?}\u{1b}[39m\u{1b}[22m",
        ctx.expect_ident(current_func.func.sig.ident)
    );

    match instr {
        Air::PushIConst(_, _, _)
        | Air::Exit
        | Air::Jmp(_)
        | Air::SwapReg
        | Air::MovIConst(_, _) => {}
        Air::PushIVar { dst, kind, src } => {
            println!(
                " | {dst:?} <- {:#x}{}",
                stack.read_some_int(*src, *kind),
                kind.as_str()
            );
        }
        Air::PushIReg { dst, kind, src } => {
            println!(
                " | {dst:?} <- {:#x}{}",
                match src {
                    Reg::A => rega.r(),
                    Reg::B => regb.r(),
                },
                kind.as_str()
            );
        }
        Air::Printf => {
            println!(" | printf `{}` characters from `{:#x}`", regb.r(), rega.r());
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
        Air::MovIVar(reg, var, kind) => {
            let int = stack.read_some_int(*var, *kind);
            println!(" | {reg:?} <- {int:#x}{}", kind.as_str());
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
                &args.vars
            );
        }
        Air::AddAB => {
            let result = rega.r() + regb.r();
            println!(" | A <- {} <- A({}) + B({})", result, rega.r(), regb.r());
        }
        Air::SubAB => {
            let result = rega.r() - regb.r();
            println!(" | A <- {} <- A({}) - B({})", result, rega.r(), regb.r());
        }
        Air::MulAB => {
            let result = rega.r() * regb.r();
            println!(" | A <- {} <- A({}) * B({})", result, rega.r(), regb.r());
        }
        Air::EqAB => {
            let result = rega.r() == regb.r();
            println!(" | A <- {} <- A({}) == B({})", result, rega.r(), regb.r());
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
