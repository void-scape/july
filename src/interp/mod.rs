use crate::air::{Air, AirFunc, Reg};
use crate::ir::ctx::Ctx;
use crate::ir::ty::Ty;
use anstream::println;
use stack::*;

mod stack;

#[derive(Debug)]
pub enum InterpErr {
    NoEntry,
    InvalidFunc,
    NoReturn,
    DryStream,
}

pub type IResult = Result<(), InterpErr>;

pub fn run(ctx: &Ctx, air_funcs: &[AirFunc]) -> Result<i32, InterpErr> {
    let main = air_funcs
        .iter()
        .find(|f| ctx.expect_ident(f.func.sig.ident) == "main")
        .ok_or_else(|| InterpErr::NoEntry)?;
    entry(ctx, main, air_funcs, &mut Stack::default())
}

#[derive(Default)]
pub struct CallStack<'a> {
    stack: Vec<(LR<'a>, usize)>,
}

impl<'a> CallStack<'a> {
    pub fn push_lr(&mut self, lr: LR<'a>) {
        self.stack.push((lr, 0));
    }

    pub fn grow_stack(&mut self, bytes: usize) {
        if let Some((_, stack)) = self.stack.last_mut() {
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
) -> Result<i32, InterpErr> {
    let mut current_func = main;
    let mut instrs = main.instrs.iter();
    let mut call_stack = CallStack::default();
    let mut rega = Rega::default();
    let mut regb = Rega::default();

    let mut total_instr_num = 0;
    let mut instr_num = 0;

    loop {
        let Some(instr) = instrs.next() else {
            return Err(InterpErr::DryStream);
        };

        debug_instr(
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

        match instr {
            Air::Ret => match call_stack.stack.last() {
                Some((LR { func, instr }, stack_frame_size)) => {
                    current_func = func;
                    instrs = func.instrs.iter();
                    for _ in 0..*instr {
                        instrs.next();
                    }
                    instr_num = *instr;
                    *stack.sp_mut() -= stack_frame_size;
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
            Air::Call(sig) => {
                call_stack.push_lr(LR {
                    func: current_func,
                    instr: instr_num + 1,
                });
                instr_num = 0;

                let func = air_funcs
                    .iter()
                    .find(|f| f.func.sig.ident == sig.ident)
                    .ok_or_else(|| InterpErr::InvalidFunc)?;
                current_func = func;
                instrs = func.instrs.iter();
                continue;
            }

            Air::SAlloc(var, bytes) => {
                stack.alloc(*var, *bytes);
                call_stack.grow_stack(*bytes);
            }

            Air::Addr(reg, var) => {
                let addr = stack.addr(var.var) + var.offset;
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
                stack.memcpy(dst, src, *bytes);
            }

            Air::PushIConst(var, kind, int) => {
                stack.push_some_int(*var, *kind, *int);
            }
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
        }

        instr_num += 1;
        total_instr_num += 1;
    }

    Ok(rega.r() as i32)
}

fn debug_instr(
    ctx: &Ctx,
    current_func: &AirFunc,
    stack: &Stack,
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
        Air::SAlloc(var, bytes) => {
            println!(" | Addr({:?}) <- {:#x}", var, stack.sp());
            println!(" | SP({:?}) += {}", stack.sp(), bytes);
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
            let addr = stack.addr(var.var) + var.offset;
            println!(" | Addr({:?}) = {:#x}", var.var, addr,);
            println!(" | Offset({})", var.offset);
        }
        Air::SwapReg => {}
        Air::PushIConst(_, _, _) => {}
        Air::MovIConst(_, _) => {}
        Air::MovIVar(reg, var, kind) => {
            let int = stack.read_some_int(*var, *kind);
            println!(" | {reg:?} <- {int}{}", Ty::Int(*kind).as_str());
        }
        Air::Ret => {
            match call_stack.stack.last() {
                Some((LR { func, instr }, _)) => {
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
        Air::Call(sig) => {
            println!(" | call proc [{}]", ctx.expect_ident(sig.ident));
        }
        Air::PushIVar { .. } => {}
        Air::PushIReg { .. } => {}
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
    }
}
