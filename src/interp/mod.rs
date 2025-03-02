use crate::air::{Air, AirFunc, BlockId, Reg};
use crate::ir::ctx::Ctx;
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

pub fn run(ctx: &Ctx, air_funcs: &[AirFunc]) -> Result<i32, InterpErr> {
    let main = air_funcs
        .iter()
        .find(|f| ctx.expect_ident(f.func.sig.ident) == "main")
        .ok_or_else(|| InterpErr::NoEntry)?;
    entry(ctx, main, air_funcs, &mut Stack::default())
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
                Some((LR { func, instr }, stack_frame_size, block)) => {
                    current_func = func;
                    instrs = func.start().iter();
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
            // we don't do anything with the args because they are already registered in the stack
            Air::Call(sig, _args) => {
                call_stack.push_lr(
                    LR {
                        func: current_func,
                        instr: instr_num + 1,
                    },
                    current_block.expect("no current block"),
                );
                instr_num = 0;

                let func = air_funcs
                    .iter()
                    .find(|f| f.func.sig.ident == sig.ident)
                    .ok_or_else(|| InterpErr::InvalidFunc)?;
                current_func = func;
                instrs = func.start().iter();
                current_block = Some(func.start_block());
                continue;
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
                } else {
                    current_block = Some(*otherwise);
                    instrs = current_func.block(*otherwise).iter();
                }
            }
            Air::Jmp(block) => {
                current_block = Some(*block);
                instrs = current_func.block(*block).iter();
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
            Air::EqAB => {
                rega.w((rega.r() == regb.r()) as i64);
            }

            Air::Exit => {
                break;
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
        Air::PushIConst(_, _, _)
        | Air::Exit
        | Air::Jmp(_)
        | Air::SwapReg
        | Air::MovIConst(_, _)
        | Air::PushIVar { .. }
        | Air::PushIReg { .. } => {}
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
        Air::MovIVar(reg, var, kind) => {
            let int = stack.read_some_int(*var, *kind);
            println!(" | {reg:?} <- {int}{}", kind.as_str());
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
