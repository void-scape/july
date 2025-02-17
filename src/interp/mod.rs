use crate::air::{Air, AirFunc, Reg, Var};
use crate::ir::ctx::Ctx;
use crate::ir::ty::IntKind;
use anstream::println;
use std::collections::HashMap;

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
    stack: Vec<LR<'a>>,
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

#[derive(Default)]
struct Stack {
    ints: HashMap<Var, (IntKind, i64)>,
}

impl Stack {
    pub fn push_int(&mut self, var: Var, kind: IntKind, int: i64) {
        self.ints.insert(var, (kind, int));
    }

    pub fn expect_int(&self, var: Var) -> (IntKind, i64) {
        *self.ints.get(&var).expect("invalid int var")
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
            total_instr_num,
        );

        match instr {
            Air::Ret => match call_stack.stack.last() {
                Some(LR { func, instr }) => {
                    current_func = func;
                    instrs = func.instrs.iter();
                    for _ in 0..*instr {
                        instrs.next();
                    }
                    instr_num = *instr;
                    call_stack.stack.pop();
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
                call_stack.stack.push(LR {
                    func: current_func,
                    instr: instr_num,
                });
                instr_num = 0;

                let func = air_funcs
                    .iter()
                    .find(|f| f.func.sig.ident == sig.ident)
                    .ok_or_else(|| InterpErr::InvalidFunc)?;
                current_func = func;
                instrs = func.instrs.iter();
            }

            Air::PushIConst(var, kind, int) => stack.push_int(*var, *kind, *int),
            Air::PushIReg { dst, kind, src } => {
                let reg = match src {
                    Reg::A => &mut rega,
                    Reg::B => &mut regb,
                };
                stack.push_int(*dst, *kind, reg.r());
            }
            Air::PushIVar { dst, kind, src } => {
                let (int_kind, int) = stack.expect_int(*src);
                assert_eq!(*kind, int_kind);
                stack.push_int(*dst, *kind, int);
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
            Air::MovIVar(reg, var) => {
                let (_, int) = stack.expect_int(*var);
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
    total_instr_num: usize,
) {
    println!("[{total_instr_num}]\u{1b}[1m\u{1b}[35m {instr:?}\u{1b}[39m\u{1b}[22m");
    match instr {
        Air::SwapReg => {}
        Air::PushIConst(_, _, _) => {}
        Air::MovIConst(_, _) => {}
        Air::MovIVar(reg, var) => {
            let (_, int) = stack.expect_int(*var);
            println!(" | {reg:?} <- `{int}`");
        }
        Air::Ret => {
            match call_stack.stack.last() {
                Some(LR { func, instr }) => {
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
            println!(" | A = `{}`", rega.r());
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
