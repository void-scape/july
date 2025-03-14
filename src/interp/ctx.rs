use super::stack::Stack;
use super::InstrResult;
use crate::air::{Air, AirFunc, BlockId, Reg};
use crate::ir::ctx::Ctx;
use crate::ir::ty::FloatTy;
use std::ops::{BitXor, Deref};
use std::slice;

#[derive(Default)]
pub struct BitsReg(u64);

impl BitsReg {
    pub fn w(&mut self, val: u64) {
        self.0 = val;
    }

    pub fn r(&self) -> u64 {
        self.0
    }
}

pub struct Frame<'a> {
    func: &'a AirFunc<'a>,
    instr: usize,
    block: BlockId,
}

impl<'a> Frame<'a> {
    pub fn new(func: &'a AirFunc<'a>, instr: usize, block: BlockId) -> Self {
        Self { func, instr, block }
    }
}

pub struct InterpCtx<'a> {
    pub stack: Stack,
    pub ctx: &'a Ctx<'a>,
    frames: Vec<Frame<'a>>,

    func_block: Option<(&'a AirFunc<'a>, BlockId)>,
    instrs: slice::Iter<'a, Air<'a>>,
    instr: usize,

    pub a: BitsReg,
    pub b: BitsReg,
}

impl<'a> Deref for InterpCtx<'a> {
    type Target = Ctx<'a>;

    fn deref(&self) -> &Self::Target {
        self.ctx
    }
}

impl<'a> InterpCtx<'a> {
    pub fn new(ctx: &'a Ctx<'a>) -> Self {
        Self {
            ctx,
            stack: Stack::default(),
            frames: Vec::new(),
            func_block: None,
            instrs: [].iter(),
            instr: 0,
            a: BitsReg::default(),
            b: BitsReg::default(),
        }
    }

    pub fn next(&mut self) -> Option<&'a Air<'a>> {
        self.instrs.next()
    }

    pub fn incr_instr(&mut self) {
        self.instr += 1;
    }

    pub fn consts(&mut self, consts: &'a [Air<'a>]) {
        self.instrs = consts.iter();
    }

    pub fn start_func(&mut self, func: &'a AirFunc<'a>) {
        match self.func_block {
            Some((func, block)) => {
                self.push_frame(Frame::new(func, self.instr + 1, block));
            }
            None => {}
        }

        self.func_block = Some((func, func.start_block()));
        self.instrs = func.start().iter();
        self.instr = 0;
    }

    #[track_caller]
    pub fn start_block(&mut self, block: BlockId) {
        match self.func_block {
            Some((func, _)) => {
                self.func_block = Some((func, block));
                self.instrs = func.block(block).iter();
                self.instr = 0;
            }
            None => {
                panic!("called `start_block` outside of a function")
            }
        }
    }

    pub fn pop_frame(&mut self) -> InstrResult {
        match self.frames.pop() {
            Some(Frame { func, instr, block }) => {
                self.instr = instr;
                self.func_block = Some((func, block));
                self.instrs = func.block(block).iter();
                for _ in 0..instr {
                    self.instrs.next();
                }

                InstrResult::Continue
            }
            None => {
                match self.func_block {
                    Some((f, _)) => {
                        if self.expect_ident(f.func.sig.ident) == "main" {
                            return InstrResult::Break;
                        }
                    }
                    None => {}
                }
                panic!("no return");
            }
        }
    }

    pub fn push_frame(&mut self, frame: Frame<'a>) {
        self.frames.push(frame);
    }

    pub fn r(&mut self, reg: Reg) -> u64 {
        match reg {
            Reg::A => self.a.r(),
            Reg::B => self.b.r(),
        }
    }

    pub fn w(&mut self, reg: Reg, data: u64) {
        match reg {
            Reg::A => self.a.w(data),
            Reg::B => self.b.w(data),
        }
    }

    pub fn log(&mut self, instr: &Air<'a>) {
        match self.func_block {
            Some((f, block)) => {
                println!(
                    "[\u{1b}[32m{}\u{1b}[39m:{:?}::D]\u{1b}[1m\u{1b}[35m {:?}\u{1b}[39m\u{1b}[22m",
                    self.expect_ident(f.func.sig.ident),
                    block,
                    instr
                );
            }
            None => {
                println!(
                    "[\u{1b}[32m\u{1b}[39m:::D]\u{1b}[1m\u{1b}[35m {:?}\u{1b}[39m\u{1b}[22m",
                    instr
                );
            }
        }

        match instr {
            Air::PushIConst(_, _)
            | Air::Exit
            | Air::Jmp(_)
            | Air::SwapReg
            | Air::ReadSP(_)
            | Air::WriteSP(_)
            | Air::MovIConst(_, _) => {}
            Air::Fu32 => {
                let float = f32::from_bits(self.a.r() as u32);
                println!(" | {} <- {}", float as u32 as u64, float);
            }
            Air::Read { dst, addr, width } => {
                println!(
                    " | Reg({:?}) <- {:?} <- Read(Reg({:?}) = Addr({:#x})) @ {:?}",
                    dst,
                    self.stack.read_some_bits_with_addr(
                        match addr {
                            Reg::A => self.a.r(),
                            Reg::B => self.b.r(),
                        } as usize,
                        *width
                    ),
                    addr,
                    match addr {
                        Reg::A => self.a.r(),
                        Reg::B => self.b.r(),
                    },
                    width
                );
            }
            Air::Write { data, addr, width } => {
                println!(
                    " | Addr({:#x}) <- {:#x} @ {:?}",
                    match addr {
                        Reg::A => self.a.r(),
                        Reg::B => self.b.r(),
                    },
                    match data {
                        Reg::A => self.a.r(),
                        Reg::B => self.b.r(),
                    },
                    width
                );
            }
            Air::XOR(mask) => {
                println!(
                    " | Reg(A) <- {:#x} <- {:#x} XOR {:#x}",
                    mask.bitxor(self.a.r()),
                    self.a.r(),
                    mask
                );
            }
            Air::PushIVar { dst, width, src } => {
                println!(
                    " | {dst:?} <- {:?}",
                    self.stack.read_some_bits(*src, *width)
                );
            }
            Air::PushIReg { dst, width, src } => {
                println!(
                    " | {dst:?} <- {} @ {width:?}",
                    match src {
                        Reg::A => self.a.r(),
                        Reg::B => self.b.r(),
                    },
                );
            }
            Air::PrintCStr => {
                println!(" | print_c_str @ `{}`", self.a.r());
            }
            Air::IfElse {
                condition,
                then,
                otherwise,
            } => {
                println!(
                    " | if ({:#x}), then {:?}, otherwise {:?}",
                    match condition {
                        Reg::A => self.a.r(),
                        Reg::B => self.b.r(),
                    },
                    then,
                    otherwise
                );
            }
            Air::SAlloc(var, bytes) => {
                println!(" | Addr({:?}) <- {:#x}", var, self.stack.sp());
                println!(" | SP({:?}) += {}", self.stack.sp(), bytes);
            }
            Air::MemCpy { dst, src, bytes } => {
                println!(
                    " | memcpy {:#x} <- {:#x} @ {bytes} bytes",
                    match dst {
                        Reg::A => self.a.r(),
                        Reg::B => self.b.r(),
                    },
                    match src {
                        Reg::A => self.a.r(),
                        Reg::B => self.b.r(),
                    }
                );
            }
            Air::Addr(_, var) => {
                let addr = self.stack.var_addr(*var);
                println!(" | Addr({:?}) = {:#x}", var.var, addr);
                println!(" | Offset({})", var.offset);
            }
            Air::MovIVar(reg, var, width) => {
                let int = self.stack.read_some_bits(*var, *width);
                println!(" | {reg:?} <- {int:?}");
            }
            Air::Ret => {
                match self.frames.last() {
                    Some(Frame { func, instr, .. }) => match self.func_block {
                        Some((f, _)) => {
                            println!(
                                " | proc {} -> proc {} @ instr #{}",
                                self.expect_ident(f.func.sig.ident),
                                self.expect_ident(func.func.sig.ident),
                                instr
                            );
                        }
                        None => {
                            println!(
                                " | ??? -> proc {} @ instr #{}",
                                self.expect_ident(func.func.sig.ident),
                                instr
                            );
                        }
                    },
                    None => match self.func_block {
                        Some((f, _)) => {
                            println!(" | proc {} -> ???", self.expect_ident(f.func.sig.ident),);
                        }
                        None => {
                            println!(" | ??? -> ???");
                        }
                    },
                }
                println!(" | A = {:#x}", self.a.r());
            }
            Air::Call(sig, args) => {
                println!(
                    " | call proc [{}({:?})]",
                    self.expect_ident(sig.ident),
                    &args
                        .vars
                        .iter()
                        .map(|(ty, var)| (self.ty_str(*ty), var))
                        .collect::<Vec<_>>(),
                );
            }
            Air::AddAB(_) => {
                let result = self.a.r() + self.b.r();
                println!(
                    " | A <- {} <- A({}) + B({})",
                    result,
                    self.a.r(),
                    self.b.r()
                );
            }
            Air::SubAB(_) => {
                let result = self.a.r() - self.b.r();
                println!(
                    " | A <- {} <- A({}) - B({})",
                    result,
                    self.a.r(),
                    self.b.r()
                );
            }
            Air::MulAB(_) => {
                let result = self.a.r() * self.b.r();
                println!(
                    " | A <- {} <- A({}) * B({})",
                    result,
                    self.a.r(),
                    self.b.r()
                );
            }
            Air::DivAB(_) => {
                let result = self.a.r() / self.b.r();
                println!(
                    " | A <- {} <- A({}) / B({})",
                    result,
                    self.a.r(),
                    self.b.r()
                );
            }

            Air::XorAB(_) => {
                let result = self.a.r() ^ self.b.r();
                println!(
                    " | A <- {} <- A({}) ^ B({})",
                    result,
                    self.a.r(),
                    self.b.r()
                );
            }

            Air::EqAB(_) => {
                let result = self.a.r() == self.b.r();
                println!(
                    " | A <- {} <- A({}) == B({})",
                    result,
                    self.a.r(),
                    self.b.r()
                );
            }
            Air::NEqAB(_) => {
                let result = self.a.r() != self.b.r();
                println!(
                    " | A <- {} <- A({}) != B({})",
                    result,
                    self.a.r(),
                    self.b.r()
                );
            }

            Air::FAddAB(_) => {
                let lhs = self.a.r();
                let rhs = self.b.r();
                let result = f64::from_bits(lhs) + f64::from_bits(rhs);
                println!(" | A <- {:.4} <- A({}) + B({})", result, lhs, rhs);
            }
            Air::FSubAB(_) => {
                let lhs = self.a.r();
                let rhs = self.b.r();
                let result = f64::from_bits(lhs) - f64::from_bits(rhs);
                println!(" | A <- {:.4} <- A({}) - B({})", result, lhs, rhs);
            }
            Air::FMulAB(_) => {
                let lhs = self.a.r();
                let rhs = self.b.r();
                let result = f64::from_bits(lhs) * f64::from_bits(rhs);
                println!(" | A <- {:.4} <- A({}) * B({})", result, lhs, rhs);
            }
            Air::FDivAB(_) => {
                let lhs = self.a.r();
                let rhs = self.b.r();
                let result = f64::from_bits(lhs) / f64::from_bits(rhs);
                println!(" | A <- {:.4} <- A({}) / B({})", result, lhs, rhs);
            }

            Air::FEqAB(_) => {
                let lhs = self.a.r();
                let rhs = self.b.r();
                let result = f64::from_bits(lhs) == f64::from_bits(rhs);
                println!(" | A <- {} <- A({}) == B({})", result, lhs, rhs);
            }
            Air::NFEqAB(_) => {
                let lhs = self.a.r();
                let rhs = self.b.r();
                let result = f64::from_bits(lhs) != f64::from_bits(rhs);
                println!(" | A <- {} <- A({}) != B({})", result, lhs, rhs);
            }

            Air::FSqrt(ty) => {
                let arg = self.a.r();
                match ty {
                    FloatTy::F32 => {
                        let result = f32::from_bits(arg as u32).sqrt();
                        println!(" | A <- {:.4} <- A({})", result, arg);
                    }
                    FloatTy::F64 => {
                        let result = f64::from_bits(arg).sqrt();
                        println!(" | A <- {:.4} <- A({})", result, arg);
                    }
                }
            }
        }
    }
}
