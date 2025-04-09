use super::InstrResult;
use super::stack::Stack;
use crate::air::data::Bss;
use crate::air::{Air, AirFunc, BlockId, Reg};
use crate::ir::ty::store::TyStore;
use crate::ir::ty::{FloatTy, Width};
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

#[derive(Debug)]
pub struct Frame<'a> {
    pub func: &'a AirFunc<'a>,
    instr: usize,
    block: BlockId,
}

impl<'a> Frame<'a> {
    pub fn new(func: &'a AirFunc<'a>, instr: usize, block: BlockId) -> Self {
        Self { func, instr, block }
    }
}

pub struct InterpCtx<'a> {
    // garauntee that bss will be in memory for raw pointer access
    _bss: &'a Bss,
    pub stack: Stack,
    pub tys: &'a TyStore,
    pub frames: Vec<Frame<'a>>,

    func_block: Option<(&'a AirFunc<'a>, BlockId)>,
    instrs: slice::Iter<'a, Air<'a>>,
    instr: usize,

    pub a: BitsReg,
    pub b: BitsReg,
}

macro_rules! debug_op {
    ($self:expr, $op:tt) => {{
        let result = $self.a.r() $op $self.b.r();
        println!(
            " | A <- {} <- A({}) {} B({})",
            result,
            $self.a.r(),
            stringify!($op),
            $self.b.r()
        );
    }};
}

macro_rules! debug_flop {
    ($self:expr, $op:tt) => {{
        let result = f64::from_bits($self.a.r()) $op f64::from_bits($self.b.r());
        println!(
            " | A <- {:.2} <- A({:.2}) {} B({:.2})",
            result,
            f64::from_bits($self.a.r()),
            stringify!($op),
            f64::from_bits($self.b.r())
        );
    }}
}

impl<'a> InterpCtx<'a> {
    pub fn new(tys: &'a TyStore, bss: &'a Bss) -> Self {
        Self {
            tys,
            _bss: bss,
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
                        if f.sig.ident == "main" {
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

    pub fn report_backtrace(&self) {
        println!("Backtrace:");
        for (i, func) in self
            .frames
            .iter()
            .rev()
            .map(|f| f.func.sig.ident)
            .enumerate()
        {
            println!("    {i}: {func}");
        }
    }

    pub fn log(&mut self, instr: &Air<'a>) {
        match self.func_block {
            Some((f, block)) => {
                println!(
                    "[\u{1b}[32m{}\u{1b}[39m:{:?}::D]\u{1b}[1m\u{1b}[35m {:?}\u{1b}[39m\u{1b}[22m",
                    f.sig.ident, block, instr
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
            | Air::CastA { .. }
            | Air::MovIConst(_, _) => {}
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
                    self.r(*addr),
                    self.r(*data),
                    width
                );
            }
            Air::Deref { dst, addr } => {
                let addr = self.r(*addr);
                println!("| Var({dst:?}) @ Addr({:#x})", addr);
                println!(
                    "| value @ Addr({:#x}) = {:#x}",
                    addr,
                    self.stack
                        .read_some_bits_with_addr(addr as usize, Width::SIZE)
                        .to_u64()
                );
            }
            Air::PushIVar { dst, width, src } => {
                println!(
                    " | {dst:?} <- {:#x}",
                    self.stack.read_some_bits(*src, *width).to_u64()
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
                                f.sig.ident, func.sig.ident, instr
                            );
                        }
                        None => {
                            println!(" | ??? -> proc {} @ instr #{}", func.sig.ident, instr);
                        }
                    },
                    None => match self.func_block {
                        Some((f, _)) => {
                            println!(" | proc {} -> ???", f.sig.ident);
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
                    sig.ident,
                    &args
                        .vars
                        .iter()
                        .map(|(ty, var)| (format!("{ty:?}"), var))
                        .collect::<Vec<_>>(),
                );
            }

            Air::MulAB(_, _) => debug_op!(self, *),
            Air::DivAB(_, _) => debug_op!(self, /),
            Air::RemAB(_, _) => debug_op!(self, %),

            Air::AddAB(_, _) => debug_op!(self, +),
            Air::SubAB(_, _) => debug_op!(self, -),

            Air::ShlAB(_, _) => debug_op!(self, <<),
            Air::ShrAB(_, _) => debug_op!(self, >>),

            Air::BandAB(_) => debug_op!(self, &),
            Air::XorAB(_) => debug_op!(self, ^),
            Air::BorAB(_) => debug_op!(self, |),

            Air::EqAB(_, _) => debug_op!(self, ==),
            Air::NEqAB(_, _) => debug_op!(self, !=),
            Air::LtAB(_, _) => debug_op!(self, <),
            Air::GtAB(_, _) => debug_op!(self, >),
            Air::LeAB(_, _) => debug_op!(self, <=),
            Air::GeAB(_, _) => debug_op!(self, >=),

            Air::FMulAB(_) => debug_flop!(self, *),
            Air::FDivAB(_) => debug_flop!(self, /),
            Air::FRemAB(_) => debug_flop!(self, %),

            Air::FAddAB(_) => debug_flop!(self, +),
            Air::FSubAB(_) => debug_flop!(self, -),

            Air::FEqAB(_) => debug_flop!(self, ==),
            Air::NFEqAB(_) => debug_flop!(self, !=),
            Air::FLtAB(_) => debug_flop!(self, <),
            Air::FGtAB(_) => debug_flop!(self, >),
            Air::FLeAB(_) => debug_flop!(self, <=),
            Air::FGeAB(_) => debug_flop!(self, >=),

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
