use crate::air::{IntKind, OffsetVar, Var};

#[derive(Debug, Default)]
pub struct Stack {
    vars: HashMap<Var, usize>,
    stack: Vec<i64>,
    sp: usize,
}

impl Stack {
    pub fn alloc(&mut self, var: Var, bytes: usize) {
        let lhs = self.stack.len();
        let rhs = self.sp + bytes;
        if lhs <= rhs {
            for _ in lhs..rhs {
                self.stack.push(0x69);
            }
        }

        self.vars.insert(var, self.sp);
        self.sp += bytes;
    }

    pub fn sp_mut(&mut self) -> &mut usize {
        &mut self.sp
    }

    pub fn sp(&self) -> usize {
        self.sp
    }

    #[track_caller]
    pub fn read_some_int(&self, var: OffsetVar, kind: IntKind) -> i64 {
        let addr = self.addr(var.var) + var.offset;
        match kind {
            IntKind::U8 => self.read_int::<u8>(addr),
            IntKind::U16 => self.read_int::<u16>(addr),
            IntKind::U32 => self.read_int::<u32>(addr),
            IntKind::U64 => self.read_int::<u64>(addr),

            IntKind::I8 => self.read_int::<i8>(addr),
            IntKind::I16 => self.read_int::<i16>(addr),
            IntKind::I32 => self.read_int::<i32>(addr),
            IntKind::I64 => self.read_int::<i64>(addr),
        }
    }

    #[track_caller]
    pub fn push_some_int(&mut self, var: OffsetVar, kind: IntKind, int: i64) {
        let addr = self.addr(var.var) + var.offset;
        match kind {
            IntKind::U8 => {
                self.push_int(int as u8, addr);
            }
            IntKind::U16 => {
                self.push_int(int as u16, addr);
            }
            IntKind::U32 => {
                self.push_int(int as u32, addr);
            }
            IntKind::U64 => {
                self.push_int(int as u64, addr);
            }

            IntKind::I8 => {
                self.push_int(int as i8, addr);
            }
            IntKind::I16 => {
                self.push_int(int as i16, addr);
            }
            IntKind::I32 => {
                self.push_int(int as i32, addr);
            }
            IntKind::I64 => {
                self.push_int(int as i64, addr);
            }
        }
    }

    #[track_caller]
    pub fn addr(&self, var: Var) -> usize {
        *self.vars.get(&var).expect("invalid var")
    }

    pub fn memcpy(&mut self, dst: usize, src: usize, bytes: usize) {
        unsafe {
            std::mem::transmute::<&mut [i64], &mut [u8]>(self.stack.as_mut_slice())
                .copy_within(src..src + bytes, dst)
        };
    }

    fn read_int<I: Readable>(&self, addr: usize) -> i64 {
        assert!(addr < self.stack.len(), "invalid stack memory");
        I::read(
            unsafe { std::mem::transmute::<&[i64], &[u8]>(self.stack.as_slice()) },
            addr,
        )
    }

    fn push_int<I: Stackable>(&mut self, int: I, addr: usize) {
        let size = std::mem::size_of::<I>();
        if addr + size >= self.stack.len() {
            for _ in self.stack.len()..(addr + size) {
                self.stack.push(0x69);
            }
        }

        int.stack(
            unsafe { std::mem::transmute::<&mut [i64], &mut [u8]>(self.stack.as_mut_slice()) },
            addr,
        );
    }
}

trait Readable {
    fn read(stack: &[u8], addr: usize) -> i64;
}

impl Readable for u8 {
    fn read(stack: &[u8], addr: usize) -> i64 {
        (stack[addr] as u64) as i64
    }
}

impl Readable for u16 {
    fn read(stack: &[u8], addr: usize) -> i64 {
        unsafe {
            (std::mem::transmute::<[u8; 2], u16>(stack[addr..addr + 2].try_into().unwrap()) as u64)
                as i64
        }
    }
}

impl Readable for u32 {
    fn read(stack: &[u8], addr: usize) -> i64 {
        unsafe {
            (std::mem::transmute::<[u8; 4], u32>(stack[addr..addr + 4].try_into().unwrap()) as u64)
                as i64
        }
    }
}

impl Readable for u64 {
    fn read(stack: &[u8], addr: usize) -> i64 {
        unsafe {
            std::mem::transmute::<[u8; 8], u64>(stack[addr..addr + 8].try_into().unwrap()) as i64
        }
    }
}

impl Readable for i8 {
    fn read(stack: &[u8], addr: usize) -> i64 {
        stack[addr] as i64
    }
}

impl Readable for i16 {
    fn read(stack: &[u8], addr: usize) -> i64 {
        unsafe {
            std::mem::transmute::<[u8; 2], i16>(stack[addr..addr + 2].try_into().unwrap()) as i64
        }
    }
}

impl Readable for i32 {
    fn read(stack: &[u8], addr: usize) -> i64 {
        unsafe {
            std::mem::transmute::<[u8; 4], i32>(stack[addr..addr + 4].try_into().unwrap()) as i64
        }
    }
}

impl Readable for i64 {
    fn read(stack: &[u8], addr: usize) -> i64 {
        unsafe { std::mem::transmute::<[u8; 8], i64>(stack[addr..addr + 8].try_into().unwrap()) }
    }
}

trait Stackable {
    fn stack(&self, stack: &mut [u8], addr: usize);
}

use std::collections::HashMap;

use crate::impl_stackable;
impl_stackable!(u8, u16, u32, u64);
impl_stackable!(i8, i16, i32, i64);

#[macro_export]
macro_rules! impl_stackable {
    ($ty8:ident, $ty16:ident, $ty32:ident, $ty64:ident) => {
        impl Stackable for $ty8 {
            fn stack(&self, stack: &mut [u8], addr: usize) {
                stack[addr] = *self as u8;
            }
        }

        impl Stackable for $ty16 {
            fn stack(&self, stack: &mut [u8], addr: usize) {
                stack[addr] = *self as u8;
                stack[addr + 1] = (*self >> 8) as u8;
            }
        }

        impl Stackable for $ty32 {
            fn stack(&self, stack: &mut [u8], addr: usize) {
                stack[addr] = *self as u8;
                stack[addr + 1] = (*self >> 8) as u8;
                stack[addr + 2] = (*self >> 16) as u8;
                stack[addr + 3] = (*self >> 24) as u8;
            }
        }

        impl Stackable for $ty64 {
            fn stack(&self, stack: &mut [u8], addr: usize) {
                stack[addr] = *self as u8;
                stack[addr + 1] = (*self >> 8) as u8;
                stack[addr + 2] = (*self >> 16) as u8;
                stack[addr + 3] = (*self >> 24) as u8;
                stack[addr + 4] = (*self >> 32) as u8;
                stack[addr + 5] = (*self >> 40) as u8;
                stack[addr + 6] = (*self >> 48) as u8;
                stack[addr + 7] = (*self >> 56) as u8;
            }
        }
    };
}
