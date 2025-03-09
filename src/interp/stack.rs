use crate::air::{Addr, Bits, OffsetVar, Var};
use crate::ir::ty::Width;
use std::collections::HashMap;

#[derive(Debug, Default)]
pub struct Stack {
    vars: HashMap<Var, usize>,
    stack: Vec<i64>,
    sp: usize,
}

impl Stack {
    pub fn len(&self) -> usize {
        self.stack.len()
    }

    pub fn allocated(&self, var: Var) -> bool {
        self.vars.contains_key(&var)
    }

    pub fn anon_alloc(&mut self, bytes: usize) -> Addr {
        let lhs = self.stack.len();
        let rhs = self.sp + bytes;
        if lhs <= rhs {
            for _ in lhs..rhs {
                self.stack.push(0x69);
            }
        }

        let sp = self.sp;
        self.sp += bytes;
        // TODO: align less harsh
        let remainder = 8 - (self.sp % 8);
        self.sp += remainder;
        unsafe { self.stack.as_ptr().cast::<u8>().add(sp).addr() as u64 }
    }

    pub fn alloc(&mut self, var: Var, bytes: usize) {
        self.vars.insert(var, self.sp);
        self.anon_alloc(bytes);
    }

    pub fn sp_mut(&mut self) -> &mut usize {
        &mut self.sp
    }

    pub fn sp(&self) -> usize {
        self.sp
    }

    /// Vars cannot overlap and their spacing is maintained by sp. Caller must uphold type safety.
    pub unsafe fn var_ptr_mut(&mut self, var: OffsetVar) -> *mut u8 {
        if var.deref {
            panic!();
            //self.read_var::<u64>(var) as *const u64 as *mut u8
        } else {
            let offset = self.addr(var.var) + var.offset;
            unsafe { self.stack.as_mut_ptr().cast::<u8>().add(offset) }
        }
    }

    #[track_caller]
    pub fn addr(&self, var: Var) -> usize {
        *self
            .vars
            .get(&var)
            .unwrap_or_else(|| panic!("invalid var: {var:?}"))
    }

    pub unsafe fn memcpy(&mut self, dst: usize, src: usize, bytes: usize) {
        unsafe { std::ptr::copy_nonoverlapping(src as *mut u8, dst as *mut u8, bytes) };
    }

    #[track_caller]
    pub fn read_some_bits(&self, var: OffsetVar, width: Width) -> Bits {
        if var.deref {
            unimplemented!();
        }

        let addr = self.addr(var.var) + var.offset;
        match width {
            Width::W8 => self.read_bits::<u8>(addr),
            Width::W16 => self.read_bits::<u16>(addr),
            Width::W32 => self.read_bits::<u32>(addr),
            Width::W64 => self.read_bits::<u64>(addr),
        }
    }

    pub fn push_some_bits(&mut self, var: OffsetVar, bits: Bits) {
        let addr = self.addr(var.var) + var.offset;
        match bits {
            Bits::B8(bits) => self.push::<u8>(bits, addr),
            Bits::B16(bits) => self.push::<u16>(bits, addr),
            Bits::B32(bits) => self.push::<u32>(bits, addr),
            Bits::B64(bits) => self.push::<u64>(bits, addr),
        }
    }

    pub fn read_var<I: Read>(&self, var: OffsetVar) -> I {
        self.read(self.addr(var.var) + var.offset)
    }

    pub fn push_var<I: StackBits>(&mut self, var: OffsetVar, bits: I) {
        self.push(bits, self.addr(var.var) + var.offset);
    }

    fn read<I: Read>(&self, addr: usize) -> I {
        assert!(addr < self.stack.len(), "invalid stack memory");
        I::read(
            unsafe { std::mem::transmute::<&[i64], &[u8]>(self.stack.as_slice()) },
            addr,
        )
    }

    fn read_bits<I: ReadBits>(&self, addr: usize) -> Bits {
        assert!(addr < self.stack.len(), "invalid stack memory");
        I::read_bits(
            unsafe { std::mem::transmute::<&[i64], &[u8]>(self.stack.as_slice()) },
            addr,
        )
    }

    fn push<I: StackBits>(&mut self, bits: I, addr: usize) {
        let size = std::mem::size_of::<I>();
        if addr + size >= self.stack.len() {
            for _ in self.stack.len()..(addr + size) {
                self.stack.push(0x69);
            }
        }

        bits.stack_bits(
            unsafe { std::mem::transmute::<&mut [i64], &mut [u8]>(self.stack.as_mut_slice()) },
            addr,
        );
    }
}

pub trait Read {
    fn read(stack: &[u8], addr: usize) -> Self;
}

macro_rules! impl_read {
    ($ty:ident) => {
        impl Read for $ty {
            fn read(stack: &[u8], addr: usize) -> Self {
                $ty::from_le_bytes(
                    stack[addr..addr + std::mem::size_of::<$ty>()]
                        .try_into()
                        .unwrap(),
                )
            }
        }
    };
}

impl_read!(u8);
impl_read!(u16);
impl_read!(u32);
impl_read!(u64);
impl_read!(i8);
impl_read!(i16);
impl_read!(i32);
impl_read!(i64);
impl_read!(f32);
impl_read!(f64);

trait ReadBits {
    fn read_bits(stack: &[u8], addr: usize) -> Bits;
}

macro_rules! impl_readable {
    ($ty:ident, Bits::$bits:ident) => {
        impl ReadBits for $ty {
            fn read_bits(stack: &[u8], addr: usize) -> Bits {
                Bits::$bits(unsafe {
                    std::mem::transmute::<[u8; std::mem::size_of::<$ty>()], $ty>(
                        stack[addr..addr + std::mem::size_of::<$ty>()]
                            .try_into()
                            .unwrap(),
                    )
                })
            }
        }
    };
}

impl_readable!(u8, Bits::B8);
impl_readable!(u16, Bits::B16);
impl_readable!(u32, Bits::B32);
impl_readable!(u64, Bits::B64);

pub trait StackBits {
    fn stack_bits(&self, stack: &mut [u8], addr: usize);
}

#[macro_export]
macro_rules! impl_stack_bits {
    ($ty:ident) => {
        impl StackBits for $ty {
            fn stack_bits(&self, stack: &mut [u8], addr: usize) {
                let bytes = self.to_le_bytes();
                for i in 0..std::mem::size_of::<$ty>() {
                    //println!("addr: {} = byte: {}", addr + i, bytes[i]);
                    stack[addr + i] = bytes[i];
                }
            }
        }
    };
}

impl_stack_bits!(u8);
impl_stack_bits!(u16);
impl_stack_bits!(u32);
impl_stack_bits!(u64);
impl_stack_bits!(i8);
impl_stack_bits!(i16);
impl_stack_bits!(i32);
impl_stack_bits!(i64);
impl_stack_bits!(f32);
impl_stack_bits!(f64);
