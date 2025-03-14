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

    pub fn bytes(&self) -> usize {
        self.len() * 8
    }

    pub fn anon_alloc(&mut self, bytes: usize) -> Addr {
        let sp = self.sp as u64;
        self.sp += bytes;
        self.sp = (self.sp + 7) & !7; // what the fuck?

        for _ in self.stack.len()..self.sp / 8 {
            self.stack.push(0x6969696969696969);
        }

        self.start_addr() as u64 + sp
    }

    pub fn alloc(&mut self, var: Var, bytes: usize) {
        self.vars.insert(var, self.sp);
        self.anon_alloc(bytes);
    }

    pub fn sp(&self) -> usize {
        self.sp
    }

    pub fn sp_mut(&mut self) -> &mut usize {
        &mut self.sp
    }

    pub fn var_addr(&self, var: OffsetVar) -> usize {
        unsafe { self.var_ptr(var).addr() }
    }

    /// Vars cannot overlap and their spacing is maintained by sp. Caller must uphold type safety.
    pub unsafe fn var_ptr(&self, var: OffsetVar) -> *const u8 {
        let offset = self.addr(var.var) + var.offset;
        unsafe { self.stack.as_ptr().cast::<u8>().add(offset) }
    }

    /// Vars cannot overlap and their spacing is maintained by sp. Caller must uphold type safety.
    pub unsafe fn var_ptr_mut(&mut self, var: OffsetVar) -> *mut u8 {
        let offset = self.addr(var.var) + var.offset;
        unsafe { self.stack.as_mut_ptr().cast::<u8>().add(offset) }
    }

    /// `dst` and `src` must not overlap.
    pub unsafe fn memcpy(&mut self, dst: usize, src: usize, bytes: usize) {
        unsafe { std::ptr::copy_nonoverlapping(src as *mut u8, dst as *mut u8, bytes) };
    }

    #[track_caller]
    fn addr(&self, var: Var) -> usize {
        *self
            .vars
            .get(&var)
            .unwrap_or_else(|| panic!("invalid var: {var:?}"))
    }

    #[track_caller]
    pub fn read_some_bits(&mut self, var: OffsetVar, width: Width) -> Bits {
        let addr = self.var_addr(var);
        self.read_some_bits_with_addr(addr, width)
    }

    pub fn read_some_bits_with_addr(&self, addr: usize, width: Width) -> Bits {
        match width {
            Width::W8 => self.read_bits::<u8>(addr),
            Width::W16 => self.read_bits::<u16>(addr),
            Width::W32 => self.read_bits::<u32>(addr),
            Width::W64 => self.read_bits::<u64>(addr),
        }
    }

    pub fn push_some_bits(&mut self, var: OffsetVar, bits: Bits) {
        let addr = self.var_addr(var);
        match bits {
            Bits::B8(bits) => self.write::<u8>(bits, addr),
            Bits::B16(bits) => self.write::<u16>(bits, addr),
            Bits::B32(bits) => self.write::<u32>(bits, addr),
            Bits::B64(bits) => self.write::<u64>(bits, addr),
        }
    }

    pub fn read_var<I: Read>(&self, var: OffsetVar) -> I {
        self.read(self.var_addr(var))
    }

    pub fn push_var<I: WriteBits>(&mut self, var: OffsetVar, bits: I) {
        self.write(bits, self.var_addr(var));
    }

    fn read_bits<I: ReadBits>(&self, addr: usize) -> Bits {
        I::read_bits(addr as *const u8)
    }

    pub fn write_bits(&self, bits: Bits, addr: usize) {
        let ptr = addr as *mut u8;
        match bits {
            Bits::B8(bits) => bits.write_bits(ptr),
            Bits::B16(bits) => bits.write_bits(ptr),
            Bits::B32(bits) => bits.write_bits(ptr),
            Bits::B64(bits) => bits.write_bits(ptr),
        }
    }

    fn read<I: Read>(&self, addr: usize) -> I {
        assert!(
            addr <= self.end_addr(),
            "invalid stack memory: read from {addr:#x}, stack ends at {:#x}",
            self.end_addr()
        );
        assert!(
            addr >= self.start_addr(),
            "invalid stack memory: read from {addr:#x}, stack starts at {:#x}",
            self.start_addr()
        );

        let index = addr - self.start_addr();
        unsafe {
            I::read(
                std::slice::from_raw_parts(self.start_addr() as *mut u8, self.bytes()),
                index,
            )
        }
    }

    fn write<I: WriteBits>(&mut self, bits: I, addr: usize) {
        assert!(
            addr <= self.end_addr(),
            "invalid stack memory: write to {addr:#x}, stack ends at {:#x}",
            self.end_addr()
        );
        assert!(
            addr >= self.start_addr(),
            "invalid stack memory: write to {addr:#x}, stack start at {:#x}",
            self.start_addr()
        );

        bits.write_bits(addr as *mut u8);
    }

    fn start_addr(&self) -> usize {
        self.stack.as_ptr().addr()
    }

    fn end_addr(&self) -> usize {
        self.sp() + self.stack.as_ptr().addr()
    }
}

pub trait Read {
    fn read(stack: &[u8], index: usize) -> Self;
}

macro_rules! impl_read {
    ($ty:ident) => {
        impl Read for $ty {
            fn read(stack: &[u8], index: usize) -> Self {
                $ty::from_le_bytes(
                    stack[index..index + std::mem::size_of::<$ty>()]
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
    fn read_bits(ptr: *const u8) -> Bits;
}

macro_rules! impl_read_bits {
    ($ty:ident, Bits::$bits:ident) => {
        impl ReadBits for $ty {
            fn read_bits(ptr: *const u8) -> Bits {
                Bits::$bits(unsafe {
                    std::mem::transmute::<[u8; std::mem::size_of::<$ty>()], $ty>(
                        std::slice::from_raw_parts(ptr, std::mem::size_of::<$ty>())
                            [..std::mem::size_of::<$ty>()]
                            .try_into()
                            .unwrap(),
                    )
                })
            }
        }
    };
}

impl_read_bits!(u8, Bits::B8);
impl_read_bits!(u16, Bits::B16);
impl_read_bits!(u32, Bits::B32);
impl_read_bits!(u64, Bits::B64);

pub trait WriteBits {
    fn write_bits(&self, ptr: *mut u8);
}

#[macro_export]
macro_rules! impl_write_bits {
    ($ty:ident) => {
        impl WriteBits for $ty {
            fn write_bits(&self, mut ptr: *mut u8) {
                let bytes = self.to_le_bytes();
                for i in 0..std::mem::size_of::<$ty>() {
                    unsafe {
                        *ptr = bytes[i];
                        ptr = ptr.add(1);
                    }
                }
            }
        }
    };
}

impl_write_bits!(u8);
impl_write_bits!(u16);
impl_write_bits!(u32);
impl_write_bits!(u64);
impl_write_bits!(i8);
impl_write_bits!(i16);
impl_write_bits!(i32);
impl_write_bits!(i64);
impl_write_bits!(f32);
impl_write_bits!(f64);
