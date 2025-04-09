#![feature(maybe_uninit_slice)]

use std::alloc::Layout;
use std::cell::{Cell, RefCell};
use std::fmt::Debug;
use std::mem::MaybeUninit;
use std::ptr::{NonNull, slice_from_raw_parts_mut};

#[derive(Debug, Default)]
pub struct BlobArena {
    arena: Arena<u8>,
}

impl BlobArena {
    pub const fn new() -> Self {
        Self {
            arena: Arena::new(),
        }
    }

    #[track_caller]
    pub fn alloc<'a, T>(&self, elem: T) -> &'a mut T
    where
        T: Copy,
    {
        assert!(std::mem::size_of::<T>() > 0);
        self.align_grow::<T>(Layout::new::<T>());

        let mem = self.arena.ptr.get() as *mut T;
        unsafe {
            std::ptr::write(mem, elem);
            self.arena
                .ptr
                .set(self.arena.ptr.get().add(std::mem::size_of::<T>()));
            &mut *mem
        }
    }

    pub fn alloc_str<'a>(&self, str: &str) -> &'a str {
        unsafe { std::mem::transmute(self.alloc_slice(str.as_bytes())) }
    }

    #[track_caller]
    pub fn alloc_slice<'a, T>(&self, slice: &[T]) -> &'a mut [T]
    where
        T: Copy,
    {
        assert!(
            !slice.is_empty(),
            "tried to `BlobArena::alloc_slice` an empty slice"
        );
        assert!(std::mem::size_of::<T>() > 0);
        let layout = Layout::for_value::<[T]>(slice);
        self.align_grow::<T>(layout);

        let mem = self.arena.ptr.get() as *mut T;
        unsafe {
            mem.copy_from_nonoverlapping(slice.as_ptr(), slice.len());
            self.arena.ptr.set(self.arena.ptr.get().add(layout.size()));
            &mut *slice_from_raw_parts_mut(mem, slice.len())
        }
    }

    pub fn alloc_str_ptr<'a>(&self, str: &str) -> *mut u8 {
        self.alloc_slice_ptr(str.as_bytes())
    }

    #[track_caller]
    pub fn alloc_slice_ptr<'a, T>(&self, slice: &[T]) -> *mut T
    where
        T: Copy,
    {
        assert!(!slice.is_empty());
        assert!(std::mem::size_of::<T>() > 0);
        assert!(!std::mem::needs_drop::<T>());
        let layout = Layout::for_value::<[T]>(slice);
        self.align_grow::<T>(layout);

        let mem = self.arena.ptr.get() as *mut T;
        unsafe {
            mem.copy_from_nonoverlapping(slice.as_ptr(), slice.len());
            self.arena.ptr.set(self.arena.ptr.get().add(layout.size()));
            mem
            //&mut *slice_from_raw_parts_mut(mem, slice.len())
        }
    }

    fn align_grow<T>(&self, layout: Layout) {
        if self.arena.remaining_bytes() < layout.size() {
            self.arena.grow();
        }
        assert!(self.arena.remaining_bytes() >= layout.size());

        let bytes = self.arena.ptr.get().align_offset(layout.align());
        unsafe { self.arena.ptr.set(self.arena.ptr.get().add(bytes)) };
    }
}

pub struct Arena<T> {
    ptr: Cell<*mut T>,
    last: Cell<*mut T>,
    chunks: RefCell<Vec<Chunk<T>>>,
    grow_size: RefCell<usize>,
    // _owned: PhantomData<T>,
}

impl<T> Debug for Arena<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Arena").finish()
    }
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(unused)]
impl<T> Arena<T> {
    pub const fn new() -> Self {
        assert!(
            std::mem::size_of::<T>() > 0,
            "arena does not support zero sized types"
        );

        Self {
            ptr: Cell::new(std::ptr::null_mut()),
            last: Cell::new(std::ptr::null_mut()),
            chunks: RefCell::new(Vec::new()),
            grow_size: RefCell::new(4096),
            //_owned: PhantomData,
        }
    }

    pub fn alloc(&self, item: T) -> &mut T {
        if self.last == self.ptr {
            self.grow();
        }

        unsafe {
            let ptr = self.ptr.get();
            std::ptr::write(ptr, item);
            self.ptr.set(self.ptr.get().add(1));
            &mut *ptr
        }
    }

    pub fn remaining_bytes(&self) -> usize {
        if self.last == self.ptr {
            self.grow();
        }

        let curr = self.ptr.get().addr();
        let end = self.last.get().addr();
        end.saturating_sub(curr)
    }

    fn grow(&self) {
        let mut chunks = self.chunks.borrow_mut();

        if let Some(chunk) = chunks.last_mut() {
            chunk.elems = self.ptr.get().addr().saturating_sub(chunk.as_ptr().addr())
                / std::mem::size_of::<T>();
        }

        let elems = *self.grow_size.borrow() / std::mem::size_of::<T>();
        let mut chunk = Chunk::new(elems);
        self.ptr.set(chunk.as_ptr());
        self.last.set(chunk.end());
        chunks.push(chunk);

        *self.grow_size.borrow_mut() *= 2;
    }

    fn destroy_last(&self, mut chunk: Chunk<T>) {
        let start = chunk.as_ptr().addr();
        let end = self.ptr.get().addr();
        let diff = end.saturating_sub(start);

        if diff > 0 {
            chunk.destroy(diff / std::mem::size_of::<T>());
        }
    }
}

//unsafe impl<#[may_dangle] T> Drop for Arena<T> {
impl<T> Drop for Arena<T> {
    fn drop(&mut self) {
        let mut chunks = self.chunks.borrow_mut();
        if let Some(chunk) = chunks.pop() {
            self.destroy_last(chunk);

            for chunk in chunks.iter_mut() {
                chunk.destroy(chunk.elems);
            }
        }
    }
}

#[derive(PartialEq, Eq)]
struct Chunk<T> {
    buf: NonNull<[MaybeUninit<T>]>,
    elems: usize,
}

impl<T> Chunk<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            buf: NonNull::from(Box::leak(Box::new_uninit_slice(capacity))),
            elems: 0,
        }
    }

    pub fn as_ptr(&mut self) -> *mut T {
        self.buf.as_ptr() as *mut T
    }

    pub fn end(&mut self) -> *mut T {
        unsafe { self.as_ptr().add(self.buf.len()) }
    }

    pub fn destroy(&mut self, elems: usize) {
        if std::mem::needs_drop::<T>() {
            unsafe {
                let buf = self.buf.as_mut();
                std::ptr::drop_in_place(&mut buf[..elems].assume_init_mut());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;

    trait TestType: Debug + PartialEq {
        fn from_usize(i: usize) -> Self;
        fn as_usize(&self) -> usize;
    }

    fn test_simple<T: TestType>() {
        let arena = Arena::<T>::default();
        let val = arena.alloc(T::from_usize(1));
        assert_eq!(T::from_usize(1), *val);
        let val = arena.alloc(T::from_usize(2));
        assert_eq!(T::from_usize(2), *val);
    }

    fn test_blob_simple<T: TestType + Copy>() {
        let arena = BlobArena::default();
        let old_val = arena.alloc(T::from_usize(1));
        assert_eq!(T::from_usize(1), *old_val);
        let val = arena.alloc(T::from_usize(2));
        assert_eq!(T::from_usize(2), *val);
        assert_eq!(T::from_usize(1), *old_val);
    }

    fn test_blob_slice<T: TestType + Copy>() {
        let arena = BlobArena::default();
        let slice = &[T::from_usize(0), T::from_usize(1), T::from_usize(2)];
        let new_slice = arena.alloc_slice(slice);
        assert_eq!(new_slice, slice);
        let t = T::from_usize(12);
        assert_eq!(t, *arena.alloc(t));
        assert_eq!(new_slice, slice);
        let t = T::from_usize(64);
        assert_eq!(t, *arena.alloc(t));
        assert_eq!(new_slice, slice);
    }

    fn test_many_chunks<T: TestType>() {
        let elems = 1024 * 3;
        let arena = Arena::<T>::default();
        let mut refs = Vec::with_capacity(elems);
        for i in 0..elems {
            refs.push(arena.alloc(T::from_usize(i)));
        }

        for i in 0..elems {
            assert_eq!(i, refs[i].as_usize());
            *refs[i] = T::from_usize(0);
        }

        for i in 0..elems {
            assert_eq!(*refs[i], T::from_usize(0));
        }
    }

    fn test_blob_many_chunks<T: TestType + Copy>() {
        let elems = 1024 * 3;
        let arena = BlobArena::default();
        let mut refs = Vec::with_capacity(elems);
        for i in 0..elems {
            refs.push(arena.alloc(T::from_usize(i)));
        }

        for i in 0..elems {
            assert_eq!(i, refs[i].as_usize());
            *refs[i] = T::from_usize(0);
        }

        for i in 0..elems {
            assert_eq!(*refs[i], T::from_usize(0));
        }
    }

    fn suite<T: TestType>() {
        test_simple::<T>();
        test_many_chunks::<T>();
    }

    fn blob_suite<T: TestType + Copy>() {
        test_blob_simple::<T>();
        test_blob_slice::<T>();
        test_blob_many_chunks::<T>();
    }

    impl TestType for i32 {
        fn as_usize(&self) -> usize {
            *self as usize
        }

        fn from_usize(i: usize) -> Self {
            i as i32
        }
    }

    #[test]
    fn primitives() {
        suite::<i32>();
        blob_suite::<i32>();
    }

    #[derive(Debug, PartialEq)]
    struct NeedDrop(Box<usize>);

    impl TestType for NeedDrop {
        fn as_usize(&self) -> usize {
            *self.0
        }

        fn from_usize(i: usize) -> Self {
            Self(Box::new(i))
        }
    }

    #[test]
    fn needs_drop() {
        suite::<NeedDrop>();
    }
}
