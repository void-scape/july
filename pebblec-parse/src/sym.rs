use crate::lex::buffer::Span;
use indexmap::IndexSet;
use pebblec_arena::BlobArena;
use std::cell::RefCell;
use std::sync::LazyLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Symbol(usize);

impl Symbol {
    pub fn intern(string: &str) -> Self {
        with_sym_arena_mut(|sym_arena| {
            if let Some(index) = sym_arena.sym_index(string) {
                return Self(index);
            }

            let string: &'static str = sym_arena.arena.alloc_str(string);
            let (idx, new) = sym_arena.strings.insert_full(string);
            debug_assert!(new);

            Symbol(idx)
        })
    }

    pub fn as_str(&self) -> &str {
        with_sym_arena(|sym_arena| sym_arena.strings.get_index(self.0).copied().unwrap())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ident {
    pub sym: Symbol,
    pub span: Span,
}

impl Ident {
    pub fn as_str(&self) -> &str {
        self.sym.as_str()
    }
}

#[derive(Debug, Default)]
struct SymArena {
    arena: BlobArena,
    strings: IndexSet<&'static str>,
}

impl SymArena {
    pub fn sym_index(&self, sym: &str) -> Option<usize> {
        self.strings.get_index_of(sym)
    }
}

thread_local! {
    static SYM_ARENA: LazyLock<RefCell<SymArena>> = LazyLock::new(|| RefCell::new(SymArena::default()));
}

fn with_sym_arena<R>(f: impl FnOnce(&SymArena) -> R) -> R {
    SYM_ARENA.with(|sym_arena| f(&sym_arena.borrow()))
}

fn with_sym_arena_mut<R>(f: impl FnOnce(&mut SymArena) -> R) -> R {
    SYM_ARENA.with(|sym_arena| f(&mut sym_arena.borrow_mut()))
}
