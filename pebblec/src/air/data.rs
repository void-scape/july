use crate::ir::mem::Layout;
use crate::ir::ty::Ty;
use pebblec_arena::BlobArena;

#[derive(Debug, Default)]
pub struct Bss {
    data: BlobArena,
}

impl Bss {
    pub fn str_lit(&mut self, str: &str) -> (BssEntry, usize) {
        let raw = String::from_utf8(str.as_bytes().to_vec()).unwrap();
        let fmt = raw.replace("\\n", "\n").replace("\\0", "\0");
        let data = self.data.alloc_str_ptr(&fmt);
        let len = fmt.len();
        (BssEntry::str_lit(data), len)
    }
}

#[derive(Debug, Clone, Hash)]
pub struct BssEntry {
    ty: Ty,
    layout: Layout,
    data: *const u8,
}

impl PartialEq for BssEntry {
    fn eq(&self, other: &Self) -> bool {
        self.ty == other.ty && self.layout == other.layout
    }
}

impl BssEntry {
    pub fn str_lit(data: *const u8) -> BssEntry {
        Self {
            ty: Ty::STR_LIT,
            layout: Layout::FAT_PTR,
            data,
        }
    }

    pub fn addr(&self) -> usize {
        self.data.addr()
    }
}
