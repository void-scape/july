use crate::arena::BlobArena;
use crate::ir::mem::Layout;
use crate::ir::ty::store::TyId;

#[derive(Default)]
pub struct Bss {
    //entries: Vec<BssEntry>,
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

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct BssEntry {
    ty: TyId,
    layout: Layout,
    data: *const u8,
}

impl BssEntry {
    pub fn str_lit(data: *const u8) -> Self {
        Self {
            ty: TyId::STR_LIT,
            layout: Layout::FAT_PTR,
            data,
        }
    }

    pub fn addr(&self) -> usize {
        self.data.addr()
    }
}
