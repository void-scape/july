#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Layout {
    pub size: usize,
    pub alignment: usize,
}

impl Layout {
    pub const PTR: Self = Self::splat(8);
    pub const FAT_PTR: Self = Self::new(16, 8);

    pub const fn new(size: usize, alignment: usize) -> Self {
        Self { size, alignment }
    }

    pub const fn splat(n: usize) -> Self {
        Self::new(n, n)
    }

    pub fn align_shift(&self) -> u8 {
        match self.alignment {
            1 => 0,
            2 => 1,
            4 => 2,
            8 => 3,
            _ => panic!("invalid alignment"),
        }
    }
}
