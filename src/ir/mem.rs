#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    pub size: usize,
    pub alignment: usize,
}

impl Layout {
    pub fn new(size: usize, alignment: usize) -> Self {
        Self { size, alignment }
    }

    pub fn splat(n: usize) -> Self {
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
