use super::ty::Ty;

#[derive(Debug)]
pub enum Lit<'a> {
    Int(i64),
    Str(&'a str),
}

impl Lit<'_> {
    pub fn is_int(&self) -> bool {
        matches!(self, Self::Int(_))
    }

    pub fn satisfies(&self, ty: Ty) -> bool {
        match self {
            Self::Int(_) => ty.is_int(),
            _ => todo!()
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LitId(usize);

#[derive(Debug, Default)]
pub struct LitStore<'a> {
    lits: Vec<Lit<'a>>,
}

impl<'a> LitStore<'a> {
    pub fn store(&mut self, lit: Lit<'a>) -> LitId {
        let idx = self.lits.len();
        self.lits.push(lit);
        LitId(idx)
    }

    #[track_caller]
    pub fn lit(&self, id: LitId) -> &Lit<'a> {
        self.lits.get(id.0).expect("invalid block id")
    }
}
