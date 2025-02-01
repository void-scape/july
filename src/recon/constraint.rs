use crate::ir::prelude::Ty;

#[derive(Debug)]
pub enum Constraint {
    Arch(Arch),
    Abs(Ty),
}

impl Constraint {
    pub fn unify(constraints: Vec<Constraint>) -> Result<Ty, ()> {
        let mut archs = Vec::with_capacity(constraints.len());
        let mut abs = None;

        for c in constraints.into_iter() {
            match c {
                Constraint::Abs(ty) => abs = Some(ty),
                Constraint::Arch(a) => archs.push(a),
            }
        }

        let ty = abs.ok_or(())?;

        for arch in archs.iter() {
            if !arch.satisfied(ty) {
                return Err(());
            }
        }

        Ok(ty)
    }
}

#[derive(Debug)]
pub enum Arch {
    Int,
}

impl Arch {
    pub fn satisfied(&self, ty: Ty) -> bool {
        match self {
            Arch::Int => ty.is_int(),
        }
    }
}
