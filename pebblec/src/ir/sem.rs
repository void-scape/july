use super::*;
use std::ops::Deref;

pub fn sem_analysis_pre_typing<'a>(ctx: &Ctx<'a>) -> Result<(), Diag> {
    let mut ctx = SemCtx::new(ctx);

    ctx.sem_try(entry);
    ctx.sem_func(end_is_return);

    if ctx.diags.is_empty() {
        Ok(())
    } else {
        Err(Diag::bundle(std::mem::take(&mut ctx.diags)))
    }
}

pub fn sem_analysis(_ctx: &Ctx, _key: &TypeKey) -> Result<(), Diag> {
    Ok(())
}

struct SemCtx<'a> {
    ctx: &'a Ctx<'a>,
    //key: &'a TypeKey,
    diags: Vec<Diag>,
}

impl<'a> SemCtx<'a> {
    pub fn new(ctx: &'a Ctx<'a>) -> Self {
        Self {
            diags: Vec::new(),
            ctx,
        }
    }

    pub fn sem_try(&mut self, f: impl Fn(&mut SemCtx) -> Result<(), Diag>) {
        if let Err(diag) = f(self) {
            self.diags.push(diag);
        }
    }

    pub fn sem_func(&mut self, f: impl Fn(&SemCtx, &Func) -> Result<(), Diag>) {
        let mut errs = Vec::new();
        for func in self.funcs.iter() {
            if !func.is_intrinsic() {
                if let Err(diag) = f(self, func) {
                    errs.push(diag);
                }
            }
        }
        self.diags.extend(errs);
    }
}

impl<'a> Deref for SemCtx<'a> {
    type Target = Ctx<'a>;

    fn deref(&self) -> &Self::Target {
        self.ctx
    }
}

fn entry(ctx: &mut SemCtx) -> Result<(), Diag> {
    if let Some(func) = ctx
        .funcs
        .iter()
        .find(|f| ctx.expect_ident(f.sig.ident) == "main")
    {
        if func.sig.params.len() > 0 {
            Err(ctx.report_error(func.sig.span, "`main` cannot have any parameters (sorry)"))
        } else if func.sig.ty != Ty::I32 && !func.sig.ty.is_unit() {
            Err(ctx.report_error(func.sig.span, "`main` must return `i32` or `()`"))
        } else {
            Ok(())
        }
    } else {
        let help = "consider adding `main: () {}`";
        let error = "could not find entry point `main`";

        let buf = ctx.source_map.buffers().next().unwrap();
        if buf.len() == 0 {
            Err(ctx.report_error(
                Span::from_range(0..0).with_source(buf.source_id() as u32),
                format!("{}: {}", error, help),
            ))
        } else {
            let span = ctx.span(buf.last().unwrap());
            Err(ctx.report_error(span, error).msg(Msg::help(span, help)))
        }
    }
}

fn end_is_return(ctx: &SemCtx, func: &Func) -> Result<(), Diag> {
    if func.sig.ty == Ty::UNIT && func.block.end.is_some_and(|b| !b.is_unit(ctx)) {
        Err(ctx
            .report_error(
                func.block.end.as_ref().unwrap().span(),
                "invalid return type: expected `()`",
            )
            .msg(Msg::help(func.sig.span, "function has no return type")))
    } else if func.sig.ty != Ty::UNIT && func.block.end.is_none() {
        if let Some(stmt) = func.block.stmts.last() {
            match stmt {
                Stmt::Semi(semi) => match semi {
                    SemiStmt::Ret(_) => {
                        return Ok(());
                    }
                    _ => {}
                },
                _ => {}
            }

            Err(ctx
                .report_error(
                    stmt.span(),
                    format!(
                        "invalid return type: expected `{}`, got `()`",
                        func.sig.ty.to_string(ctx)
                    ),
                )
                .msg(Msg::help(func.sig.span, "inferred from signature")))
        } else {
            Err(ctx.report_error(
                func.block.span,
                format!(
                    "invalid return type: expected `{}`",
                    func.sig.ty.to_string(ctx)
                ),
            ))
        }
    } else {
        Ok(())
    }
}
