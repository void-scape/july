use super::*;
use std::ops::Deref;

pub fn sem_analysis_pre_typing<'a>(ctx: &Ctx<'a>) -> Result<(), Diag> {
    let mut ctx = SemCtx::new(ctx);

    ctx.sem_try(entry);
    //ctx.sem_func(end_is_return);

    if ctx.diags.is_empty() {
        Ok(())
    } else {
        Err(Diag::bundle(std::mem::take(&mut ctx.diags)))
    }
}

pub fn sem_analysis(_ctx: &Ctx, _key: &TypeKey) -> Result<(), Diag> {
    // TODO: end_is_return with full type resolution (method calls)
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

    //pub fn sem_func(&mut self, f: impl Fn(&SemCtx, &Func) -> Result<(), Diag>) {
    //    let mut errs = Vec::new();
    //    for func in self.funcs.iter() {
    //        if !func.is_intrinsic() {
    //            if let Err(diag) = f(self, func) {
    //                errs.push(diag);
    //            }
    //        }
    //    }
    //    self.diags.extend(errs);
    //}
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
            Err(ctx
                .report_error(span, error)
                .msg(Msg::help(&ctx.source_map, span, help)))
        }
    }
}
