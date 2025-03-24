use super::*;
use pebblec_parse::lex::buffer::Buffer;
use std::ops::Deref;

pub fn sem_analysis_pre_typing<'a>(ctx: &Ctx<'a>) -> Result<(), Diag<'a>> {
    let mut ctx = SemCtx::new(ctx);

    ctx.sem_try(entry);
    ctx.sem_func(end_is_return);

    if ctx.diags.is_empty() {
        Ok(())
    } else {
        Err(Diag::bundle(std::mem::take(&mut ctx.diags)))
    }
}

pub fn sem_analysis<'a>(_ctx: &Ctx<'a>, _key: &TypeKey) -> Result<(), Diag<'a>> {
    Ok(())
}

struct SemCtx<'a, 'b> {
    ctx: &'a Ctx<'b>,
    //key: &'a TypeKey,
    diags: Vec<Diag<'b>>,
}

impl<'a, 'b> SemCtx<'a, 'b> {
    pub fn new(
        ctx: &'a Ctx<'b>,
        //key: &'a TypeKey
    ) -> Self {
        Self {
            diags: Vec::new(),
            ctx,
            //key,
        }
    }

    pub fn sem_try(&mut self, f: impl Fn(&mut SemCtx<'a, 'b>) -> Result<(), Diag<'b>>) {
        if let Err(diag) = f(self) {
            self.diags.push(diag);
        }
    }

    pub fn sem_func(&mut self, f: impl Fn(&SemCtx<'a, 'b>, &Func) -> Result<(), Diag<'b>>) {
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

impl<'a, 'b> Deref for SemCtx<'a, 'b> {
    type Target = Ctx<'b>;

    fn deref(&self) -> &Self::Target {
        self.ctx
    }
}

fn entry<'a, 'b>(ctx: &mut SemCtx<'a, 'b>) -> Result<(), Diag<'b>> {
    if let Some(func) = ctx
        .funcs
        .iter()
        .find(|f| ctx.expect_ident(f.sig.ident) == "main")
    {
        if func.sig.params.len() > 0 {
            Err(ctx.report_error(func.sig.span, "`main` cannot have any parameters"))
        } else if func.sig.ty != ctx.tys.get_ty_id(&Ty::Int(IntTy::new_32(Sign::I))).unwrap() {
            Err(ctx.report_error(func.sig.span, "`main` must return `i32`"))
        } else {
            Ok(())
        }
    } else {
        let help = "consider adding a `main: () [-> i32]` function";
        let error = "could not find entry point `main`";

        todo!();
        //if ctx.token_buffer().len() == 0 {
        //    Err(ctx
        //        .report_error(ctx.token_buffer(), error)
        //        .wrap(ctx.report_help(Span::empty(), help)))
        //} else {
        //    Err(ctx
        //        .report_error(Span::empty(), error)
        //        .wrap(ctx.report_help(ctx.token_buffer().last().unwrap(), help)))
        //}
    }
}

fn end_is_return<'a, 'b>(ctx: &SemCtx<'a, 'b>, func: &Func) -> Result<(), Diag<'b>> {
    if func.sig.ty == TyId::UNIT && func.block.end.is_some_and(|b| !b.is_unit(ctx)) {
        Err(ctx
            .report_error(
                func.block.end.as_ref().unwrap().span(),
                "invalid return type: expected `()`",
            )
            .msg(Msg::help(func.sig.span, "function has no return type")))
    } else if func.sig.ty != TyId::UNIT && func.block.end.is_none() {
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
                        ctx.ty_str(func.sig.ty)
                    ),
                )
                .msg(Msg::help(func.sig.span, "inferred from signature")))
        } else {
            Err(ctx.report_error(
                func.block.span,
                format!(
                    "invalid return type: expected `{}`",
                    ctx.ty_str(func.sig.ty)
                ),
            ))
        }
    } else {
        Ok(())
    }
}
