use super::ctx::CtxFmt;
use super::*;
use crate::lex::buffer::Buffer;
use std::ops::Deref;

pub fn sem_analysis_pre_typing<'a>(ctx: &'a Ctx<'a>) -> Result<(), ()> {
    let mut ctx = SemCtx::new(ctx);

    ctx.sem_try(entry);
    ctx.sem_try(unique_funcs);
    ctx.sem_func(end_is_return);
    //ctx.sem_func(validate_loop_block);

    if ctx.diags.is_empty() {
        Ok(())
    } else {
        Err(diagnostic::report_set(ctx.diags.into_iter()))
    }
}

pub fn sem_analysis<'a>(_ctx: &'a Ctx<'a>, _key: &'a TypeKey) -> Result<(), ()> {
    Ok(())
}

struct SemCtx<'a> {
    ctx: &'a Ctx<'a>,
    //key: &'a TypeKey,
    diags: Vec<Diag<'a>>,
}

impl<'a> SemCtx<'a> {
    pub fn new(
        ctx: &'a Ctx<'a>,
        //key: &'a TypeKey
    ) -> Self {
        Self {
            diags: Vec::new(),
            ctx,
            //key,
        }
    }

    pub fn sem_try(&mut self, f: impl Fn(&mut SemCtx<'a>) -> Result<(), Diag<'a>>) {
        if let Err(diag) = f(self) {
            self.diags.push(diag);
        }
    }

    pub fn sem_func(&mut self, f: impl Fn(&SemCtx<'a>, &Func) -> Result<(), Diag<'a>>) {
        let mut errs = Vec::new();
        for func in self.funcs.iter() {
            if let Err(diag) = f(self, func) {
                errs.push(diag);
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

fn entry<'a>(ctx: &mut SemCtx<'a>) -> Result<(), Diag<'a>> {
    if !ctx
        .funcs
        .iter()
        .find(|f| ctx.expect_ident(f.sig.ident) == "main")
        .is_some()
    {
        let help = "consider adding a `main: () [-> i32]` function";
        let error = "could not find entry point `main`";

        if ctx.token_buffer().len() == 0 {
            Err(ctx
                .report_error(Span::empty(), error)
                .wrap(ctx.report_help(Span::empty(), help)))
        } else {
            Err(ctx
                .report_error(Span::empty(), error)
                .wrap(ctx.report_help(ctx.token_buffer().last().unwrap(), help)))
        }
    } else {
        Ok(())
    }
}

fn unique_funcs<'a>(ctx: &mut SemCtx<'a>) -> Result<(), Diag<'a>> {
    let mut set = HashSet::with_capacity(ctx.funcs.len());
    for f in ctx.funcs.iter() {
        if !set.insert(f.sig.ident) {
            let ident = f.sig.ident;
            let occurences = ctx
                .funcs
                .iter()
                .filter(|f| f.sig.ident == ident)
                .collect::<Vec<_>>();

            return Err(ctx.errors(
                format!("`{}` defined multiple times", ident.to_string(ctx)),
                occurences.iter().map(|f| Msg::error(f.name_span, "")),
            ));
        }
    }

    Ok(())
}

fn end_is_return<'a>(ctx: &SemCtx<'a>, func: &Func) -> Result<(), Diag<'a>> {
    if func.sig.ty == ctx.tys.unit()
        && func
            .block
            .end
            .is_some_and(|b| b.is_unit(ctx).is_some_and(|c| !c))
    {
        Err(ctx
            .report_error(
                func.block.end.as_ref().unwrap().span(),
                "invalid return type: expected `()`",
            )
            .msg(Msg::help(func.sig.span, "function has no return type")))
    } else if func.sig.ty != ctx.tys.unit() && func.block.end.is_none() {
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

//fn validate_loop_block<'a>(ctx: &SemCtx<'a>, func: &Func) -> Result<(), Diag<'a>> {
//    for stmt in func.block.stmts.iter() {
//        if let Stmt::Open(Expr::Loop(block)) = stmt {
//            if let Some(end) = block.end {
//                // TODO: ctx.expr_ty(end), or something
//                return Err(ctx.report_error(end.span(), "mismatched types: expected `()`"));
//            }
//        } else if let Stmt::Semi(SemiStmt::Ret(ret)) = stmt {
//            // TODO: should there be a place for kind of checking, or should it be done during type inferece?
//            match ret.expr {
//                None => {
//                    if !ctx.tys.is_unit(func.sig.ty) {
//                        return Err(ctx.mismatch(ret.span, func.sig.ty, TyId::UNIT));
//                    }
//                }
//                Some(_) => {
//                    if ctx.tys.is_unit(func.sig.ty) {
//                        // TODO: ctx.expr_ty(end), or something
//                        return Err(ctx.report_error(ret.span, "mismatched types: expected `()`"));
//                    }
//                }
//            }
//        }
//    }
//
//    Ok(())
//}
