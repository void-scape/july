use crate::diagnostic::{self, Diag, Msg};
use crate::ir::ctx::Ctx;
use crate::ir::ident::Ident;
use crate::ir::lit::Lit;
use crate::lex::buffer::{Span, TokenQuery};
use crate::lex::buffer::{TokenBuffer, TokenId};
use crate::parse::rules::prelude as rules;
use crate::parse::Item;
use enom::{Enum, EnumDef, Variant};
use ident::IdentId;
use sig::Param;
use sig::Sig;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use strukt::{Field, FieldDef, Struct, StructDef};
use ty::store::TyId;
use ty::TypeKey;

use self::resolve::resolve_types;
use self::sem::sem_analysis;

pub mod ctx;
pub mod enom;
pub mod ident;
pub mod lit;
pub mod mem;
pub mod resolve;
pub mod sem;
pub mod sig;
pub mod strukt;
pub mod ty;

pub const SYM_DEF: &str = "undefined symbol";

pub fn lower<'a>(tokens: &'a TokenBuffer<'a>, items: &'a [Item]) -> Result<(Ctx<'a>, TypeKey), ()> {
    let mut ctx = Ctx::new(tokens);
    ctx.tys.register_builtins(&mut ctx.idents);

    let structs = items
        .iter()
        .filter_map(|i| match i {
            Item::Struct(strukt) => Some(strukt),
            _ => None,
        })
        .map(|s| (ctx.store_ident(s.name).id, s))
        .collect();
    if let Err(e) = add_structs(&mut ctx, &structs) {
        diagnostic::report(e);
        return Err(());
    }
    ctx.build_type_layouts();

    let sigs = lower_set(
        items
            .iter()
            .filter_map(|i| match i {
                Item::Func(func) => Some(func),
                _ => None,
            })
            .map(|f| func_sig(&mut ctx, f)),
    )?;
    if let Err(e) = ctx.store_sigs(sigs) {
        diagnostic::report(e);
        return Err(());
    }

    let funcs = lower_set(
        items
            .iter()
            .filter_map(|i| match i {
                Item::Func(func) => Some(func),
                _ => None,
            })
            .map(|f| func(&mut ctx, f)),
    )?;
    ctx.store_funcs(funcs);

    let key = match resolve_types(&mut ctx) {
        Ok(key) => key,
        Err(diags) => {
            for diag in diags.into_iter() {
                diagnostic::report(diag);
            }
            return Err(());
        }
    };

    sem_analysis(&ctx, &key).map(|_| (ctx, key))
}

fn add_structs<'a>(
    ctx: &mut Ctx<'a>,
    structs: &HashMap<IdentId, &'a rules::Struct>,
) -> Result<(), Diag<'a>> {
    let mut defined = HashSet::with_capacity(structs.len());
    let mut processing = Vec::with_capacity(structs.len());

    for strukt in structs.values() {
        add_structs_recur(ctx, &structs, &mut defined, &mut processing, strukt)?;
    }
    Ok(())
}

struct StructInfo<'a> {
    id: IdentId,
    strukt: &'a rules::Struct,
}

fn add_structs_recur<'a>(
    ctx: &mut Ctx<'a>,
    structs: &HashMap<IdentId, &'a rules::Struct>,
    defined: &mut HashSet<IdentId>,
    processing: &mut Vec<StructInfo<'a>>,
    rules_strukt: &'a rules::Struct,
) -> Result<(), Diag<'a>> {
    let name = ctx.store_ident(rules_strukt.name).id;
    if defined.contains(&name) {
        return Ok(());
    }

    if processing.iter().any(|info| info.id == name) {
        return Err(report_cycle(ctx, processing, rules_strukt));
    }

    processing.push(StructInfo {
        id: name,
        strukt: rules_strukt,
    });

    for field in rules_strukt.fields.iter() {
        let ty = ctx.store_ident(field.ty).id;
        if !ctx.tys.builtin(ctx.ident(field.ty)) {
            add_structs_recur(ctx, structs, defined, processing, structs.get(&ty).unwrap())?;
        }
    }

    processing.pop();
    defined.insert(name);

    let strukt = strukt(ctx, rules_strukt)?;
    ctx.tys.store_struct(strukt);
    Ok(())
}

fn report_cycle<'a>(
    ctx: &mut Ctx<'a>,
    processing: &[StructInfo<'a>],
    current: &'a rules::Struct,
) -> Diag<'a> {
    let mut title = String::from("recursive types without indirection: ");
    let mut msgs = Vec::new();

    let cycle_start = processing
        .iter()
        .position(|info| info.id == ctx.store_ident(current.name).id)
        .unwrap();

    let current = StructInfo {
        id: ctx.store_ident(current.name).id,
        strukt: current,
    };
    let cycle_members = processing[cycle_start..]
        .iter()
        .chain(std::iter::once(&current));

    let cycle_with_wrap = cycle_members
        .clone()
        .chain(std::iter::once(&processing[cycle_start]));
    for (i, (curr, next)) in cycle_with_wrap
        .clone()
        .zip(cycle_with_wrap.skip(1))
        .enumerate()
    {
        if i > 0 {
            title.push_str(&format!("`{}`", ctx.ident(curr.strukt.name)));
            title.push_str(", ");
        }

        if let Some(field) = curr
            .strukt
            .fields
            .iter()
            .find(|f| ctx.store_ident(f.ty).id == next.id)
        {
            msgs.push(Msg::error(ctx.span(curr.strukt.name), ""));
            msgs.push(Msg::note(field.span, ""));
        }
    }

    title.truncate(title.len() - 2);
    ctx.errors(title, msgs)
}

fn lower_set<'a, O>(items: impl Iterator<Item = Result<O, Diag<'a>>>) -> Result<Vec<O>, ()> {
    let mut error = false;
    let mut set = Vec::new();
    for item in items {
        match item {
            Ok(item) => {
                set.push(item);
            }
            Err(diag) => {
                diagnostic::report(diag);
                error = true;
            }
        }
    }
    (!error).then_some(set).ok_or(())
}

fn enom<'a>(ctx: &mut Ctx<'a>, enom: &rules::Enum) -> Result<Enum, Diag<'a>> {
    let mut variant_names = Vec::with_capacity(enom.variants.len());

    for field in enom.variants.iter() {
        if variant_names.contains(&ctx.ident(field.name)) {
            return Err(ctx.errors(
                "failed to parse enum",
                [
                    Msg::error(ctx.span(field.name), "variant already declared"),
                    Msg::note(
                        ctx.span(enom.name),
                        format!("while parsing `{}`", ctx.ident(enom.name)),
                    ),
                ],
            ));
        }
        variant_names.push(ctx.ident(field.name));
    }

    Ok(Enum {
        span: enom.span,
        name: ctx.store_ident(enom.name),
        variants: enom.variants.iter().map(|f| variant(ctx, f)).collect(),
    })
}

fn variant<'a>(ctx: &mut Ctx<'a>, variant: &rules::Variant) -> Variant {
    Variant {
        span: variant.span,
        name: ctx.store_ident(variant.name),
        //ty: ctx
        //    .ty(field.ty)
        //    .map(|t| Ty::Ty(t))
        //    .unwrap_or_else(|| Ty::Struct(ctx.store_ident(field.ty).id)),
    }
}

fn strukt<'a>(ctx: &mut Ctx<'a>, strukt: &rules::Struct) -> Result<Struct, Diag<'a>> {
    let mut field_names = Vec::with_capacity(strukt.fields.len());

    for field in strukt.fields.iter() {
        if field_names.contains(&ctx.ident(field.name)) {
            return Err(ctx.errors(
                "failed to parse struct",
                [
                    Msg::error(ctx.span(field.name), "field already declared"),
                    Msg::note(
                        ctx.span(strukt.name),
                        format!("while parsing `{}`", ctx.ident(strukt.name)),
                    ),
                ],
            ));
        }
        field_names.push(ctx.ident(field.name));
    }

    Ok(Struct {
        span: strukt.span,
        name: ctx.store_ident(strukt.name),
        fields: strukt.fields.iter().map(|f| field(ctx, f)).collect(),
    })
}

fn field<'a>(ctx: &mut Ctx<'a>, field: &rules::Field) -> Field {
    let name = ctx.store_ident(field.ty).id;

    Field {
        span: field.span,
        name: ctx.store_ident(field.name),
        ty: ctx
            .tys
            .ty_id(name)
            .unwrap_or_else(|| panic!("invalid field type: {}", ctx.expect_ident(name))),
    }
}

#[derive(Debug, Clone)]
pub struct Func {
    pub name_span: Span,
    pub sig: Sig,
    pub block: Block,
}

impl Func {
    pub fn hash(&self) -> FuncHash {
        self.sig.hash()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FuncHash(pub u64);

fn func_sig<'a>(ctx: &mut Ctx<'a>, func: &rules::Func) -> Result<Sig, Diag<'a>> {
    Ok(Sig {
        span: func.span,
        ident: ctx.store_ident(func.name).id,
        params: params(ctx, func)?,
        ty: if let Some(ty) = func.ty {
            let ident = ctx.store_ident(ty).id;
            match ctx.tys.ty_id(ident) {
                Some(ty) => ty,
                None => {
                    return Err(
                        ctx.report_error(ty, format!("expected type, got `{}`", ctx.ident(ty)))
                    )
                }
            }
        } else {
            ctx.tys.unit()
        },
    })
}

fn params<'a>(ctx: &mut Ctx<'a>, func: &rules::Func) -> Result<Vec<Param>, Diag<'a>> {
    let mut params = Vec::with_capacity(func.params.len());
    for p in func.params.iter() {
        params.push(Param {
            span: Span::from_spans(ctx.span(p.name), ctx.span(p.ty)),
            ty_binding: Span::from_spans(ctx.span(p.colon), ctx.span(p.ty)),
            ident: ctx.store_ident(p.name),
            ty: {
                let ident = ctx.store_ident(p.ty).id;
                match ctx.tys.ty_id(ident) {
                    Some(ty) => ty,
                    None => {
                        return Err(ctx.report_error(
                            p.ty,
                            format!("expected type, got `{}`", ctx.expect_ident(ident)),
                        ))
                    }
                }
            },
        });
    }

    Ok(params)
}

fn func<'a>(ctx: &mut Ctx<'a>, func: &rules::Func) -> Result<Func, Diag<'a>> {
    Ok(Func {
        name_span: ctx.span(func.name),
        block: block(ctx, &func.block)?,
        //.map_err(|err| err.msg(Msg::note(func.block.span, "while parsing this function")))?,
        sig: func_sig(ctx, func)?,
    })
}

#[derive(Debug, Clone)]
pub struct Block {
    pub span: Span,
    pub stmts: Vec<Stmt>,
    pub end: Option<OpenStmt>,
}

fn block<'a>(ctx: &mut Ctx<'a>, block: &rules::Block) -> Result<Block, Diag<'a>> {
    let mut stmts = block
        .stmts
        .iter()
        .map(|st| stmt(ctx, st))
        .collect::<Result<Vec<_>, _>>()?;

    let end = if let Some(last) = stmts.last() {
        match last {
            Stmt::Open(end) => {
                let end = end.clone();
                stmts.pop();
                Some(end)
            }
            _ => None,
        }
    } else {
        None
    };

    Ok(Block {
        span: block.span,
        stmts,
        end,
    })
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Semi(SemiStmt),
    Open(OpenStmt),
}

impl Stmt {
    pub fn span(&self) -> Span {
        match self {
            Stmt::Semi(semi) => semi.span(),
            Stmt::Open(open) => open.span(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum OpenStmt {
    Ident(Ident),
    Lit(Lit),
    Bin(BinOp),
    Call(Call),
    Struct(StructDef),
}

impl OpenStmt {
    pub fn span(&self) -> Span {
        match self {
            Self::Ident(ident) => ident.span,
            Self::Lit(lit) => lit.span,
            Self::Bin(bin) => bin.span,
            Self::Call(call) => call.span,
            Self::Struct(def) => def.span,
        }
    }
}

#[derive(Debug, Clone)]
pub enum SemiStmt {
    Let(Let),
    Assign(Assign),
    Ret(Return),
    Bin(BinOp),
    Call(Call),
}

impl SemiStmt {
    pub fn span(&self) -> Span {
        match self {
            Self::Let(let_) => let_.span,
            Self::Assign(assign) => assign.span,
            Self::Ret(ret) => ret.span,
            Self::Bin(bin) => bin.span,
            Self::Call(call) => call.span,
        }
    }
}

fn stmt<'a>(ctx: &mut Ctx<'a>, stmt: &rules::Stmt) -> Result<Stmt, Diag<'a>> {
    Ok(match stmt {
        rules::Stmt::Let { name, ty, assign } => {
            let ty = if let Some(ty) = ty {
                let ident = ctx.store_ident(*ty).id;
                match ctx.tys.ty_id(ident) {
                    Some(ty_id) => Some((ctx.span(*ty), ty_id)),
                    None => {
                        return Err(ctx.report_error(
                            ty,
                            format!("`{}` is not a type, expected a type", ctx.ident(*ty)),
                        ));
                    }
                }
            } else {
                None
            };

            Stmt::Semi(SemiStmt::Let(Let {
                span: ctx.span(*name),
                lhs: let_target(ctx, &rules::Expr::Ident(*name)),
                rhs: let_expr(ctx, assign)?,
                ty,
            }))
        }
        rules::Stmt::Semi(expr) => match expr {
            rules::Expr::Assign(assign) => Stmt::Semi(SemiStmt::Assign(Assign {
                span: assign.assign_span,
                kind: assign.kind,
                lhs: assign_target(ctx, &assign.lhs)?,
                rhs: let_expr(ctx, &assign.rhs)?,
            })),
            rules::Expr::Ret(span, expr) => Stmt::Semi(SemiStmt::Ret(Return {
                span: *span,
                expr: match expr {
                    Some(expr) => Some(open_stmt(ctx, &expr)?),
                    None => None,
                },
            })),
            rules::Expr::Bin(_, _, _) => Stmt::Semi(SemiStmt::Bin(bin_op(ctx, expr)?)),
            rules::Expr::Call { span, func, args } => {
                Stmt::Semi(SemiStmt::Call(call(ctx, *span, *func, args)?))
            }
            expr => todo!("{expr:#?}"),
        },
        rules::Stmt::Open(expr) => Stmt::Open(open_stmt(ctx, &expr)?),
    })
}

fn open_stmt<'a>(ctx: &mut Ctx<'a>, expr: &rules::Expr) -> Result<OpenStmt, Diag<'a>> {
    Ok(match expr {
        rules::Expr::Ident(ident) => OpenStmt::Ident(ctx.store_ident(*ident)),
        rules::Expr::Lit(lit) => OpenStmt::Lit(ctx.store_int(*lit)),
        rules::Expr::Bin(_, _, _) => OpenStmt::Bin(bin_op(ctx, expr)?),
        rules::Expr::Call { span, func, args } => OpenStmt::Call(call(ctx, *span, *func, args)?),
        rules::Expr::StructDef(def) => OpenStmt::Struct(struct_def(ctx, def)?),
        _ => todo!(),
    })
}

#[derive(Debug, Clone)]
pub struct Let {
    pub span: Span,
    pub ty: Option<(Span, TyId)>,
    pub lhs: LetTarget,
    pub rhs: Expr,
}

#[derive(Debug, Clone, Copy)]
pub enum LetTarget {
    Ident(Ident),
}

#[derive(Debug, Clone)]
pub enum Expr {
    Ident(Ident),
    Lit(Lit),
    Bin(Box<BinOp>),
    Struct(StructDef),
    Enum(EnumDef),
    Call(Call),
}

impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Self::Ident(ident) => ident.span,
            Self::Lit(lit) => lit.span,
            Self::Call(call) => call.span,
            Self::Bin(bin) => bin.span,
            Self::Struct(def) => def.span,
            Self::Enum(enom) => enom.span,
        }
    }

    pub fn is_integral(&self, ctx: &Ctx) -> Option<bool> {
        match self {
            Self::Ident(_) => None,
            Self::Lit(lit) => Some(ctx.expect_lit(lit.kind).is_int()),
            Self::Call(call) => Some(ctx.tys.ty(call.sig.ty).is_int()),
            Self::Bin(bin) => bin.is_integral(ctx),
            Self::Struct(_) | Self::Enum(_) => Some(false),
        }
    }
}

fn let_target<'a>(ctx: &mut Ctx<'a>, expr: &rules::Expr) -> LetTarget {
    match expr {
        rules::Expr::Ident(ident) => LetTarget::Ident(ctx.store_ident(*ident)),
        _ => todo!(),
    }
}

fn let_expr<'a>(ctx: &mut Ctx<'a>, expr: &rules::Expr) -> Result<Expr, Diag<'a>> {
    Ok(match expr {
        rules::Expr::Ident(ident) => Expr::Ident(ctx.store_ident(*ident)),
        rules::Expr::Lit(lit) => Expr::Lit(ctx.store_int(*lit)),
        rules::Expr::Bin(_, _, _) => Expr::Bin(Box::new(bin_op(ctx, expr)?)),
        rules::Expr::StructDef(def) => Expr::Struct(struct_def(ctx, def)?),
        rules::Expr::EnumDef(def) => Expr::Enum(enum_def(ctx, def)?),
        rules::Expr::Call { span, func, args } => Expr::Call(call(ctx, *span, *func, args)?),
        _ => todo!(),
    })
}

fn enum_def<'a>(ctx: &mut Ctx<'a>, def: &rules::EnumDef) -> Result<EnumDef, Diag<'a>> {
    Ok(EnumDef {
        span: def.span,
        name: ctx.store_ident(def.name),
        variant: variant(ctx, &def.variant),
    })
}

fn struct_def<'a>(ctx: &mut Ctx<'a>, def: &rules::StructDef) -> Result<StructDef, Diag<'a>> {
    let id = ctx.store_ident(def.name).id;

    Ok(StructDef {
        span: def.span,
        id: ctx.expect_struct_id(id),
        fields: def
            .fields
            .iter()
            .map(|f| field_def(ctx, f))
            .collect::<Result<_, _>>()?,
    })
}

fn field_def<'a>(ctx: &mut Ctx<'a>, def: &rules::FieldDef) -> Result<FieldDef, Diag<'a>> {
    Ok(FieldDef {
        span: def.span,
        name: ctx.store_ident(def.name),
        expr: let_expr(ctx, &def.expr)?,
    })
}

#[derive(Debug, Clone)]
pub struct BinOp {
    pub span: Span,
    pub kind: BinOpKind,
    pub lhs: Expr,
    pub rhs: Expr,
}

impl BinOp {
    /// Contains all integral components.
    pub fn is_integral(&self, ctx: &Ctx) -> Option<bool> {
        self.lhs
            .is_integral(ctx)
            .map(|i| self.rhs.is_integral(ctx).map(|o| i && o))?
    }
}

/// Descend into `bin` and recover the full accessor path.
pub fn descend_bin_op_field(ctx: &Ctx, bin: &BinOp, accesses: &mut Vec<Ident>) {
    if bin.kind == BinOpKind::Field {
        match bin.lhs {
            Expr::Ident(ident) => {
                if let Expr::Bin(bin) = &bin.rhs {
                    accesses.push(ident);
                    descend_bin_op_field(ctx, bin, accesses);
                } else {
                    let Expr::Ident(other) = bin.rhs else {
                        unreachable!()
                    };

                    accesses.push(other);
                    accesses.push(ident);
                }
            }
            _ => {}
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOpKind {
    Add,
    Sub,
    Mul,
    Field,
}

impl BinOpKind {
    pub fn is_primitive(&self) -> bool {
        !self.is_field()
    }

    pub fn is_field(&self) -> bool {
        matches!(self, Self::Field)
    }
}

fn bin_op<'a>(ctx: &mut Ctx<'a>, expr: &rules::Expr) -> Result<BinOp, Diag<'a>> {
    Ok(match expr {
        rules::Expr::Bin(op, lhs, rhs) => BinOp {
            span: op.span,
            kind: op.kind,
            lhs: pexpr(ctx, lhs)?,
            rhs: pexpr(ctx, rhs)?,
        },
        _ => todo!(),
    })
}

fn pexpr<'a>(ctx: &mut Ctx<'a>, expr: &rules::Expr) -> Result<Expr, Diag<'a>> {
    Ok(match expr {
        rules::Expr::Ident(ident) => Expr::Ident(ctx.store_ident(*ident)),
        rules::Expr::Lit(lit) => Expr::Lit(ctx.store_int(*lit)),
        rules::Expr::Bin(_, _, _) => Expr::Bin(Box::new(bin_op(ctx, expr)?)),
        rules::Expr::Call { span, func, args } => Expr::Call(call(ctx, *span, *func, args)?),
        rules::Expr::StructDef(def) => Expr::Struct(struct_def(ctx, def)?),
        _ => todo!(),
    })
}

#[derive(Debug, Clone)]
pub struct Assign {
    pub span: Span,
    pub kind: AssignKind,
    pub lhs: AssignTarget,
    pub rhs: Expr,
}

// TODO: more assignment kinds
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssignKind {
    Equals,
    Add,
}

#[derive(Debug, Clone)]
pub enum AssignTarget {
    Ident(Ident),
    Field(BinOp),
}

fn assign_target<'a>(ctx: &mut Ctx<'a>, expr: &rules::Expr) -> Result<AssignTarget, Diag<'a>> {
    Ok(match expr {
        rules::Expr::Ident(ident) => AssignTarget::Ident(ctx.store_ident(*ident)),
        rules::Expr::Bin(bin, _, _) => {
            if bin.kind == BinOpKind::Field {
                AssignTarget::Field(bin_op(ctx, expr)?)
            } else {
                todo!()
            }
        }
        _ => todo!(),
    })
}

#[derive(Debug, Clone)]
pub struct Return {
    pub span: Span,
    pub expr: Option<OpenStmt>,
}

#[derive(Debug, Clone)]
pub struct Call {
    pub span: Span,
    pub sig: Sig,
    pub args: Args,
}

#[derive(Debug, Default, Clone)]
pub struct Args {
    pub args: Vec<Expr>,
}

impl Args {
    pub fn is_empty(&self) -> bool {
        self.args.is_empty()
    }
}

fn args<'a>(ctx: &mut Ctx<'a>, args: &[rules::Expr]) -> Result<Args, Diag<'a>> {
    Ok(Args {
        args: args
            .iter()
            .map(|arg| pexpr(ctx, arg))
            .collect::<Result<_, _>>()?,
    })
}

fn call<'a>(
    ctx: &mut Ctx<'a>,
    span: Span,
    name: TokenId,
    call_args: &[rules::Expr],
) -> Result<Call, Diag<'a>> {
    let id = ctx.store_ident(name).id;
    Ok(Call {
        sig: ctx
            .get_sig(id)
            .ok_or_else(|| ctx.report_error(name, "function is not defined"))?
            .clone(),
        args: args(ctx, call_args)?,
        span,
    })
}
