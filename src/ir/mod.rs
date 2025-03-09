use self::lit::LitKind;
use self::sem::sem_analysis_pre_typing;
use self::sig::Linkage;
use self::ty::{FloatTy, IntTy, Sign, Ty};
use crate::air::ctx::AirCtx;
use crate::diagnostic::{self, Diag, Msg};
use crate::ir::ctx::Ctx;
use crate::ir::ident::Ident;
use crate::ir::lit::Lit;
use crate::lex::buffer::{Buffer, Span, TokenQuery};
use crate::lex::buffer::{TokenBuffer, TokenId};
use crate::lex::kind::TokenKind;
use crate::parse::rules::prelude::{self as rules, Attr};
use crate::parse::Item;
use enom::{Enum, EnumDef, Variant};
use ident::IdentId;
use resolve::resolve_types;
use sem::sem_analysis;
use sig::Param;
use sig::Sig;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use strukt::{Field, FieldDef, Struct, StructDef};
use ty::store::TyId;
use ty::TypeKey;

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

pub type ConstEvalOrder = Vec<IdentId>;

pub fn lower<'a>(
    tokens: &'a TokenBuffer<'a>,
    items: &'a [Item],
) -> Result<(Ctx<'a>, TypeKey, ConstEvalOrder), ()> {
    let mut ctx = Ctx::new(tokens);

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

    let extern_sigs = lower_set(
        items
            .iter()
            .filter_map(|i| match i {
                Item::Extern(func) => Some(func),
                _ => None,
            })
            .map(|f| extern_sig(&mut ctx, f)),
    )?;
    if let Err(e) = ctx.store_sigs(extern_sigs) {
        diagnostic::report(e);
        return Err(());
    }

    let consts = items
        .iter()
        .filter_map(|i| match i {
            Item::Const(konst) => Some(konst),
            _ => None,
        })
        .map(|s| (ctx.store_ident(s.name).id, s))
        .collect();
    let eval_order = match add_consts(&mut ctx, &consts) {
        Ok(order) => order,
        Err(e) => {
            diagnostic::report(e);
            return Err(());
        }
    };

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

    sem_analysis_pre_typing(&ctx)?;
    let key = match resolve_types(&mut ctx) {
        Ok(key) => key,
        Err(diags) => {
            for diag in diags.into_iter() {
                diagnostic::report(diag);
            }
            return Err(());
        }
    };
    sem_analysis(&ctx, &key).map(|_| (ctx, key, eval_order))
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
    let mut errors = Vec::new();

    let name = ctx.store_ident(rules_strukt.name).id;
    if defined.contains(&name) {
        return Ok(());
    }

    if processing.iter().any(|info| info.id == name) {
        return Err(report_struct_cycle(ctx, processing, rules_strukt));
    }

    processing.push(StructInfo {
        id: name,
        strukt: rules_strukt,
    });

    for field in rules_strukt.fields.iter() {
        match field.ty.peel_refs() {
            rules::PType::Simple(id) => {
                if !ctx.tys.is_builtin(ctx.ident(*id)) {
                    if let Some(strukt) = field.ty.retrieve_struct(ctx, structs) {
                        add_structs_recur(ctx, structs, defined, processing, strukt)?;
                    } else {
                        errors.push(
                            ctx.report_error(field.ty.span(ctx.token_buffer()), "undefined type"),
                        );
                    }
                }
            }
            rules::PType::Ref { .. } => unreachable!(),
        }
    }

    processing.pop();
    defined.insert(name);

    match strukt(ctx, rules_strukt) {
        Ok(strukt) => {
            ctx.tys.store_struct(strukt);
        }
        Err(e) => errors.push(e),
    }

    if !errors.is_empty() {
        Err(Diag::bundle(errors))
    } else {
        Ok(())
    }
}

// TODO: no cycle with indirection
fn report_struct_cycle<'a>(
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

        if let Some(field) = curr.strukt.fields.iter().find(|f| {
            ctx.store_ident(match f.ty {
                rules::PType::Simple(id) => id,
                rules::PType::Ref { .. } => unreachable!(),
            })
            .id == next.id
        }) {
            msgs.push(Msg::error(ctx.span(curr.strukt.name), ""));
            msgs.push(Msg::note(field.span, ""));
        }
    }

    title.truncate(title.len() - 2);
    ctx.errors(title, msgs)
}

fn add_consts<'a>(
    ctx: &mut Ctx<'a>,
    consts: &HashMap<IdentId, &'a rules::Const>,
) -> Result<Vec<IdentId>, Diag<'a>> {
    let mut eval_order = Vec::new();
    let mut defined = HashSet::with_capacity(consts.len());
    let mut processing = Vec::with_capacity(consts.len());

    for strukt in consts.values() {
        add_consts_recur(
            ctx,
            &consts,
            &mut defined,
            &mut processing,
            strukt,
            &mut eval_order,
        )?;
    }

    Ok(eval_order)
}

struct ConstInfo<'a> {
    id: IdentId,
    konst: &'a rules::Const,
}

fn add_consts_recur<'a>(
    ctx: &mut Ctx<'a>,
    consts: &HashMap<IdentId, &'a rules::Const>,
    defined: &mut HashSet<IdentId>,
    processing: &mut Vec<ConstInfo<'a>>,
    rules_const: &'a rules::Const,
    evaluation_order: &mut Vec<IdentId>,
) -> Result<(), Diag<'a>> {
    let name = ctx.store_ident(rules_const.name).id;
    if defined.contains(&name) {
        return Ok(());
    }

    if processing.iter().any(|info| info.id == name) {
        return Err(report_const_cycle(ctx, processing, rules_const));
    }

    processing.push(ConstInfo {
        id: name,
        konst: rules_const,
    });

    match rules_const.expr {
        rules::Expr::Lit(_) => {
            evaluation_order.push(name);
        }
        rules::Expr::Ident(other) => {
            let ident = ctx.store_ident(other);
            if let Some(other) = consts.get(&ident.id) {
                add_consts_recur(ctx, consts, defined, processing, other, evaluation_order)?;
                evaluation_order.push(name);
            } else {
                return Err(ctx.undeclared(ident));
            }
        }
        _ => unimplemented!(),
    }

    processing.pop();
    defined.insert(name);

    let konst = konst(ctx, rules_const)?;
    ctx.tys.store_const(ctx.intern(konst));
    Ok(())
}

fn report_const_cycle<'a>(
    ctx: &mut Ctx<'a>,
    processing: &[ConstInfo<'a>],
    current: &'a rules::Const,
) -> Diag<'a> {
    let mut title = String::from("recursive const definitions: ");
    let mut msgs = Vec::new();

    let cycle_start = processing
        .iter()
        .position(|info| info.id == ctx.store_ident(current.name).id)
        .unwrap();

    let current = ConstInfo {
        id: ctx.store_ident(current.name).id,
        konst: current,
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
            title.push_str(&format!("`{}`", ctx.ident(curr.konst.name)));
            title.push_str(", ");
        }

        match curr.konst.expr {
            rules::Expr::Ident(ident) => {
                if ctx.store_ident(ident).id == next.id {
                    msgs.push(Msg::error(ctx.span(curr.konst.name), ""));
                    msgs.push(Msg::note(ctx.span(ident), ""));
                }
            }
            rules::Expr::Lit(_) => {}
            _ => unimplemented!(),
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

#[derive(Debug, Clone, Copy)]
pub struct Const<'a> {
    pub span: Span,
    pub name: Ident,
    pub ty: TyId,
    pub expr: &'a Expr<'a>,
}

fn konst<'a>(ctx: &mut Ctx<'a>, konst: &rules::Const) -> Result<Const<'a>, Diag<'a>> {
    let expr = pexpr(ctx, &konst.expr)?;
    Ok(Const {
        span: konst.span,
        name: ctx.store_ident(konst.name),
        ty: ptype(ctx, &konst.ty)?.1,
        expr: ctx.intern(expr),
    })
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
    }
}

fn strukt<'a>(ctx: &mut Ctx<'a>, strukt: &rules::Struct) -> Result<Struct, Diag<'a>> {
    let mut field_names = Vec::with_capacity(strukt.fields.len());

    for field in strukt.fields.iter() {
        if field_names.contains(&ctx.ident(field.name)) {
            return Err(ctx
                .report_error(ctx.span(field.name), "failed to parse struct")
                .msg(Msg::note(
                    ctx.span(strukt.name),
                    format!("while parsing `{}`", ctx.ident(strukt.name)),
                )));
        }
        field_names.push(ctx.ident(field.name));
    }

    Ok(Struct {
        span: strukt.span,
        name: ctx.store_ident(strukt.name),
        fields: strukt
            .fields
            .iter()
            .map(|f| field(ctx, f))
            .collect::<Result<_, _>>()?,
    })
}

fn field<'a>(ctx: &mut Ctx<'a>, field: &rules::Field) -> Result<Field, Diag<'a>> {
    Ok(Field {
        span: field.span,
        name: ctx.store_ident(field.name),
        ty: ptype(ctx, &field.ty)?.1,
    })
}

#[derive(Debug, Clone)]
pub struct Func<'a> {
    pub name_span: Span,
    pub sig: &'a Sig<'a>,
    pub block: Block<'a>,
    pub attrs: Vec<Attr>,
}

impl Func<'_> {
    pub fn hash(&self) -> FuncHash {
        self.sig.hash()
    }

    pub fn is_intrinsic(&self) -> bool {
        self.has_attr(Attr::Intrinsic)
    }

    pub fn has_attr(&self, attr: Attr) -> bool {
        self.attrs.iter().any(|a| *a == attr)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FuncHash(pub u64);

fn func_sig<'a>(ctx: &mut Ctx<'a>, func: &rules::Func) -> Result<Sig<'a>, Diag<'a>> {
    let params = params(ctx, &func.params)?;

    Ok(Sig {
        span: func.span,
        ident: ctx.store_ident(func.name).id,
        params: ctx.intern_slice(&params),
        ty: func
            .ty
            .as_ref()
            .map(|t| ptype(ctx, t).map(|t| t.1))
            .transpose()?
            .unwrap_or(ctx.tys.unit()),
        linkage: Linkage::Local,
    })
}

fn extern_sig<'a>(ctx: &mut Ctx<'a>, func: &rules::ExternFunc) -> Result<Sig<'a>, Diag<'a>> {
    let params = params(ctx, &func.params)?;

    match ctx.as_str(func.convention) {
        "C" => {}
        c => {
            return Err(ctx.report_error(
                func.convention,
                format!("Unknown calling convention `{}`", c),
            ))
        }
    }

    let Some(link) = func.link else {
        return Err(ctx.report_error(
            func.name,
            format!(
                "Unknown linkage for `{}`, specify with the `link(\"<path>\")` attribute",
                ctx.as_str(func.name),
            ),
        ));
    };

    Ok(Sig {
        span: func.span,
        ident: ctx.store_ident(func.name).id,
        params: ctx.intern_slice(&params),
        ty: func
            .ty
            .as_ref()
            .map(|t| ptype(ctx, t).map(|t| t.1))
            .transpose()?
            .unwrap_or(ctx.tys.unit()),
        linkage: Linkage::External {
            link: ctx.intern_str(ctx.as_str(link)),
        },
    })
}

fn ptype<'a>(ctx: &mut Ctx<'a>, ty: &rules::PType) -> Result<(Span, TyId), Diag<'a>> {
    Ok(match ty {
        rules::PType::Simple(id) => {
            let ty = match ctx.ident(*id) {
                "u8" => ctx.tys.ty_id(&Ty::Int(IntTy::new_8(Sign::U))),
                "u16" => ctx.tys.ty_id(&Ty::Int(IntTy::new_16(Sign::U))),
                "u32" => ctx.tys.ty_id(&Ty::Int(IntTy::new_32(Sign::U))),
                "u64" => ctx.tys.ty_id(&Ty::Int(IntTy::new_64(Sign::U))),
                "i8" => ctx.tys.ty_id(&Ty::Int(IntTy::new_8(Sign::I))),
                "i16" => ctx.tys.ty_id(&Ty::Int(IntTy::new_16(Sign::I))),
                "i32" => ctx.tys.ty_id(&Ty::Int(IntTy::new_32(Sign::I))),
                "i64" => ctx.tys.ty_id(&Ty::Int(IntTy::new_64(Sign::I))),
                "f32" => ctx.tys.ty_id(&Ty::Float(FloatTy::F32)),
                "f64" => ctx.tys.ty_id(&Ty::Float(FloatTy::F64)),
                "bool" => ctx.tys.ty_id(&Ty::Bool),
                "str" => ctx.tys.ty_id(&Ty::Str),
                _ => {
                    let ident = ctx.store_ident(*id).id;
                    if let Some(strukt) = ctx.tys.struct_id(ident) {
                        ctx.tys.ty_id(&Ty::Struct(strukt))
                    } else {
                        return Err(ctx.report_error(
                            ctx.span(*id),
                            format!("expected type, got `{}`", ctx.ident(*id)),
                        ));
                    }
                }
            };

            (ctx.span(*id), ty)
        }
        rules::PType::Ref { inner, .. } => {
            let inner = ptype(ctx, inner)?;
            let inner_ty = ctx.intern(ctx.tys.ty(inner.1));
            (
                ty.span(ctx.token_buffer()),
                ctx.tys.ty_id(&Ty::Ref(inner_ty)),
            )
        }
    })
}

fn params<'a>(ctx: &mut Ctx<'a>, fparams: &[rules::Param]) -> Result<Vec<Param>, Diag<'a>> {
    let mut params = Vec::with_capacity(fparams.len());
    for p in fparams.iter() {
        let (span, ty) = ptype(ctx, &p.ty)?;
        params.push(Param {
            span: Span::from_spans(ctx.span(p.name), span),
            ty_binding: Span::from_spans(ctx.span(p.colon), span),
            ident: ctx.store_ident(p.name),
            ty,
        });
    }

    Ok(params)
}

fn func<'a>(ctx: &mut Ctx<'a>, func: &rules::Func) -> Result<Func<'a>, Diag<'a>> {
    let sig = func_sig(ctx, func)?;

    Ok(Func {
        name_span: ctx.span(func.name),
        block: block(ctx, &func.block)?,
        attrs: func.attributes.clone(),
        sig: ctx.intern(sig),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub struct Block<'a> {
    pub span: Span,
    pub stmts: &'a [Stmt<'a>],
    pub end: Option<&'a Expr<'a>>,
}

fn block<'a>(ctx: &mut Ctx<'a>, block: &rules::Block) -> Result<Block<'a>, Diag<'a>> {
    let mut stmts = block
        .stmts
        .iter()
        .map(|st| stmt(ctx, st))
        .collect::<Result<Vec<_>, _>>()?;

    let end = if let Some(last) = stmts.last() {
        match last {
            Stmt::Open(end) => {
                let end = ctx.intern(*end);
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
        stmts: ctx.intern_slice(&stmts),
        end,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub enum Stmt<'a> {
    Semi(SemiStmt<'a>),
    Open(Expr<'a>),
}

impl Stmt<'_> {
    pub fn span(&self) -> Span {
        match self {
            Stmt::Semi(semi) => semi.span(),
            Stmt::Open(open) => open.span(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub enum SemiStmt<'a> {
    Let(Let<'a>),
    Assign(Assign<'a>),
    Ret(Return<'a>),
    Expr(Expr<'a>),
}

impl SemiStmt<'_> {
    pub fn span(&self) -> Span {
        match self {
            Self::Let(let_) => let_.span,
            Self::Assign(assign) => assign.span,
            Self::Ret(ret) => ret.span,
            Self::Expr(expr) => expr.span(),
        }
    }
}

fn stmt<'a>(ctx: &mut Ctx<'a>, stmt: &rules::Stmt) -> Result<Stmt<'a>, Diag<'a>> {
    Ok(match stmt {
        rules::Stmt::Let { name, ty, assign } => Stmt::Semi(SemiStmt::Let(Let {
            span: ctx.span(*name),
            lhs: let_target(ctx, &rules::Expr::Ident(*name)),
            rhs: let_expr(ctx, assign)?,
            ty: ty.as_ref().map(|t| ptype(ctx, &t)).transpose()?,
        })),
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
                    Some(expr) => Some(pexpr(ctx, &expr)?),
                    None => None,
                },
            })),
            e => Stmt::Semi(SemiStmt::Expr(pexpr(ctx, e)?)),
        },
        rules::Stmt::Open(expr) => Stmt::Open(pexpr(ctx, &expr)?),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub struct Let<'a> {
    pub span: Span,
    pub ty: Option<(Span, TyId)>,
    pub lhs: LetTarget,
    pub rhs: Expr<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LetTarget {
    Ident(Ident),
}

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub enum Expr<'a> {
    Ident(Ident),
    Lit(Lit<'a>),
    Str(StrLit<'a>),
    Bool(BoolLit),
    Bin(BinOp<'a>),
    Access(Access<'a>),
    Struct(StructDef<'a>),
    Enum(EnumDef),
    Call(Call<'a>),
    Block(Block<'a>),
    If(If<'a>),
    Loop(Loop<'a>),
    Ref(TakeRef<'a>),
    Deref(Deref<'a>),
}

impl Expr<'_> {
    pub fn span(&self) -> Span {
        match self {
            Self::Bool(bool) => bool.span,
            Self::Ident(ident) => ident.span,
            Self::Lit(lit) => lit.span,
            Self::Call(call) => call.span,
            Self::Bin(bin) => bin.span,
            Self::Struct(def) => def.span,
            Self::Enum(enom) => enom.span,
            Self::Block(block) => block.span,
            Self::If(if_) => if_.span,
            Self::Loop(block) => block.span,
            Self::Str(str) => str.span,
            Self::Ref(ref_) => ref_.span,
            Self::Access(access) => access.span,
            Self::Deref(deref) => deref.span,
        }
    }

    pub fn is_unit(&self, ctx: &Ctx) -> Option<bool> {
        match self {
            Self::Ident(_) => None,
            Self::Deref(_)
            | Self::Str(_)
            | Self::Struct(_)
            | Self::Enum(_)
            | Self::Lit(_)
            | Self::Bool(_) => Some(false),
            Self::Call(call) => Some(ctx.tys.is_unit(call.sig.ty)),
            Self::If(if_) => {
                let block = if_.block.is_unit(ctx);
                let otherwise = if_.otherwise.map(|o| o.is_unit(ctx));
                match (block, otherwise) {
                    (Some(c1), Some(Some(c2))) => Some(c1 && c2),
                    (Some(c), _) => Some(c),
                    (None, Some(Some(c))) => Some(c),
                    _ => None,
                }
            }
            Self::Access(_) | Self::Ref(_) | Self::Bin(_) => Some(false),
            Self::Loop(_) => Some(true),
            Self::Block(block) => match block.end {
                Some(end) => end.is_unit(ctx),
                None => Some(true),
            },
        }
    }

    pub fn is_integral(&self, ctx: &Ctx) -> Option<bool> {
        match self {
            Self::Deref(_) | Self::Access(_) | Self::Ident(_) => None,
            Self::Lit(lit) => Some(lit.kind.is_int()),
            Self::Call(call) => Some(ctx.tys.ty(call.sig.ty).is_int()),
            Self::Bin(bin) => bin.is_integral(ctx),
            Self::Loop(_)
            | Self::Ref(_)
            | Self::Str(_)
            | Self::Bool(_)
            | Self::Struct(_)
            | Self::Enum(_) => Some(false),
            Self::If(_) | Self::Block(_) => todo!(),
        }
    }

    pub fn infer<'a>(&self, ctx: &AirCtx<'a>) -> Result<InferTy, Diag<'a>> {
        Ok(match self {
            Self::Lit(lit) => match lit.kind {
                LitKind::Int(_) => InferTy::Int,
                LitKind::Float(_) => InferTy::Float,
            },
            Self::Ident(ident) => InferTy::Ty(ctx.var_ty(ident.id)),
            Self::Access(access) => InferTy::Ty(aquire_access_ty(ctx, access)),
            Self::Call(call) => InferTy::Ty(call.sig.ty),
            Self::Str(_) => InferTy::Ty(TyId::STR_LIT),
            Self::Bin(bin) => {
                let lhs = bin.lhs.infer(ctx)?;
                let rhs = bin.lhs.infer(ctx)?;
                assert_eq!(lhs, rhs);
                match bin.kind {
                    BinOpKind::Add | BinOpKind::Sub | BinOpKind::Mul => lhs,
                    BinOpKind::Eq => InferTy::Ty(TyId::BOOL),
                }
            }
            _ => todo!(),
        })
    }
}

pub fn aquire_access_ty<'a>(ctx: &AirCtx<'a>, access: &Access) -> TyId {
    let ty = match access.lhs {
        Expr::Ident(ident) => ctx.var_ty(ident.id),
        _ => unimplemented!(),
    };

    let ty = ctx.tys.ty(ty);
    let id = match ty.peel_refs().0 {
        Ty::Struct(id) => id,
        Ty::Int(_) | Ty::Unit | Ty::Bool | Ty::Ref(_) | Ty::Str | Ty::Float(_) => {
            unreachable!()
        }
    };
    let mut strukt = ctx.tys.strukt(*id);

    for (i, acc) in access.accessors.iter().enumerate() {
        let ty = strukt.field_ty(acc.id);
        if i == access.accessors.len() - 1 {
            return ty;
        }

        match ctx.tys.ty(ty) {
            Ty::Struct(id) => {
                strukt = ctx.tys.strukt(id);
            }
            Ty::Int(_) | Ty::Unit | Ty::Bool | Ty::Ref(_) | Ty::Str | Ty::Float(_) => {
                unreachable!()
            }
        }
    }

    // there must be atleast one access, since bin is of type field
    unreachable!()
}

#[derive(Debug, PartialEq, Eq)]
pub enum InferTy {
    Ty(TyId),
    Int,
    Float,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StrLit<'a> {
    pub span: Span,
    pub val: &'a str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BoolLit {
    pub span: Span,
    pub val: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub struct Loop<'a> {
    pub span: Span,
    pub block: Block<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub struct TakeRef<'a> {
    pub span: Span,
    pub inner: &'a Expr<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub struct Deref<'a> {
    pub span: Span,
    pub inner: &'a Expr<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub struct If<'a> {
    pub span: Span,
    pub condition: &'a Expr<'a>,
    pub block: &'a Expr<'a>,
    pub otherwise: Option<&'a Expr<'a>>,
}

fn if_<'a>(
    ctx: &mut Ctx<'a>,
    iff: TokenId,
    expr: &rules::Expr,
    blck: &rules::Block,
    otherwise: Option<&rules::Block>,
) -> Result<If<'a>, Diag<'a>> {
    let condition = pexpr(ctx, expr)?;

    // TODO: pre reduce these
    let blck = Expr::Block(block(ctx, blck)?);
    let otherwise = match otherwise {
        Some(blck) => {
            let otherwise = Expr::Block(block(ctx, blck)?);
            Some(ctx.intern(otherwise))
        }
        None => None,
    };

    Ok(If {
        span: Span::from_spans(ctx.span(iff), blck.span()),
        condition: ctx.intern(condition),
        block: ctx.intern(blck),
        otherwise,
    })
}

fn let_target<'a>(ctx: &mut Ctx<'a>, expr: &rules::Expr) -> LetTarget {
    match expr {
        rules::Expr::Ident(ident) => LetTarget::Ident(ctx.store_ident(*ident)),
        _ => todo!(),
    }
}

fn let_expr<'a>(ctx: &mut Ctx<'a>, expr: &rules::Expr) -> Result<Expr<'a>, Diag<'a>> {
    Ok(match expr {
        rules::Expr::Ident(ident) => Expr::Ident(ctx.store_ident(*ident)),
        rules::Expr::Lit(lit) => Expr::Lit(plit(ctx, *lit)?),
        rules::Expr::Bin(op, lhs, rhs) => Expr::Bin(bin_op(ctx, *op, lhs, rhs)?),
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

fn struct_def<'a>(ctx: &mut Ctx<'a>, def: &rules::StructDef) -> Result<StructDef<'a>, Diag<'a>> {
    let id = ctx.store_ident(def.name).id;
    let fields = def
        .fields
        .iter()
        .map(|f| field_def(ctx, f))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(StructDef {
        span: def.span,
        id: ctx
            .tys
            .struct_id(id)
            .ok_or_else(|| ctx.report_error(def.name, "undefined type"))?,
        fields: ctx.intern_slice(&fields),
    })
}

fn field_def<'a>(ctx: &mut Ctx<'a>, def: &rules::FieldDef) -> Result<FieldDef<'a>, Diag<'a>> {
    Ok(FieldDef {
        span: def.span,
        name: ctx.store_ident(def.name),
        expr: let_expr(ctx, &def.expr)?,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub struct BinOp<'a> {
    pub span: Span,
    pub kind: BinOpKind,
    pub lhs: &'a Expr<'a>,
    pub rhs: &'a Expr<'a>,
}

impl BinOp<'_> {
    /// Contains all integral components.
    pub fn is_integral(&self, ctx: &Ctx) -> Option<bool> {
        self.lhs
            .is_integral(ctx)
            .map(|i| self.rhs.is_integral(ctx).map(|o| i && o))?
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOpKind {
    Add,
    Sub,
    Mul,
    Eq,
}

impl BinOpKind {
    pub fn output_is_input(&self) -> bool {
        match self {
            Self::Add | Self::Sub | Self::Mul => true,
            Self::Eq => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub struct Access<'a> {
    pub span: Span,
    pub lhs: &'a Expr<'a>,
    pub accessors: &'a [Ident],
}

fn bin_op<'a>(
    ctx: &mut Ctx<'a>,
    kind: BinOpKind,
    lhs: &rules::Expr,
    rhs: &rules::Expr,
) -> Result<BinOp<'a>, Diag<'a>> {
    Ok({
        let lhs_expr = pexpr(ctx, lhs)?;
        let rhs_expr = pexpr(ctx, rhs)?;

        BinOp {
            span: Span::from_spans(lhs_expr.span(), rhs_expr.span()),
            lhs: ctx.intern(lhs_expr),
            rhs: ctx.intern(rhs_expr),
            kind,
        }
    })
}

fn access<'a>(
    ctx: &mut Ctx<'a>,
    span: Span,
    mut lhs: &rules::Expr,
    field: TokenId,
) -> Result<Access<'a>, Diag<'a>> {
    //pub span: Span,
    //pub lhs: &'a Expr<'a>,
    //pub accessors: &'a [Ident],

    let mut accessors = vec![ctx.store_ident(field)];
    let mut span = span;

    loop {
        match lhs {
            rules::Expr::Ident(_) => {
                break;
            }
            rules::Expr::Access {
                span: next_span,
                lhs: next_lhs,
                field,
            } => {
                span = Span::from_spans(span, *next_span);
                lhs = next_lhs;
                accessors.push(ctx.store_ident(*field));
            }
            _ => return Err(ctx.report_error(span, "invalid field access")),
        }
    }

    let lhs = pexpr(ctx, lhs)?;
    Ok(Access {
        span,
        lhs: ctx.intern(lhs),
        accessors: ctx.intern_slice(&accessors),
    })
}

fn pexpr<'a>(ctx: &mut Ctx<'a>, expr: &rules::Expr) -> Result<Expr<'a>, Diag<'a>> {
    Ok(match expr {
        rules::Expr::Ident(ident) => Expr::Ident(ctx.store_ident(*ident)),
        rules::Expr::Lit(lit) => Expr::Lit(plit(ctx, *lit)?),
        rules::Expr::Bin(op, lhs, rhs) => Expr::Bin(bin_op(ctx, *op, lhs, rhs)?),
        rules::Expr::Call { span, func, args } => Expr::Call(call(ctx, *span, *func, args)?),
        rules::Expr::StructDef(def) => Expr::Struct(struct_def(ctx, def)?),
        rules::Expr::If(iff, expr, block, otherwise) => {
            Expr::If(if_(ctx, *iff, expr, block, otherwise.as_ref())?)
        }
        rules::Expr::Bool(id) => Expr::Bool(BoolLit {
            span: ctx.span(*id),
            val: ctx.kind(*id) == TokenKind::True,
        }),
        rules::Expr::Str(str) => Expr::Str(StrLit {
            span: ctx.span(*str),
            val: ctx.intern_str(ctx.as_str(*str)),
        }),
        rules::Expr::TakeRef(ampersand, inner) => Expr::Ref(take_ref(ctx, *ampersand, inner)?),
        rules::Expr::Loop(loop_, blck) => Expr::Loop(Loop {
            span: ctx.span(*loop_),
            block: block(ctx, blck)?,
        }),
        rules::Expr::Access { span, lhs, field } => Expr::Access(access(ctx, *span, lhs, *field)?),
        //rules::Expr::Deref(asterisk, inner) => Expr::Deref(deref(ctx, *asterisk, inner)?),
        _ => todo!(),
    })
}

fn take_ref<'a>(
    ctx: &mut Ctx<'a>,
    ampersand: TokenId,
    inner: &rules::Expr,
) -> Result<TakeRef<'a>, Diag<'a>> {
    let expr = pexpr(ctx, inner)?;
    let inner = ctx.intern(expr);
    Ok(TakeRef {
        span: Span::from_spans(ctx.span(ampersand), inner.span()),
        inner,
    })
}

fn deref<'a>(
    ctx: &mut Ctx<'a>,
    asterisk: TokenId,
    inner: &rules::Expr,
) -> Result<Deref<'a>, Diag<'a>> {
    let expr = pexpr(ctx, inner)?;
    let inner = ctx.intern(expr);
    Ok(Deref {
        span: Span::from_spans(ctx.span(asterisk), inner.span()),
        inner,
    })
}

fn plit<'a>(ctx: &mut Ctx<'a>, lit: TokenId) -> Result<Lit<'a>, Diag<'a>> {
    let str = ctx.as_str(lit);
    match (
        if str.contains("0x") {
            i64::from_str_radix(&str[2..], 16)
        } else {
            str.parse()
        },
        str.parse::<f64>(),
    ) {
        (Ok(val), _) => Ok(Lit {
            span: ctx.span(lit),
            kind: ctx.intern(LitKind::Int(val)),
        }),
        (Err(_), Ok(val)) => Ok(Lit {
            span: ctx.span(lit),
            kind: ctx.intern(LitKind::Float(val)),
        }),
        (Err(_), Err(_)) => Err(ctx.report_error(lit, "expected a literal")),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub struct Assign<'a> {
    pub span: Span,
    pub kind: AssignKind,
    pub lhs: AssignTarget<'a>,
    pub rhs: Expr<'a>,
}

// TODO: more assignment kinds
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssignKind {
    Equals,
    Add,
}

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub enum AssignTarget<'a> {
    Ident(Ident),
    Access(Access<'a>),
}

fn assign_target<'a>(ctx: &mut Ctx<'a>, expr: &rules::Expr) -> Result<AssignTarget<'a>, Diag<'a>> {
    Ok(match expr {
        rules::Expr::Ident(ident) => AssignTarget::Ident(ctx.store_ident(*ident)),
        rules::Expr::Access { span, lhs, field } => {
            AssignTarget::Access(access(ctx, *span, lhs, *field)?)
        }
        _ => todo!(),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub struct Return<'a> {
    pub span: Span,
    pub expr: Option<Expr<'a>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub struct Call<'a> {
    pub span: Span,
    pub sig: &'a Sig<'a>,
    pub args: &'a [Expr<'a>],
}

fn args<'a>(ctx: &mut Ctx<'a>, args: &[rules::Expr]) -> Result<&'a [Expr<'a>], Diag<'a>> {
    let args = args
        .iter()
        .map(|arg| pexpr(ctx, arg))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ctx.intern_slice(&args))
}

fn call<'a>(
    ctx: &mut Ctx<'a>,
    span: Span,
    name: TokenId,
    call_args: &[rules::Expr],
) -> Result<Call<'a>, Diag<'a>> {
    let id = ctx.store_ident(name).id;
    Ok(Call {
        sig: ctx
            .get_sig(id)
            .ok_or_else(|| ctx.report_error(name, "function is not defined"))?,
        args: args(ctx, call_args)?,
        span,
    })
}
