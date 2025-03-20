use self::ctx::CtxFmt;
use self::lit::LitKind;
use self::sem::sem_analysis_pre_typing;
use self::sig::Linkage;
use self::ty::{FloatTy, IntTy, Sign, Ty};
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
    match resolve_types(&mut ctx) {
        Ok(key) => sem_analysis(&ctx, &key).map(|_| (ctx, key, eval_order)),
        Err(diag) => {
            diagnostic::report(diag);
            Err(())
        }
    }
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
            rules::PType::Array { .. } => {}
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
                rules::PType::Array { .. } | rules::PType::Ref { .. } => unreachable!(),
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
            .unwrap_or(TyId::UNIT),
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
            .unwrap_or(TyId::UNIT),
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
            let (_, inner) = ptype(ctx, inner)?;
            let inner_ty = ctx.intern(ctx.tys.ty(inner));
            (
                ty.span(ctx.token_buffer()),
                ctx.tys.ty_id(&Ty::Ref(inner_ty)),
            )
        }
        rules::PType::Array { span, size, inner } => {
            let (_, inner) = ptype(ctx, inner)?;
            let inner_ty = ctx.intern(ctx.tys.ty(inner));
            (*span, ctx.tys.ty_id(&Ty::Array(*size, inner_ty)))
        }
    })
}

fn params<'a>(ctx: &mut Ctx<'a>, fparams: &[rules::Param]) -> Result<Vec<Param>, Diag<'a>> {
    let mut params = Vec::with_capacity(fparams.len());
    for p in fparams.iter() {
        match p {
            rules::Param::Named { name, colon, ty } => {
                let (span, ty) = ptype(ctx, ty)?;
                params.push(Param::Named {
                    span: Span::from_spans(ctx.span(*name), span),
                    ty_binding: Span::from_spans(ctx.span(*colon), span),
                    ident: ctx.store_ident(*name),
                    ty,
                });
            }
            rules::Param::Slf(t) => params.push(Param::Slf(ctx.store_ident(*t))),
            rules::Param::SlfRef(t) => params.push(Param::SlfRef(ctx.store_ident(*t))),
        }
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
            rhs: pexpr(ctx, assign)?,
            ty: ty.as_ref().map(|t| ptype(ctx, &t)).transpose()?,
        })),
        rules::Stmt::Semi(expr) => match expr {
            rules::Expr::Assign(assign) => Stmt::Semi(SemiStmt::Assign(Assign {
                span: expr.span(ctx.token_buffer()),
                kind: assign.kind,
                lhs: pexpr(ctx, &assign.lhs)?,
                rhs: pexpr(ctx, &assign.rhs)?,
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
    Break(Span),
    Continue(Span),
    Ident(Ident),
    Lit(Lit<'a>),
    Str(StrLit<'a>),
    Bool(BoolLit),
    Bin(BinOp<'a>),
    Access(Access<'a>),
    Unary(Unary<'a>),
    Struct(StructDef<'a>),
    Enum(EnumDef),
    Call(Call<'a>),
    //MethodCall(MethodCall<'a>),
    Block(Block<'a>),
    If(If<'a>),
    Loop(Loop<'a>),
    For(ForLoop<'a>),
    Array(ArrDef<'a>),
    IndexOf(IndexOf<'a>),
    Range(Range<'a>),
    Cast(Cast<'a>),
}

impl Expr<'_> {
    pub fn span(&self) -> Span {
        match self {
            Self::Continue(span) => *span,
            Self::Break(span) => *span,
            Self::Bool(bool) => bool.span,
            Self::Ident(ident) => ident.span,
            Self::Lit(lit) => lit.span,
            Self::Call(call) => call.span,
            //Self::MethodCall(call) => call.span,
            Self::Bin(bin) => bin.span,
            Self::Struct(def) => def.span,
            Self::Enum(enom) => enom.span,
            Self::Block(block) => block.span,
            Self::If(if_) => if_.span,
            Self::Loop(block) => block.span,
            Self::For(for_) => for_.span,
            Self::Str(str) => str.span,
            Self::Access(access) => access.span,
            Self::Unary(unary) => unary.span,
            Self::Array(arr) => match arr {
                ArrDef::Elems { span, .. } => *span,
                ArrDef::Repeated { span, .. } => *span,
            },
            Self::IndexOf(index) => index.span,
            Self::Range(range) => range.span,
            Self::Cast(cast) => cast.span,
        }
    }

    pub fn is_unit(&self, ctx: &Ctx) -> bool {
        match self {
            Self::Ident(_)
            | Self::Unary(_)
            | Self::Array(_)
            | Self::Str(_)
            | Self::Struct(_)
            | Self::Enum(_)
            | Self::Access(_)
            | Self::Bin(_)
            | Self::Lit(_)
            | Self::Range(_)
            | Self::IndexOf(_)
            | Self::Cast(_)
            | Self::Bool(_) => false,
            Self::Call(call) => ctx.tys.is_unit(call.sig.ty),
            //Self::MethodCall(call) => ctx.tys.is_unit(call.sig.ty),
            Self::If(if_) => {
                let block = if_.block.is_unit(ctx);
                let otherwise = if_.otherwise.map(|o| o.is_unit(ctx));
                match (block, otherwise) {
                    (c1, Some(c2)) => c1 && c2,
                    (c, None) => c,
                }
            }
            Self::Break(_) | Self::Continue(_) | Self::Loop(_) | Self::For(_) => true,
            Self::Block(block) => match block.end {
                Some(end) => end.is_unit(ctx),
                None => true,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub struct Cast<'a> {
    pub span: Span,
    pub lhs: &'a Expr<'a>,
    pub ty: TyId,
}

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub struct Range<'a> {
    pub span: Span,
    pub start: Option<&'a Expr<'a>>,
    pub end: Option<&'a Expr<'a>>,
    pub inclusive: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub struct IndexOf<'a> {
    pub span: Span,
    pub array: &'a Expr<'a>,
    pub index: &'a Expr<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub enum ArrDef<'a> {
    Elems {
        span: Span,
        exprs: &'a [Expr<'a>],
    },
    Repeated {
        span: Span,
        expr: &'a Expr<'a>,
        num: &'a Expr<'a>,
    },
}

impl ArrDef<'_> {
    pub fn span(&self) -> Span {
        match self {
            Self::Repeated { span, .. } => *span,
            Self::Elems { span, .. } => *span,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub struct Unary<'a> {
    pub span: Span,
    pub inner: &'a Expr<'a>,
    pub kind: UOpKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UOpKind {
    Deref,
    Ref,
    Not,
    Neg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferTy {
    Ty(TyId),
    Int,
    Float,
}

impl InferTy {
    pub fn is_abs(&self) -> bool {
        matches!(self, InferTy::Ty(_))
    }

    pub fn equiv(self, other: Self, ctx: &Ctx) -> bool {
        match self {
            Self::Int => match other {
                Self::Float => false,
                Self::Int => true,
                Self::Ty(ty) => ctx.tys.ty(ty).is_int(),
            },
            Self::Float => match other {
                Self::Float => true,
                Self::Int => false,
                Self::Ty(ty) => ctx.tys.ty(ty).is_float(),
            },
            Self::Ty(ty) => match other {
                Self::Int => ctx.tys.ty(ty).is_int(),
                Self::Float => ctx.tys.ty(ty).is_float(),
                Self::Ty(other_ty) => other_ty == ty,
            },
        }
    }

    pub fn to_string(self, ctx: &Ctx) -> String {
        match self {
            Self::Int => "{integer}".to_owned(),
            Self::Float => "{float}".to_owned(),
            Self::Ty(ty) => ty.to_string(ctx),
        }
    }
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
pub struct ForLoop<'a> {
    pub span: Span,
    pub iter: Ident,
    pub iterable: &'a Expr<'a>,
    pub block: Block<'a>,
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

    let span = match otherwise {
        Some(o) => o.span(),
        None => blck.span(),
    };

    Ok(If {
        span: Span::from_spans(ctx.span(iff), span),
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

//fn enum_def<'a>(ctx: &mut Ctx<'a>, def: &rules::EnumDef) -> Result<EnumDef, Diag<'a>> {
//    Ok(EnumDef {
//        span: def.span,
//        name: ctx.store_ident(def.name),
//        variant: variant(ctx, &def.variant),
//    })
//}

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
        expr: pexpr(ctx, &def.expr)?,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub struct BinOp<'a> {
    pub span: Span,
    pub kind: BinOpKind,
    pub lhs: &'a Expr<'a>,
    pub rhs: &'a Expr<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOpKind {
    Add,
    Sub,
    Mul,
    Div,

    Xor,

    Eq,
    Ne,
    Gt,
    Lt,
}

impl BinOpKind {
    pub fn output_is_input(&self) -> bool {
        match self {
            Self::Add | Self::Sub | Self::Mul | Self::Div => true,
            Self::Xor => true,
            Self::Ne | Self::Eq | Self::Gt | Self::Lt => false,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "*",
            Self::Div => "/",

            Self::Xor => "^",

            Self::Eq => "==",
            Self::Ne => "!=",
            Self::Gt => ">",
            Self::Lt => "<",
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
    let mut accessors = vec![ctx.store_ident(field)];
    let mut span = span;

    loop {
        match lhs {
            rules::Expr::Access {
                span: next_span,
                lhs: next_lhs,
                field,
            } => {
                span = Span::from_spans(span, *next_span);
                lhs = next_lhs;
                accessors.push(ctx.store_ident(*field));
            }
            _ => {
                break;
            }
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
        rules::Expr::Loop(loop_, blck) => Expr::Loop(Loop {
            span: ctx.span(*loop_),
            block: block(ctx, blck)?,
        }),
        rules::Expr::Access { span, lhs, field } => Expr::Access(access(ctx, *span, lhs, *field)?),
        rules::Expr::Unary(operator, kind, expr) => Expr::Unary({
            let expr = pexpr(ctx, expr)?;
            Unary {
                span: Span::from_spans(ctx.span(*operator), expr.span()),
                kind: *kind,
                inner: ctx.intern(expr),
            }
        }),
        rules::Expr::Array(arr) => Expr::Array(array(ctx, arr)?),
        rules::Expr::IndexOf { span, array, index } => {
            Expr::IndexOf(index_of(ctx, *span, array, index)?)
        }
        rules::Expr::Break(t) => Expr::Break(ctx.span(*t)),
        rules::Expr::Continue(t) => Expr::Continue(ctx.span(*t)),
        rules::Expr::For {
            span,
            iter,
            iterable,
            block,
        } => Expr::For(for_loop(ctx, *span, *iter, iterable, block)?),
        rules::Expr::Range {
            span,
            start,
            end,
            inclusive,
        } => Expr::Range({
            Range {
                start: start
                    .as_ref()
                    .map(|s| pexpr(ctx, &s))
                    .transpose()?
                    .map(|s| ctx.intern(s)),
                end: end
                    .as_ref()
                    .map(|e| pexpr(ctx, &e))
                    .transpose()?
                    .map(|e| ctx.intern(e)),
                span: *span,
                inclusive: *inclusive,
            }
        }),
        rules::Expr::Paren(inner) => pexpr(ctx, inner)?,
        rules::Expr::Cast { span, lhs, ty, .. } => {
            let lhs = pexpr(ctx, lhs)?;
            Expr::Cast(Cast {
                span: *span,
                lhs: ctx.intern(lhs),
                ty: ptype(ctx, ty)?.1,
            })
        }
        _ => todo!(),
    })
}

fn for_loop<'a>(
    ctx: &mut Ctx<'a>,
    span: Span,
    iter: TokenId,
    iterable: &rules::Expr,
    blck: &rules::Block,
) -> Result<ForLoop<'a>, Diag<'a>> {
    let iterable = pexpr(ctx, iterable)?;
    Ok(ForLoop {
        span,
        iter: ctx.store_ident(iter),
        iterable: ctx.intern(iterable),
        block: block(ctx, blck)?,
    })
}

fn index_of<'a>(
    ctx: &mut Ctx<'a>,
    span: Span,
    array: &rules::Expr,
    index: &rules::Expr,
) -> Result<IndexOf<'a>, Diag<'a>> {
    let array = pexpr(ctx, array)?;
    let index = pexpr(ctx, index)?;

    Ok(IndexOf {
        span,
        array: ctx.intern(array),
        index: ctx.intern(index),
    })
}

fn array<'a>(ctx: &mut Ctx<'a>, arr: &rules::ArrDef) -> Result<ArrDef<'a>, Diag<'a>> {
    let mut errors = Vec::new();

    match arr {
        rules::ArrDef::Elems {
            span,
            exprs: pexprs,
        } => {
            let mut exprs = Vec::with_capacity(pexprs.len());
            for expr in pexprs.iter() {
                match pexpr(ctx, expr) {
                    Ok(expr) => exprs.push(expr),
                    Err(diag) => errors.push(diag),
                }
            }

            if !errors.is_empty() {
                Err(Diag::bundle(errors))
            } else {
                Ok(ArrDef::Elems {
                    span: *span,
                    exprs: ctx.intern_slice(&exprs),
                })
            }
        }
        rules::ArrDef::Repeated { span, expr, num } => {
            let expr = pexpr(ctx, expr)?;
            let num = pexpr(ctx, num)?;

            Ok(ArrDef::Repeated {
                span: *span,
                expr: ctx.intern(expr),
                num: ctx.intern(num),
            })
        }
    }
}

fn plit<'a>(ctx: &mut Ctx<'a>, lit: TokenId) -> Result<Lit<'a>, Diag<'a>> {
    let str = ctx.as_str(lit);
    match (
        if str.contains("0x") {
            u64::from_str_radix(&str[2..], 16)
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
    pub lhs: Expr<'a>,
    pub rhs: Expr<'a>,
}

// TODO: more assignment kinds
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssignKind {
    Equals,
    Add,
    Sub,
}

impl AssignKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Equals => "=",
            Self::Add => "+=",
            Self::Sub => "-=",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub struct Return<'a> {
    pub span: Span,
    pub expr: Option<Expr<'a>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub struct MethodCall<'a> {
    pub span: Span,
    pub sig: &'a Sig<'a>,
    pub lhs: &'a Expr<'a>,
    pub args: &'a [Expr<'a>],
}

fn method_call<'a>(
    ctx: &mut Ctx<'a>,
    span: Span,
    lhs: &rules::Expr,
    name: TokenId,
    call_args: &[rules::Expr],
) -> Result<MethodCall<'a>, Diag<'a>> {
    todo!()
    //let name = ctx.store_ident(name);
    //let strukt = ctx.tys.expect_struct_id(strukt);
    //let lhs = pexpr(ctx, lhs)?;

    //Ok(MethodCall {
    //    sig: ctx
    //        .get_method_sig(strukt, name.id)
    //        .ok_or_else(|| ctx.report_error(name, "function is not defined"))?,
    //    lhs: ctx.intern(lhs),
    //    args: args(ctx, call_args)?,
    //    span,
    //})
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
