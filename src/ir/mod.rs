use self::lit::LitKind;
use self::sem::sem_analysis_pre_typing;
use self::sig::Linkage;
use self::ty::{IntTy, Sign, Ty};
use crate::diagnostic::{self, Diag, Msg};
use crate::ir::ctx::Ctx;
use crate::ir::ident::Ident;
use crate::ir::lit::Lit;
use crate::lex::buffer::{Buffer, Span, TokenQuery};
use crate::lex::buffer::{TokenBuffer, TokenId};
use crate::lex::kind::TokenKind;
use crate::parse::rules::prelude::{self as rules, Attr, PBinOpKind};
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

pub fn lower<'a>(tokens: &'a TokenBuffer<'a>, items: &'a [Item]) -> Result<(Ctx<'a>, TypeKey), ()> {
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
        match field.ty.peel_refs() {
            rules::PType::Simple(id) => {
                if !ctx.tys.builtin(ctx.ident(*id)) {
                    if let Some(strukt) = field.ty.retrieve_struct(ctx, structs) {
                        add_structs_recur(ctx, structs, defined, processing, strukt)?;
                    } else {
                        todo!()
                    }
                }
            }
            rules::PType::Ref { .. } => unreachable!(),
        }
    }

    processing.pop();
    defined.insert(name);

    let strukt = strukt(ctx, rules_strukt)?;
    ctx.tys.store_struct(strukt);
    Ok(())
}

// TODO: no cycle with indirection
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
        //.map_err(|err| err.msg(Msg::note(func.block.span, "while parsing this function")))?,
        sig: ctx.intern(sig),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Block<'a> {
    pub span: Span,
    pub stmts: &'a [Stmt<'a>],
    pub end: Option<&'a Expr<'a>>,
}

//// TODO: could be the span + source file
//#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
//pub struct BlockHash(u64);
//
//impl Block<'_> {
//    pub fn hash(&self) -> BlockHash {
//        let mut hash = DefaultHasher::new();
//        <Self as Hash>::hash(self, &mut hash);
//        BlockHash(hash.finish())
//    }
//}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Expr<'a> {
    Ident(Ident),
    Lit(Lit<'a>),
    Str(Span, &'a str),
    Bool(Span, bool),
    Bin(BinOp<'a>),
    Access(Access<'a>),
    Struct(StructDef<'a>),
    Enum(EnumDef),
    Call(Call<'a>),
    Block(Block<'a>),
    If(If<'a>),
    Loop(Block<'a>),
    Ref(TakeRef<'a>),
    Deref(Deref<'a>),
}

impl Expr<'_> {
    pub fn span(&self) -> Span {
        match self {
            Self::Bool(span, _) => *span,
            Self::Ident(ident) => ident.span,
            Self::Lit(lit) => lit.span,
            Self::Call(call) => call.span,
            Self::Bin(bin) => bin.span,
            Self::Struct(def) => def.span,
            Self::Enum(enom) => enom.span,
            Self::Block(block) => block.span,
            Self::If(if_) => if_.span,
            Self::Loop(block) => block.span,
            Self::Str(span, _) => *span,
            Self::Ref(ref_) => ref_.span,
            Self::Access(access) => access.span,
            Self::Deref(deref) => deref.span,
        }
    }

    pub fn is_unit(&self, ctx: &Ctx) -> Option<bool> {
        match self {
            Self::Ident(_) => None,
            Self::Deref(_)
            | Self::Str(_, _)
            | Self::Struct(_)
            | Self::Enum(_)
            | Self::Lit(_)
            | Self::Bool(_, _) => Some(false),
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
            | Self::Str(_, _)
            | Self::Bool(_, _)
            | Self::Struct(_)
            | Self::Enum(_) => Some(false),
            Self::If(_) | Self::Block(_) => todo!(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TakeRef<'a> {
    pub span: Span,
    pub inner: &'a Expr<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Deref<'a> {
    pub span: Span,
    pub inner: &'a Expr<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
        rules::Expr::Bin(op, lhs, rhs) => bin_op(ctx, op.kind, lhs, rhs)?,
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
        id: ctx.expect_struct_id(id),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Access<'a> {
    pub span: Span,
    pub lhs: &'a Expr<'a>,
    pub accessors: &'a [Ident],
}

fn bin_op<'a>(
    ctx: &mut Ctx<'a>,
    op: rules::PBinOpKind,
    lhs: &rules::Expr,
    rhs: &rules::Expr,
) -> Result<Expr<'a>, Diag<'a>> {
    Ok({
        let lhs_expr = pexpr(ctx, lhs)?;
        let rhs_expr = pexpr(ctx, rhs)?;

        match op {
            rules::PBinOpKind::Bin(kind) => Expr::Bin(BinOp {
                span: Span::from_spans(lhs_expr.span(), rhs_expr.span()),
                lhs: ctx.intern(lhs_expr),
                rhs: ctx.intern(rhs_expr),
                kind,
            }),
            rules::PBinOpKind::Accessor => {
                let mut accessors = Vec::new();
                descend_bin_op_field(ctx, op, lhs, rhs, &mut accessors);
                Expr::Access(Access {
                    span: Span::from_spans(lhs_expr.span(), rhs_expr.span()),
                    lhs: ctx.intern(rhs_expr),
                    accessors: ctx.intern_slice(&accessors[1..]),
                })
            }
        }
    })
}

pub fn descend_bin_op_field(
    ctx: &mut Ctx,
    op: rules::PBinOpKind,
    lhs: &rules::Expr,
    rhs: &rules::Expr,
    accesses: &mut Vec<Ident>,
) {
    if op == PBinOpKind::Accessor {
        match lhs {
            rules::Expr::Ident(ident) => {
                if let rules::Expr::Bin(bin, lhs, rhs) = rhs {
                    accesses.push(ctx.store_ident(*ident));
                    descend_bin_op_field(ctx, bin.kind, lhs, rhs, accesses);
                } else {
                    let rules::Expr::Ident(other) = rhs else {
                        unimplemented!()
                    };

                    accesses.push(ctx.store_ident(*other));
                    accesses.push(ctx.store_ident(*ident));
                }
            }
            _ => {}
        }
    }
}

fn pexpr<'a>(ctx: &mut Ctx<'a>, expr: &rules::Expr) -> Result<Expr<'a>, Diag<'a>> {
    Ok(match expr {
        rules::Expr::Ident(ident) => Expr::Ident(ctx.store_ident(*ident)),
        rules::Expr::Lit(lit) => Expr::Lit(plit(ctx, *lit)?),
        rules::Expr::Bin(op, lhs, rhs) => bin_op(ctx, op.kind, lhs, rhs)?,
        rules::Expr::Call { span, func, args } => Expr::Call(call(ctx, *span, *func, args)?),
        rules::Expr::StructDef(def) => Expr::Struct(struct_def(ctx, def)?),
        rules::Expr::If(iff, expr, block, otherwise) => {
            Expr::If(if_(ctx, *iff, expr, block, otherwise.as_ref())?)
        }
        rules::Expr::Bool(id) => Expr::Bool(ctx.span(*id), ctx.kind(*id) == TokenKind::True),
        rules::Expr::Str(str) => Expr::Str(ctx.span(*str), ctx.intern_str(ctx.as_str(*str))),
        rules::Expr::TakeRef(ampersand, inner) => Expr::Ref(take_ref(ctx, *ampersand, inner)?),
        rules::Expr::Loop(blck) => Expr::Loop(block(ctx, blck)?),
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
    assert!(ctx.kind(lit) == TokenKind::Int);
    Ok(Lit {
        span: ctx.span(lit),
        kind: ctx.intern(LitKind::Int(ctx.int_lit(lit))),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssignTarget<'a> {
    Ident(Ident),
    Access(Access<'a>),
}

fn assign_target<'a>(ctx: &mut Ctx<'a>, expr: &rules::Expr) -> Result<AssignTarget<'a>, Diag<'a>> {
    Ok(match expr {
        rules::Expr::Ident(ident) => AssignTarget::Ident(ctx.store_ident(*ident)),
        rules::Expr::Bin(op, lhs, rhs) => match bin_op(ctx, op.kind, lhs, rhs)? {
            Expr::Access(access) => AssignTarget::Access(access),
            _ => {
                return Err(ctx.report_error(op.span, "expected an identifier or a structure field"))
            }
        },
        _ => todo!(),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Return<'a> {
    pub span: Span,
    pub expr: Option<Expr<'a>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
