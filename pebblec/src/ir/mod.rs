use self::ctx::CtxFmt;
use self::lit::LitKind;
use self::sem::sem_analysis_pre_typing;
use self::sig::Linkage;
use self::ty::{FloatTy, IntTy, Sign, Ty, TyKind};
use crate::comp::CompErr;
use crate::ir::ctx::Ctx;
use crate::ir::ident::Ident;
use crate::ir::lit::Lit;
use enom::{Enum, EnumDef, Variant};
use ident::IdentId;
use indexmap::IndexMap;
use pebblec_parse::diagnostic::{Diag, Msg};
use pebblec_parse::lex::buffer::TokenId;
use pebblec_parse::lex::buffer::{Span, TokenQuery};
use pebblec_parse::lex::kind::TokenKind;
use pebblec_parse::lex::source::SourceMap;
use pebblec_parse::rules::prelude::PType;
use pebblec_parse::rules::prelude::{self as rules, Attr};
use pebblec_parse::{AssignKind, ItemKind, UOpKind};
use pebblec_parse::{BinOpKind, Item};
use resolve::resolve_types;
use sem::sem_analysis;
use sig::Param;
use sig::Sig;
use std::collections::HashSet;
use std::hash::Hash;
use strukt::{Field, FieldDef, Struct, StructDef};
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

#[derive(Debug)]
pub struct Ir<'ctx> {
    pub ctx: Ctx<'ctx>,
    pub key: TypeKey,
    pub const_eval_order: Vec<IdentId>,
}

pub fn lower<'a>(mut source_map: SourceMap) -> Result<Ir<'a>, CompErr> {
    let items = source_map.parse()?;
    let ctx = Ctx::new(source_map);
    match lower_items(ctx, items) {
        Ok(ir) => Ok(ir),
        Err(diag) => {
            diag.report();
            Err(CompErr::Ir)
        }
    }
}

pub fn lower_items<'a>(mut ctx: Ctx<'a>, mut items: Vec<Item>) -> Result<Ir<'a>, Diag> {
    let mut attrs = Vec::new();
    for item in items.iter_mut() {
        match &mut item.kind {
            ItemKind::Attr(attr) => attrs.push(attr),
            ItemKind::Func(func) => {
                for attr in attrs.drain(..) {
                    _ = func
                        .parse_attr(&ctx.source_map.buffer(item.source).stream(), attr)
                        .map_err(Diag::report);
                }
            }
            ItemKind::Extern(exturn) => {
                for attr in attrs.drain(..) {
                    _ = exturn
                        .parse_attr(&ctx.source_map.buffer(item.source).stream(), attr)
                        .map_err(Diag::report);
                }
            }
            _ => {
                for attr in attrs.drain(..) {
                    ctx.report_warn(attr.span, "attribute ignored").report();
                }
            }
        }
    }

    // TODO: check for duplicate struct definitions
    let structs = items
        .iter()
        .filter_map(|i| match &i.kind {
            ItemKind::Struct(strukt) => Some(strukt),
            _ => None,
        })
        .map(|s| (ctx.store_ident(s.name).id, s))
        .collect();
    add_structs(&mut ctx, &structs)?;
    ctx.build_type_layouts();

    let sigs = lower_set(
        items
            .iter()
            .filter_map(|i| match &i.kind {
                ItemKind::Func(func) => Some(func),
                _ => None,
            })
            .map(|f| func_sig(&mut ctx, None, f)),
    )?;
    ctx.store_sigs(sigs)?;

    let impls = lower_set(
        items
            .iter()
            .filter_map(|i| match &i.kind {
                ItemKind::Impl(impul) => Some(impul),
                _ => None,
            })
            .map(|impul| ptype(&mut ctx, &impul.ty).map(|ty| (impul, ty.1))),
    )?;

    let method_sigs = lower_set(impls.iter().map(|(impul, ty)| {
        impul
            .funcs
            .iter()
            .map(|f| func_sig(&mut ctx, Some(*ty), f))
            .collect::<Result<Vec<_>, _>>()
            .map(|sigs| (ty, sigs))
    }))?;
    for (ty, sigs) in method_sigs.into_iter() {
        ctx.store_impl_sigs(*ty, sigs)?;
    }

    let extern_sigs = lower_set(
        items
            .iter()
            .filter_map(|i| match &i.kind {
                ItemKind::Extern(func) => Some(func),
                _ => None,
            })
            .map(|f| {
                f.funcs
                    .iter()
                    .map(|f| extern_sig(&mut ctx, f))
                    .collect::<Result<Vec<_>, _>>()
            }),
    )?;
    ctx.store_sigs(extern_sigs.into_iter().flatten())?;

    let consts = items
        .iter()
        .filter_map(|i| match &i.kind {
            ItemKind::Const(konst) => Some(konst),
            _ => None,
        })
        .map(|c| (ctx.store_ident(c.name).id, c))
        .collect();
    let const_eval_order = add_consts(&mut ctx, &consts)?;

    let methods = lower_set(impls.iter().map(|(impul, ty)| {
        impul
            .funcs
            .iter()
            .map(|f| func(&mut ctx, Some(*ty), f))
            .collect::<Result<Vec<_>, _>>()
    }))?;
    ctx.store_funcs(methods.into_iter().flatten());

    let funcs = lower_set(
        items
            .iter()
            .filter_map(|i| match &i.kind {
                ItemKind::Func(func) => Some(func),
                _ => None,
            })
            .map(|f| func(&mut ctx, None, f)),
    )?;
    ctx.store_funcs(funcs);

    sem_analysis_pre_typing(&ctx)?;
    let key = resolve_types(&mut ctx)?;
    sem_analysis(&ctx, &key)?;

    Ok(Ir {
        ctx,
        key,
        const_eval_order,
    })
}

fn add_structs<'a, 'ctx>(
    ctx: &mut Ctx<'ctx>,
    structs: &IndexMap<IdentId, &'a rules::Struct>,
) -> Result<(), Diag> {
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

fn add_structs_recur<'a, 'ctx>(
    ctx: &mut Ctx<'ctx>,
    structs: &IndexMap<IdentId, &'a rules::Struct>,
    defined: &mut HashSet<IdentId>,
    processing: &mut Vec<StructInfo<'a>>,
    rules_strukt: &'a rules::Struct,
) -> Result<(), Diag> {
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
        let mut ty = field.ty.peel_refs();

        // TODO: introduce indirection detection so that structs can have references to themselves
        loop {
            match ty {
                rules::PType::Simple(_, id) => {
                    if !ctx.tys.is_builtin(ctx.as_str(id).as_ref()) {
                        if let Some(strukt) = retrieve_struct(&field.ty, ctx, structs) {
                            add_structs_recur(ctx, structs, defined, processing, strukt)?;
                        } else {
                            errors.push(ctx.report_error(field.ty.span(), "undefined type"));
                        }
                    }

                    break;
                }
                rules::PType::Array { inner, .. } => {
                    ty = &*inner;
                }
                rules::PType::Slice { inner, .. } => {
                    ty = &*inner;
                }
                rules::PType::Ref { inner, .. } => {
                    ty = &*inner;
                }
            }
        }
    }

    processing.pop();
    defined.insert(name);

    match strukt(ctx, rules_strukt) {
        Ok(strukt) => {
            for field in strukt.fields.iter() {
                if !field.ty.is_sized() {
                    errors.push(ctx.report_error(field.span, "struct fields must be sized"));
                }
            }

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

// TODO: need some sort of namespace so that cycles accross files can be reported without panicking
//
// TODO: no cycle with indirection
fn report_struct_cycle<'a, 'ctx>(
    ctx: &mut Ctx<'ctx>,
    processing: &[StructInfo<'a>],
    current: &'a rules::Struct,
) -> Diag {
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
            title.push_str(&format!("`{}`", ctx.as_str(curr.strukt.name)));
            title.push_str(", ");
        }

        if let Some(field) = curr.strukt.fields.iter().find(|f| {
            ctx.store_ident(match f.ty {
                rules::PType::Simple(_, id) => id,
                rules::PType::Array { .. }
                | rules::PType::Slice { .. }
                | rules::PType::Ref { .. } => unreachable!(),
            })
            .id == next.id
        }) {
            msgs.push(Msg::error_span(&ctx.source_map, ctx.span(curr.strukt.name)));
            msgs.push(Msg::note_span(&ctx.source_map, field.span));
        }
    }

    title.truncate(title.len() - 2);
    let span = ctx.span(processing[cycle_start].strukt.name);
    ctx.report_error(span, title).msgs(msgs)
}

/// TODO: this does not consider indirection, this is strictly for descending type relationships for
/// type sizing
pub fn retrieve_struct<'a, 'ctx>(
    ty: &PType,
    ctx: &mut Ctx<'ctx>,
    structs: &IndexMap<IdentId, &'a rules::Struct>,
) -> Option<&'a rules::Struct> {
    match ty {
        PType::Simple(_, id) => {
            let ident = ctx.store_ident(*id).id;
            structs.get(&ident).map(|s| *s)
        }
        PType::Ref { inner, .. } => retrieve_struct(inner, ctx, structs),
        PType::Array { inner, .. } => retrieve_struct(inner, ctx, structs),
        PType::Slice { inner, .. } => retrieve_struct(inner, ctx, structs),
    }
}

pub fn add_consts<'a, 'ctx>(
    ctx: &mut Ctx<'ctx>,
    consts: &IndexMap<IdentId, &'a rules::Const>,
) -> Result<Vec<IdentId>, Diag> {
    let mut eval_order = Vec::new();
    let mut defined = HashSet::with_capacity(consts.len());
    let mut processing = Vec::with_capacity(consts.len());

    for konst in consts.values() {
        add_consts_recur(
            ctx,
            &consts,
            &mut defined,
            &mut processing,
            konst,
            &mut eval_order,
        )?;
    }

    Ok(eval_order)
}

struct ConstInfo<'a> {
    id: IdentId,
    konst: &'a rules::Const,
}

fn add_consts_recur<'a, 'ctx>(
    ctx: &mut Ctx<'ctx>,
    consts: &IndexMap<IdentId, &'a rules::Const>,
    defined: &mut HashSet<IdentId>,
    processing: &mut Vec<ConstInfo<'a>>,
    rules_const: &'a rules::Const,
    evaluation_order: &mut Vec<IdentId>,
) -> Result<(), Diag> {
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

    process_expr(
        ctx,
        consts,
        defined,
        processing,
        rules_const,
        evaluation_order,
        name,
        &rules_const.expr,
    )?;

    processing.pop();
    defined.insert(name);

    let konst = konst(ctx, rules_const)?;
    ctx.store_const(konst);
    Ok(())
}

fn process_expr<'a, 'ctx>(
    ctx: &mut Ctx<'ctx>,
    consts: &IndexMap<IdentId, &'a rules::Const>,
    defined: &mut HashSet<IdentId>,
    processing: &mut Vec<ConstInfo<'a>>,
    rules_const: &'a rules::Const,
    evaluation_order: &mut Vec<IdentId>,
    name_of_const: IdentId,
    expr: &rules::Expr,
) -> Result<(), Diag> {
    match expr {
        rules::Expr::Lit(_) => {
            evaluation_order.push(name_of_const);
        }
        rules::Expr::Ident(other) => {
            let ident = ctx.store_ident(*other);
            if let Some(other) = consts.get(&ident.id) {
                add_consts_recur(ctx, consts, defined, processing, other, evaluation_order)?;
                evaluation_order.push(name_of_const);
            } else {
                return Err(ctx.undeclared(ident));
            }
        }
        rules::Expr::Bin(_, _, lhs, rhs) => {
            process_expr(
                ctx,
                consts,
                defined,
                processing,
                rules_const,
                evaluation_order,
                name_of_const,
                lhs,
            )?;
            process_expr(
                ctx,
                consts,
                defined,
                processing,
                rules_const,
                evaluation_order,
                name_of_const,
                rhs,
            )?;
        }
        _ => unimplemented!(),
    }

    Ok(())
}

// TODO: need some sort of namespace so that cycles accross files can be reported without panicking
fn report_const_cycle<'a, 'ctx>(
    ctx: &mut Ctx<'ctx>,
    processing: &[ConstInfo<'a>],
    current: &'a rules::Const,
) -> Diag {
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
            title.push_str(&format!("`{}`", ctx.as_str(curr.konst.name)));
            title.push_str(", ");
        }

        match curr.konst.expr {
            rules::Expr::Ident(ident) => {
                if ctx.store_ident(ident).id == next.id {
                    msgs.push(Msg::error_span(&ctx.source_map, ctx.span(curr.konst.name)));
                    msgs.push(Msg::note_span(&ctx.source_map, ctx.span(ident)));
                }
            }
            rules::Expr::Lit(_) => {}
            _ => unimplemented!(),
        }
    }

    title.truncate(title.len() - 2);
    let span = ctx.span(processing[cycle_start].konst.name);
    ctx.report_error(span, title).msgs(msgs)
}

fn lower_set<'a, O>(items: impl Iterator<Item = Result<O, Diag>>) -> Result<Vec<O>, Diag> {
    let mut errors = Vec::new();
    let mut set = Vec::new();
    for item in items {
        match item {
            Ok(item) => {
                set.push(item);
            }
            Err(diag) => {
                errors.push(diag);
            }
        }
    }

    if !errors.is_empty() {
        Err(Diag::bundle(errors))
    } else {
        Ok(set)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Const<'a> {
    pub span: Span,
    pub name: Ident,
    pub ty: Ty,
    pub expr: &'a Expr<'a>,
}

fn konst<'a>(ctx: &mut Ctx<'a>, konst: &rules::Const) -> Result<Const<'a>, Diag> {
    let expr = pexpr(ctx, &konst.expr)?;
    Ok(Const {
        span: konst.span,
        name: ctx.store_ident(konst.name),
        ty: ptype(ctx, &konst.ty)?.1,
        expr: ctx.intern(expr),
    })
}

#[allow(unused)]
fn enom<'a>(ctx: &mut Ctx<'a>, enom: &rules::Enum) -> Result<Enum, Diag> {
    let mut variant_names = Vec::with_capacity(enom.variants.len());

    for field in enom.variants.iter() {
        if variant_names.contains(&ctx.as_str(field.name)) {
            todo!()
            //return Err(ctx.errors(
            //    "failed to parse enum",
            //    [
            //        Msg::error(ctx.span(field.name), "variant already declared"),
            //        Msg::note(
            //            ctx.span(enom.name),
            //            format!("while parsing `{}`", ctx.as_str(enom.name)),
            //        ),
            //    ],
            //));
        }
        variant_names.push(ctx.as_str(field.name));
    }

    Ok(Enum {
        span: enom.span,
        name: ctx.store_ident(enom.name),
        variants: enom.variants.iter().map(|f| variant(ctx, f)).collect(),
    })
}

#[allow(unused)]
fn variant<'a>(ctx: &mut Ctx<'a>, variant: &rules::Variant) -> Variant {
    Variant {
        span: variant.span,
        name: ctx.store_ident(variant.name),
    }
}

fn strukt<'a>(ctx: &mut Ctx<'a>, strukt: &rules::Struct) -> Result<Struct, Diag> {
    let mut field_names = Vec::with_capacity(strukt.fields.len());

    for field in strukt.fields.iter() {
        if field_names.contains(&ctx.as_str(field.name)) {
            return Err(ctx
                .report_error(ctx.span(field.name), "failed to parse struct")
                .msg(Msg::note(
                    &ctx.source_map,
                    ctx.span(strukt.name),
                    format!("while parsing `{}`", ctx.as_str(strukt.name)),
                )));
        }
        field_names.push(ctx.as_str(field.name));
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

fn field<'a>(ctx: &mut Ctx<'a>, field: &rules::Field) -> Result<Field, Diag> {
    Ok(Field {
        span: field.span,
        name: ctx.store_ident(field.name),
        ty: ptype(ctx, &field.ty)?.1,
    })
}

#[derive(Debug, Clone, PartialEq)]
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

fn func_sig<'a>(
    ctx: &mut Ctx<'a>,
    method_self: Option<Ty>,
    func: &rules::Func,
) -> Result<Sig<'a>, Diag> {
    let params = params(ctx, &func.params)?;

    let ty = func
        .ty
        .as_ref()
        .map(|t| ptype(ctx, &t).map(|t| t.1))
        .transpose()?
        .unwrap_or(Ty::UNIT);

    // TODO: cascading `Self`?
    //// check for `Self` type
    //let ty = if func
    //    .ty.as_ref()
    //    .is_some_and(|t| matches!(t, PType::Simple(_, t) if ctx.as_str(t) == "Self"))
    //{
    //    match method_self {
    //        Some(ty) => ty,
    //        None => {
    //            return Err(ctx.report_error(
    //                func.ty.as_ref().unwrap().span(),
    //                "`Self` cannot resolve to a type",
    //            ));
    //        }
    //    }
    //} else {
    //    func.ty
    //        .as_ref()
    //        .map(|t| ptype(ctx, &t).map(|t| t.1))
    //        .transpose()?
    //        .unwrap_or(Ty::UNIT)
    //};

    Ok(Sig {
        span: func.span,
        ident: ctx.store_ident(func.name).id,
        params: ctx.intern_slice(&params),
        method_self,
        ty,
        linkage: Linkage::Local,
    })
}

fn extern_sig<'a>(ctx: &mut Ctx<'a>, func: &rules::ExternFunc) -> Result<Sig<'a>, Diag> {
    let params = params(ctx, &func.params)?;

    match ctx.as_str(func.convention).as_ref() {
        "C" => {}
        c => {
            return Err(ctx.report_error(
                func.convention,
                format!("Unknown calling convention `{}`", c),
            ));
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
        method_self: None,
        ty: func
            .ty
            .as_ref()
            .map(|t| ptype(ctx, t).map(|t| t.1))
            .transpose()?
            .unwrap_or(Ty::UNIT),
        linkage: Linkage::External {
            link: ctx.intern_str(ctx.as_str(link).as_ref()),
        },
    })
}

fn ptype<'a>(ctx: &mut Ctx<'a>, ty: &rules::PType) -> Result<(Span, Ty), Diag> {
    Ok(match ty {
        rules::PType::Simple(_, id) => {
            let ty = match ctx.as_str(id).as_ref() {
                "u8" => TyKind::Int(IntTy::new_8(Sign::U)),
                "u16" => TyKind::Int(IntTy::new_16(Sign::U)),
                "u32" => TyKind::Int(IntTy::new_32(Sign::U)),
                "u64" => TyKind::Int(IntTy::new_64(Sign::U)),
                "i8" => TyKind::Int(IntTy::new_8(Sign::I)),
                "i16" => TyKind::Int(IntTy::new_16(Sign::I)),
                "i32" => TyKind::Int(IntTy::new_32(Sign::I)),
                "i64" => TyKind::Int(IntTy::new_64(Sign::I)),
                "f32" => TyKind::Float(FloatTy::F32),
                "f64" => TyKind::Float(FloatTy::F64),
                "bool" => TyKind::Bool,
                "str" => TyKind::Str,
                _ => {
                    let ident = ctx.store_ident(*id).id;
                    if let Some(strukt) = ctx.tys.struct_id(ident) {
                        TyKind::Struct(strukt)
                    } else {
                        return Err(ctx.report_error(
                            ctx.span(*id),
                            format!("expected type, got `{}`", ctx.as_str(id)),
                        ));
                    }
                }
            };

            (ctx.span(*id), ctx.tys.intern_kind(ty))
        }
        rules::PType::Ref { inner, .. } => {
            let (_, inner) = ptype(ctx, inner)?;
            (ty.span(), ctx.tys.intern_kind(TyKind::Ref(inner.0)))
        }
        rules::PType::Array { span, size, inner } => {
            let (_, inner) = ptype(ctx, inner)?;
            (*span, ctx.tys.intern_kind(TyKind::Array(*size, inner.0)))
        }
        rules::PType::Slice { span, inner } => {
            let (_, inner) = ptype(ctx, inner)?;
            (*span, ctx.tys.intern_kind(TyKind::Slice(inner.0)))
        }
    })
}

fn params<'a>(ctx: &mut Ctx<'a>, fparams: &[rules::Param]) -> Result<Vec<Param>, Diag> {
    let mut params = Vec::with_capacity(fparams.len());
    for p in fparams.iter() {
        match p {
            rules::Param::Named { name, ty, .. } => {
                let (span, ty) = ptype(ctx, ty)?;
                params.push(Param::Named {
                    span,
                    ident: ctx.store_ident(*name),
                    ty,
                });
            }
            rules::Param::Slf(t) => params.push(Param::Slf(ctx.store_ident(*t))),
        }
    }

    Ok(params)
}

fn func<'a>(
    ctx: &mut Ctx<'a>,
    method_self: Option<Ty>,
    func: &rules::Func,
) -> Result<Func<'a>, Diag> {
    let sig = func_sig(ctx, method_self, func)?;

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

fn block<'a>(ctx: &mut Ctx<'a>, block: &rules::Block) -> Result<Block<'a>, Diag> {
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

fn stmt<'a>(ctx: &mut Ctx<'a>, stmt: &rules::Stmt) -> Result<Stmt<'a>, Diag> {
    Ok(match stmt {
        rules::Stmt::Let {
            span,
            name,
            ty,
            assign,
            ..
        } => Stmt::Semi(SemiStmt::Let(Let {
            span: *span,
            lhs: let_target(ctx, &rules::Expr::Ident(*name)),
            rhs: pexpr(ctx, assign)?,
            ty: ty.as_ref().map(|t| ptype(ctx, &t)).transpose()?,
        })),
        rules::Stmt::Semi(expr) => match expr {
            rules::Expr::Assign(assign) => Stmt::Semi(SemiStmt::Assign(Assign {
                span: assign.span,
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
    pub ty: Option<(Span, Ty)>,
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
    MethodCall(MethodCall<'a>),
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
            Self::MethodCall(call) => call.span,
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
}

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub struct Cast<'a> {
    pub span: Span,
    pub lhs: &'a Expr<'a>,
    pub ty: Ty,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferTy {
    Ty(Ty),
    Int,
    Float,
}

impl InferTy {
    pub fn equiv(self, other: Self) -> bool {
        match self {
            Self::Int => match other {
                Self::Float => false,
                Self::Int => true,
                Self::Ty(ty) => ty.is_int(),
            },
            Self::Float => match other {
                Self::Float => true,
                Self::Int => false,
                Self::Ty(ty) => ty.is_float(),
            },
            Self::Ty(ty) => match other {
                Self::Int => ty.is_int(),
                Self::Float => ty.is_float(),
                Self::Ty(other_ty) => other_ty.equiv(*ty.0),
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
    span: Span,
    expr: &rules::Expr,
    blck: &rules::Block,
    otherwise: Option<&rules::Block>,
) -> Result<If<'a>, Diag> {
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
        span,
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

//fn enum_def<'a>(ctx: &mut Ctx<'a>, def: &rules::EnumDef) -> Result<EnumDef, Diag> {
//    Ok(EnumDef {
//        span: def.span,
//        name: ctx.store_ident(def.name),
//        variant: variant(ctx, &def.variant),
//    })
//}

fn struct_def<'a>(ctx: &mut Ctx<'a>, def: &rules::StructDef) -> Result<StructDef<'a>, Diag> {
    let ident = ctx.store_ident(def.name).id;
    let fields = def
        .fields
        .iter()
        .map(|f| field_def(ctx, f))
        .collect::<Result<Vec<_>, _>>()?;

    let id = ctx
        .tys
        .struct_id(ident)
        .ok_or_else(|| ctx.report_error(def.name, "undefined type"))?;
    let ty = ctx.tys.intern_kind(TyKind::Struct(id));

    Ok(StructDef {
        span: def.span,
        ty,
        id,
        fields: ctx.intern_slice(&fields),
    })
}

fn field_def<'a>(ctx: &mut Ctx<'a>, def: &rules::FieldDef) -> Result<FieldDef<'a>, Diag> {
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

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub struct Access<'a> {
    pub span: Span,
    pub lhs: &'a Expr<'a>,
    pub accessors: &'a [Ident],
}

fn bin_op<'a>(
    ctx: &mut Ctx<'a>,
    span: Span,
    kind: BinOpKind,
    lhs: &rules::Expr,
    rhs: &rules::Expr,
) -> Result<BinOp<'a>, Diag> {
    Ok({
        let lhs_expr = pexpr(ctx, lhs)?;
        let rhs_expr = pexpr(ctx, rhs)?;

        BinOp {
            span,
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
) -> Result<Access<'a>, Diag> {
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

fn pexpr<'a>(ctx: &mut Ctx<'a>, expr: &rules::Expr) -> Result<Expr<'a>, Diag> {
    Ok(match expr {
        rules::Expr::Ident(ident) => Expr::Ident(ctx.store_ident(*ident)),
        rules::Expr::Lit(lit) => Expr::Lit(plit(ctx, *lit)?),
        rules::Expr::Bin(span, op, lhs, rhs) => Expr::Bin(bin_op(ctx, *span, *op, lhs, rhs)?),
        rules::Expr::Call { span, func, args } => Expr::Call(call(ctx, *span, *func, args)?),
        rules::Expr::StructDef(def) => Expr::Struct(struct_def(ctx, def)?),
        rules::Expr::If {
            span,
            condition,
            block,
            otherwise,
        } => Expr::If(if_(ctx, *span, condition, block, otherwise.as_ref())?),
        rules::Expr::Bool(id) => Expr::Bool(BoolLit {
            span: ctx.span(*id),
            val: ctx.kind(*id) == TokenKind::True,
        }),
        rules::Expr::Str(str) => Expr::Str(StrLit {
            span: ctx.span(*str),
            val: ctx.intern_str(ctx.as_str(str).as_ref()),
        }),
        rules::Expr::Loop(loop_, blck) => Expr::Loop(Loop {
            span: ctx.span(*loop_),
            block: block(ctx, blck)?,
        }),
        rules::Expr::Access { span, lhs, field } => Expr::Access(access(ctx, *span, lhs, *field)?),
        rules::Expr::Unary(span, _, kind, expr) => Expr::Unary({
            let expr = pexpr(ctx, expr)?;
            Unary {
                span: *span,
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
        rules::Expr::MethodCall {
            span,
            receiver,
            method,
            args,
        } => Expr::MethodCall(method_call(ctx, *span, receiver, *method, args)?),
        expr => panic!("expected expression: {expr:#?}"),
    })
}

fn for_loop<'a>(
    ctx: &mut Ctx<'a>,
    span: Span,
    iter: TokenId,
    iterable: &rules::Expr,
    blck: &rules::Block,
) -> Result<ForLoop<'a>, Diag> {
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
) -> Result<IndexOf<'a>, Diag> {
    let array = pexpr(ctx, array)?;
    let index = pexpr(ctx, index)?;

    Ok(IndexOf {
        span,
        array: ctx.intern(array),
        index: ctx.intern(index),
    })
}

fn array<'a>(ctx: &mut Ctx<'a>, arr: &rules::ArrDef) -> Result<ArrDef<'a>, Diag> {
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

fn plit<'a>(ctx: &mut Ctx<'a>, lit: TokenId) -> Result<Lit<'a>, Diag> {
    let str = ctx.as_str(lit);
    match (
        if str.contains("0x") {
            u64::from_str_radix(&str[2..], 16)
        } else if str.contains("0b") {
            u64::from_str_radix(&str[2..], 2)
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

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub struct Return<'a> {
    pub span: Span,
    pub expr: Option<Expr<'a>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub struct MethodCall<'a> {
    pub span: Span,
    pub receiver: MethodPath<'a>,
    pub call: Ident,
    pub args: &'a [Expr<'a>],
}

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub enum MethodPath<'a> {
    Field(&'a Expr<'a>),
    Path(Span, Ty),
}

impl MethodPath<'_> {
    pub fn span(&self) -> Span {
        match self {
            Self::Field(expr) => expr.span(),
            Self::Path(span, _) => *span,
        }
    }
}

fn method_call<'a>(
    ctx: &mut Ctx<'a>,
    span: Span,
    receiver: &rules::MethodPath,
    call: TokenId,
    call_args: &[rules::Expr],
) -> Result<MethodCall<'a>, Diag> {
    let receiver = match receiver {
        rules::MethodPath::Field(field) => MethodPath::Field({
            let expr = pexpr(ctx, field)?;
            ctx.intern(expr)
        }),
        rules::MethodPath::Path(path) => {
            assert_eq!(path.segments.len(), 1);
            let token = path.segments[0];
            let ty = ptype(ctx, &PType::Simple(ctx.span(token), token))?.1;
            MethodPath::Path(path.span, ty)
        }
    };

    Ok(MethodCall {
        span,
        receiver,
        call: ctx.store_ident(call),
        args: args(ctx, call_args)?,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub struct Call<'a> {
    pub span: Span,
    pub ident_span: Span,
    pub sig: &'a Sig<'a>,
    pub args: &'a [Expr<'a>],
}

fn args<'a>(ctx: &mut Ctx<'a>, args: &[rules::Expr]) -> Result<&'a [Expr<'a>], Diag> {
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
) -> Result<Call<'a>, Diag> {
    let id = ctx.store_ident(name).id;
    let args = args(ctx, call_args)?;
    Ok(Call {
        sig: ctx
            .get_sig(id)
            .ok_or_else(|| ctx.report_error(name, "function is not defined"))?,
        ident_span: ctx.span(name),
        args,
        span,
    })
}
