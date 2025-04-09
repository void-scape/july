use pebblec_arena::BlobArena;
use pebblec_parse::lex::buffer::{Span, TokenBuffer, TokenId, TokenQuery};
use pebblec_parse::lex::kind::TokenKind;
use pebblec_parse::matc::{Bracket, Curly, DelimPair, Paren};
use pebblec_parse::rules::prelude::{
    ArrDef, Assign, Attribute, Block, Const, Expr, ExternBlock, ExternFunc, Field, FieldDef, Func,
    Impl, MethodPath, PType, Param, Path, Stmt, Struct, StructDef, Use,
};
use std::borrow::Borrow;
use std::ops::Deref;

const INDENT: usize = 4;

pub fn indent_str<'a>(arena: &BlobArena, str: &str) -> &'a str {
    arena.alloc_str(&format!(
        "{}{}",
        String::from_utf8([' ' as u8; INDENT].to_vec()).unwrap(),
        str
    ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakCond {
    Always,
    MoreThanOne,
    Width,
}

#[derive(Debug, Clone, Copy)]
pub enum Node<'a> {
    Group(&'a [Node<'a>]),
    Indent(&'a Node<'a>, usize),
    Text(&'a str),
    FlatText(&'a str),
    BreakText(&'a str),
}

impl<'a> Node<'a> {
    pub fn space() -> Self {
        Self::Text(" ")
    }

    pub fn nl() -> Self {
        Self::Text("\n")
    }

    pub fn token<T: Borrow<TokenId>>(buf: &'a TokenBuffer, token: T) -> Self {
        let t = *token.borrow();
        if buf.kind(t) == TokenKind::Str {
            Self::Text(String::leak(format!("\"{}\"", buf.as_str(t))))
        } else {
            Self::Text(buf.as_str(t))
        }
    }

    pub fn group(arena: &BlobArena, nodes: &[Node<'a>]) -> Self {
        Self::Group(arena.alloc_slice(nodes))
    }

    pub fn set(
        arena: &BlobArena,
        condition: BreakCond,
        elems: impl ExactSizeIterator<Item = Vec<Node<'a>>>,
    ) -> Self {
        let len = elems.len();
        let mut set = Vec::with_capacity(len);

        for (i, nodes) in elems.enumerate() {
            set.extend(nodes);
            match condition {
                BreakCond::Always => {
                    if i == len - 1 {
                        set.extend([Node::Text(",")]);
                    } else {
                        set.push(Node::Text(",\n"));
                    }
                }
                BreakCond::Width => {
                    if i == len - 1 {
                        set.extend([Node::BreakText(",")]);
                    } else {
                        set.extend([Node::Text(","), Node::FlatText(" "), Node::BreakText("\n")]);
                    }
                }
                BreakCond::MoreThanOne => unreachable!(),
            }
        }

        Self::Group(arena.alloc_slice(&set))
    }

    pub fn delimited_with<T: Deref<Target = R>, R: ?Sized, D: DelimPair>(
        buf: &'a TokenBuffer,
        arena: &BlobArena,
        _delims: D,
        inner: &T,
        condition: BreakCond,
        nodify: impl FnOnce(&'a TokenBuffer, &BlobArena, &R) -> Option<Node<'a>>,
    ) -> Self {
        match condition {
            BreakCond::Width => {
                let needs_spaces = D::open() == TokenKind::OpenCurly;
                match nodify(buf, arena, T::deref(inner)) {
                    Some(node) => {
                        if !needs_spaces {
                            Self::Group(arena.alloc_slice(&[
                                Self::Text(D::open().as_str()),
                                Self::BreakText("\n"),
                                node,
                                Self::BreakText("\n"),
                                Self::Text(D::close().as_str()),
                            ]))
                        } else {
                            Self::Group(arena.alloc_slice(&[
                                Self::Text(D::open().as_str()),
                                Self::BreakText("\n"),
                                Self::FlatText(" "),
                                node,
                                Self::FlatText(" "),
                                Self::BreakText("\n"),
                                Self::Text(D::close().as_str()),
                            ]))
                        }
                    }
                    None => Self::Group(arena.alloc_slice(&[
                        Self::Text(D::open().as_str()),
                        Self::Text(D::close().as_str()),
                    ])),
                }
            }
            BreakCond::Always => {
                let needs_spaces = D::open() == TokenKind::OpenCurly;
                match nodify(buf, arena, T::deref(inner)) {
                    Some(node) => {
                        if !needs_spaces {
                            Self::Group(arena.alloc_slice(&[
                                Self::Text(D::open().as_str()),
                                Self::Text("\n"),
                                node,
                                Self::Text("\n"),
                                Self::Text(D::close().as_str()),
                            ]))
                        } else {
                            Self::Group(arena.alloc_slice(&[
                                Self::Text(D::open().as_str()),
                                Self::Text("\n"),
                                node,
                                Self::Text("\n"),
                                Self::Text(D::close().as_str()),
                            ]))
                        }
                    }
                    None => Self::Group(arena.alloc_slice(&[
                        Self::Text(D::open().as_str()),
                        Self::Text(D::close().as_str()),
                    ])),
                }
            }
            // handled in `nodify_block`
            BreakCond::MoreThanOne => unreachable!(),
        }
    }

    pub fn indent_delimited_with<T: Deref<Target = R>, R: ?Sized>(
        buf: &'a TokenBuffer,
        arena: &BlobArena,
        delims: impl DelimPair,
        inner: &T,
        condition: BreakCond,
        nodify: impl FnOnce(&'a TokenBuffer, &BlobArena, &R) -> Option<Node<'a>>,
    ) -> Self {
        Node::delimited_with(buf, arena, delims, inner, condition, |buf, arena, inner| {
            nodify(buf, arena, inner).map(|n| Node::Indent(arena.alloc(n), INDENT))
        })
    }

    // TODO: if this is ever a problem, add support for unicode width?
    pub fn flat_width(&self) -> usize {
        match self {
            Self::Group(nodes) => nodes.iter().map(|n| n.flat_width()).sum(),
            Self::Indent(node, spaces) => node.flat_width() + spaces,
            Self::Text(str) => str.len(),
            Self::FlatText(str) => str.len(),
            Self::BreakText(_) => 0,
        }
    }
}

fn check_whitespace_span<'a>(buf: &'a TokenBuffer, span: Span, collection: &mut Vec<Node<'a>>) {
    // the span can only have come from this buffer, so the token must exist
    let token = buf.token_with_start(span.start as usize).unwrap();
    check_whitespace(buf, token, collection);
}

fn check_whitespace_slice<'a>(slice: &str, collection: &mut Vec<Node<'a>>) {
    if slice.matches("\n").count() >= 2 {
        collection.push(Node::Text("\n\n"));
    }
}

pub fn check_whitespace<'a>(buf: &'a TokenBuffer, token: TokenId, collection: &mut Vec<Node<'a>>) {
    let next_span = buf.span(token);
    let end = next_span.start as usize;

    let mut slice = if let Some(prev_span) = buf.prev(token).map(|t| buf.span(t)) {
        let start = prev_span.end as usize;
        &buf.source_ref().source[start..end]
    } else {
        &buf.source_ref().source[..end]
    };

    loop {
        if let Some(index) = slice.find("//") {
            check_whitespace_slice(&slice[..index], collection);
            if let Some(end) = slice[index..].find("\n") {
                collection.extend([Node::Text(&slice[index..(index + end)]), Node::nl()]);
                slice = &slice[(index + end)..];
            } else {
                collection.push(Node::Text(&slice[index..]));
                break;
            }
        } else {
            if !slice.is_empty() {
                check_whitespace_slice(slice, collection);
            }
            break;
        }
    }
}

pub fn nodify_func<'a>(buf: &'a TokenBuffer, arena: &BlobArena, func: &Func) -> Node<'a> {
    let mut nodes = vec![
        Node::Text(buf.as_str(func.name)),
        Node::Text(buf.as_str(func.colon)),
        Node::space(),
        Node::indent_delimited_with(
            buf,
            arena,
            Paren,
            &func.params,
            BreakCond::Width,
            nodify_params,
        ),
        Node::space(),
    ];
    if let Some(ty) = &func.ty {
        nodes.extend([Node::Text("-> "), nodify_ty(buf, arena, ty), Node::space()]);
    }
    nodes.push(nodify_block(buf, arena, &func.block, BreakCond::Always));

    Node::Group(arena.alloc_slice(&nodes))
}

pub fn nodify_funcs<'a>(
    buf: &'a TokenBuffer,
    arena: &BlobArena,
    funcs: &[Func],
) -> Option<Node<'a>> {
    if funcs.is_empty() {
        None
    } else {
        Some(Node::Group(
            arena.alloc_slice(
                &funcs
                    .iter()
                    .enumerate()
                    .flat_map(|(i, f)| {
                        let mut func = Vec::new();
                        check_whitespace(buf, f.name, &mut func);
                        func.push(nodify_func(buf, arena, f));
                        if i != funcs.len() - 1 {
                            func.push(Node::nl());
                        }
                        func
                    })
                    .collect::<Vec<_>>(),
            ),
        ))
    }
}

fn nodify_params<'a>(
    buf: &'a TokenBuffer,
    arena: &BlobArena,
    params: &[Param],
) -> Option<Node<'a>> {
    if params.is_empty() {
        return None;
    }

    Some(Node::set(arena, BreakCond::Width, {
        params.iter().map(|p| match p {
            Param::Slf(t) => {
                vec![Node::token(buf, t)]
            }
            Param::Named {
                name, colon, ty, ..
            } => {
                vec![
                    Node::token(buf, name),
                    Node::token(buf, colon),
                    Node::space(),
                    nodify_ty(buf, arena, &ty),
                ]
            }
        })
    }))
}

fn nodify_ty<'a>(buf: &'a TokenBuffer, arena: &BlobArena, ty: &PType) -> Node<'a> {
    match ty {
        PType::Simple(_, t) => Node::Text(buf.as_str(*t)),
        PType::Ref { inner, .. } => {
            Node::group(arena, &[Node::Text("&"), nodify_ty(buf, arena, inner)])
        }
        PType::Array { size, inner, .. } => Node::group(
            arena,
            &[
                Node::Text("["),
                nodify_ty(buf, arena, inner),
                Node::Text("; "),
                Node::token(buf, size),
                Node::Text("]"),
            ],
        ),
        PType::Slice { inner, .. } => Node::group(
            arena,
            &[
                Node::Text("["),
                nodify_ty(buf, arena, inner),
                Node::Text("]"),
            ],
        ),
    }
}

fn nodify_block<'a>(
    buf: &'a TokenBuffer,
    arena: &BlobArena,
    block: &Block,
    mut condition: BreakCond,
) -> Node<'a> {
    if condition == BreakCond::MoreThanOne {
        if block.stmts.len() > 1 {
            condition = BreakCond::Always;
        } else {
            condition = BreakCond::Width;
        }
    }

    Node::indent_delimited_with(buf, arena, Curly, &block.stmts, condition, nodify_stmts)
}

fn nodify_stmts<'a>(buf: &'a TokenBuffer, arena: &BlobArena, stmts: &[Stmt]) -> Option<Node<'a>> {
    if stmts.is_empty() {
        return None;
    }

    let mut nodes = Vec::with_capacity(stmts.len() * 5);
    for (i, stmt) in stmts.iter().enumerate() {
        match stmt {
            Stmt::Let {
                let_,
                name,
                ty,
                assign,
                ..
            } => {
                check_whitespace(buf, *let_, &mut nodes);
                nodes.extend([Node::Text("let "), Node::token(buf, name)]);

                if let Some(ty) = ty {
                    nodes.extend([Node::Text(": "), nodify_ty(buf, arena, ty)]);
                }

                nodes.extend([
                    Node::Text(" = "),
                    nodify_expr(buf, arena, assign),
                    Node::Text(";"),
                ]);
            }
            Stmt::Semi(expr) => {
                check_whitespace_span(buf, expr.span(buf), &mut nodes);
                nodes.extend([nodify_expr(buf, arena, expr), Node::Text(";")])
            }
            Stmt::Open(expr) => {
                check_whitespace_span(buf, expr.span(buf), &mut nodes);
                nodes.push(nodify_expr(buf, arena, expr))
            }
        }

        if i != stmts.len() - 1 {
            nodes.push(Node::Text("\n"));
        }
    }

    Some(Node::Group(arena.alloc_slice(&nodes)))
}

fn nodify_expr<'a>(buf: &'a TokenBuffer, arena: &BlobArena, expr: &Expr) -> Node<'a> {
    match expr {
        Expr::Break(t)
        | Expr::Continue(t)
        | Expr::Ident(t)
        | Expr::Lit(t)
        | Expr::Str(t)
        | Expr::Bool(t) => Node::token(buf, t),
        Expr::Bin(_, kind, lhs, rhs) => Node::group(
            arena,
            &[
                nodify_expr(buf, arena, lhs),
                Node::FlatText(" "),
                Node::BreakText("\n"),
                Node::BreakText(indent_str(arena, "")),
                Node::Text(kind.as_str()),
                Node::space(),
                nodify_expr(buf, arena, rhs),
            ],
        ),
        Expr::Paren(expr) => Node::group(
            arena,
            &[
                Node::Text("("),
                nodify_expr(buf, arena, expr),
                Node::Text(")"),
            ],
        ),
        Expr::Ret(_, expr) => {
            if let Some(expr) = expr {
                Node::group(
                    arena,
                    &[Node::Text("return "), nodify_expr(buf, arena, expr)],
                )
            } else {
                Node::Text("return")
            }
        }
        Expr::Assign(Assign { kind, lhs, rhs, .. }) => Node::group(
            arena,
            &[
                nodify_expr(buf, arena, lhs),
                Node::space(),
                Node::Text(kind.as_str()),
                Node::space(),
                nodify_expr(buf, arena, rhs),
            ],
        ),
        Expr::StructDef(StructDef { name, fields, .. }) => Node::group(
            arena,
            &[
                Node::token(buf, name),
                Node::space(),
                Node::indent_delimited_with(
                    buf,
                    arena,
                    Curly,
                    fields,
                    BreakCond::Width,
                    nodify_struct_field_defs,
                ),
            ],
        ),
        Expr::Array(def) => match def {
            ArrDef::Elems { exprs, .. } => Node::indent_delimited_with(
                buf,
                arena,
                Bracket,
                exprs,
                BreakCond::Width,
                nodify_expr_set,
            ),
            ArrDef::Repeated { expr, num, .. } => Node::group(
                arena,
                &[
                    Node::Text("["),
                    nodify_expr(buf, arena, expr),
                    Node::Text("; "),
                    nodify_expr(buf, arena, num),
                    Node::Text("]"),
                ],
            ),
        },
        Expr::Access { lhs, field, .. } => Node::group(
            arena,
            &[
                nodify_expr(buf, arena, lhs),
                Node::Text("."),
                Node::token(buf, field),
            ],
        ),
        Expr::IndexOf { array, index, .. } => Node::group(
            arena,
            &[
                nodify_expr(buf, arena, array),
                Node::Text("["),
                nodify_expr(buf, arena, index),
                Node::Text("]"),
            ],
        ),
        Expr::Call { func, args, .. } => Node::group(
            arena,
            &[
                Node::token(buf, func),
                Node::indent_delimited_with(buf, arena, Paren, args, BreakCond::Width, nodify_args),
            ],
        ),
        Expr::MethodCall {
            receiver,
            method,
            args,
            ..
        } => Node::group(
            arena,
            &[
                nodify_method_receiver(buf, arena, receiver),
                match receiver {
                    MethodPath::Field(_) => Node::Text("."),
                    MethodPath::Path(_) => Node::Text("::"),
                },
                Node::token(buf, method),
                Node::indent_delimited_with(buf, arena, Paren, args, BreakCond::Width, nodify_args),
            ],
        ),
        Expr::If {
            condition,
            block,
            otherwise,
            ..
        } => {
            if let Some(otherwise) = otherwise {
                Node::group(
                    arena,
                    &[
                        Node::Text("if "),
                        nodify_expr(buf, arena, condition),
                        Node::space(),
                        nodify_block(buf, arena, block, BreakCond::MoreThanOne),
                        Node::Text(" else "),
                        nodify_block(buf, arena, otherwise, BreakCond::MoreThanOne),
                    ],
                )
            } else {
                Node::group(
                    arena,
                    &[
                        Node::Text("if "),
                        nodify_expr(buf, arena, condition),
                        Node::space(),
                        nodify_block(buf, arena, block, BreakCond::MoreThanOne),
                    ],
                )
            }
        }
        Expr::For {
            iter,
            iterable,
            block,
            ..
        } => Node::group(
            arena,
            &[
                Node::Text("for "),
                Node::token(buf, iter),
                Node::Text(" in "),
                nodify_expr(buf, arena, iterable),
                Node::space(),
                nodify_block(buf, arena, block, BreakCond::Always),
            ],
        ),
        Expr::Range {
            start,
            end,
            inclusive,
            ..
        } => {
            let mut nodes = Vec::with_capacity(4);
            if let Some(start) = start {
                nodes.push(nodify_expr(buf, arena, start));
            }

            nodes.push(Node::Text(".."));
            if *inclusive {
                nodes.push(Node::Text("="));
            }

            if let Some(end) = end {
                nodes.push(nodify_expr(buf, arena, end));
            }

            Node::group(arena, &nodes)
        }
        Expr::Cast { lhs, ty, .. } => Node::group(
            arena,
            &[
                nodify_expr(buf, arena, lhs),
                Node::Text(" as "),
                nodify_ty(buf, arena, ty),
            ],
        ),
        Expr::Loop { block, .. } => Node::group(
            arena,
            &[
                Node::Text("loop "),
                nodify_block(buf, arena, block, BreakCond::Always),
            ],
        ),
        Expr::While {
            condition, block, ..
        } => Node::group(
            arena,
            &[
                Node::Text("while "),
                nodify_expr(buf, arena, condition),
                Node::space(),
                nodify_block(buf, arena, block, BreakCond::MoreThanOne),
            ],
        ),
        Expr::Unary(_, _, kind, expr) => {
            if kind.is_prefix() {
                Node::group(
                    arena,
                    &[Node::Text(kind.as_str()), nodify_expr(buf, arena, expr)],
                )
            } else {
                Node::group(
                    arena,
                    &[nodify_expr(buf, arena, expr), Node::Text(kind.as_str())],
                )
            }
        }
    }
}

fn nodify_method_receiver<'a>(
    buf: &'a TokenBuffer,
    arena: &BlobArena,
    receiver: &MethodPath,
) -> Node<'a> {
    match receiver {
        MethodPath::Field(expr) => nodify_expr(buf, arena, expr),
        MethodPath::Path(path) => nodify_path(buf, arena, path),
    }
}

fn nodify_path<'a>(buf: &'a TokenBuffer, arena: &BlobArena, path: &Path) -> Node<'a> {
    assert!(!path.segments.is_empty());
    let mut nodes = path
        .segments
        .iter()
        .flat_map(|t| [Node::token(buf, t), Node::Text("::")])
        .collect::<Vec<_>>();
    nodes.pop();
    Node::group(arena, &nodes)
}

fn nodify_struct_fields<'a>(
    buf: &'a TokenBuffer,
    arena: &BlobArena,
    fields: &[Field],
) -> Option<Node<'a>> {
    if fields.is_empty() {
        None
    } else {
        Some(Node::set(
            arena,
            BreakCond::Always,
            fields.iter().map(|f| {
                let mut fields = Vec::new();
                check_whitespace(buf, f.name, &mut fields);
                fields.extend([
                    Node::token(buf, f.name),
                    Node::Text(": "),
                    nodify_ty(buf, arena, &f.ty),
                ]);
                fields
            }),
        ))
    }
}

fn nodify_struct_field_defs<'a>(
    buf: &'a TokenBuffer,
    arena: &BlobArena,
    fields: &[FieldDef],
) -> Option<Node<'a>> {
    if fields.is_empty() {
        None
    } else {
        Some(Node::set(
            arena,
            BreakCond::Width,
            fields.iter().map(|f| {
                let mut fields = Vec::new();
                check_whitespace(buf, f.name, &mut fields);
                fields.extend([
                    Node::token(buf, f.name),
                    Node::Text(": "),
                    nodify_expr(buf, arena, &f.expr),
                ]);
                fields
            }),
        ))
    }
}

fn nodify_expr_set<'a>(
    buf: &'a TokenBuffer,
    arena: &BlobArena,
    exprs: &[Expr],
) -> Option<Node<'a>> {
    if exprs.is_empty() {
        None
    } else {
        Some(Node::set(
            arena,
            BreakCond::Width,
            exprs.iter().map(|expr| vec![nodify_expr(buf, arena, expr)]),
        ))
    }
}

fn nodify_args<'a>(buf: &'a TokenBuffer, arena: &BlobArena, args: &[Expr]) -> Option<Node<'a>> {
    if args.is_empty() {
        None
    } else {
        Some(Node::set(
            arena,
            BreakCond::Width,
            args.iter().map(|expr| vec![nodify_expr(buf, arena, expr)]),
        ))
    }
}

pub fn nodify_struct<'a>(buf: &'a TokenBuffer, arena: &BlobArena, strukt: &Struct) -> Node<'a> {
    Node::group(
        arena,
        &[
            Node::token(buf, strukt.name),
            Node::Text(": struct "),
            Node::indent_delimited_with(
                buf,
                arena,
                Curly,
                &strukt.fields,
                BreakCond::Always,
                nodify_struct_fields,
            ),
        ],
    )
}

pub fn nodify_const<'a>(buf: &'a TokenBuffer, arena: &BlobArena, konst: &Const) -> Node<'a> {
    Node::group(
        arena,
        &[
            Node::token(buf, konst.name),
            Node::Text(": const "),
            nodify_ty(buf, arena, &konst.ty),
            Node::Text(" = "),
            Node::BreakText("\n"),
            nodify_expr(buf, arena, &konst.expr),
            Node::Text(";"),
        ],
    )
}

pub fn nodify_attr<'a>(buf: &'a TokenBuffer, arena: &BlobArena, attr: &Attribute) -> Node<'a> {
    let mut nodes = vec![Node::Text("#[")];
    nodes.extend(attr.tokens.iter().map(|t| Node::token(buf, *t)));
    nodes.push(Node::Text("]"));
    Node::group(arena, &nodes)
}

pub fn nodify_extern<'a>(
    buf: &'a TokenBuffer,
    arena: &BlobArena,
    exturn: &ExternBlock,
) -> Node<'a> {
    Node::group(
        arena,
        &[
            Node::Text("extern("),
            Node::token(buf, exturn.convention),
            Node::Text(") "),
            Node::indent_delimited_with(
                buf,
                arena,
                Curly,
                &exturn.funcs,
                BreakCond::Always,
                nodify_extern_funcs,
            ),
        ],
    )
}

fn nodify_extern_funcs<'a>(
    buf: &'a TokenBuffer,
    arena: &BlobArena,
    funcs: &[ExternFunc],
) -> Option<Node<'a>> {
    if funcs.is_empty() {
        return None;
    }

    let mut nodes = Vec::new();
    for (i, func) in funcs.iter().enumerate() {
        check_whitespace(buf, func.name, &mut nodes);
        nodes.extend([
            Node::Text(buf.as_str(func.name)),
            Node::Text(": "),
            Node::indent_delimited_with(
                buf,
                arena,
                Paren,
                &func.params,
                BreakCond::Width,
                nodify_params,
            ),
        ]);
        if let Some(ty) = &func.ty {
            nodes.extend([Node::Text(" -> "), nodify_ty(buf, arena, ty)]);
        }

        if i != funcs.len() - 1 {
            nodes.push(Node::Text(";\n"));
        } else {
            nodes.push(Node::Text(";"));
        }
    }

    Some(Node::group(arena, &nodes))
}

pub fn nodify_use<'a>(buf: &'a TokenBuffer, arena: &BlobArena, uze: &Use) -> Node<'a> {
    let mut path = vec![Node::Text("use ")];
    path.push(nodify_path(buf, arena, &uze.path));
    path.push(Node::Text(";"));

    Node::group(arena, &path)
}

pub fn nodify_impl<'a>(buf: &'a TokenBuffer, arena: &BlobArena, impul: &Impl) -> Node<'a> {
    Node::group(
        arena,
        &[
            Node::Text("impl "),
            nodify_ty(buf, arena, &impul.ty),
            Node::space(),
            Node::indent_delimited_with(
                buf,
                arena,
                Curly,
                &impul.funcs,
                BreakCond::Always,
                nodify_funcs,
            ),
        ],
    )
}
