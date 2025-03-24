use crate::node::*;
use pebblec_arena::BlobArena;
use pebblec_parse::lex::buffer::{TokenBuffer, TokenId, TokenQuery};
use pebblec_parse::lex::source::{Source, SourceMap};
use pebblec_parse::lex::{Lexer, io};
use pebblec_parse::{Item, ItemKind};
use std::borrow::Borrow;
use std::ops::Deref;
use std::path::Path;

// TODO: add config
const COLS: usize = 80;

pub fn fmt<P: AsRef<Path>>(path: P) -> Result<Option<String>, std::io::Error> {
    let str = io::read_string(path)?;
    Ok(fmt_string(str))
}

pub fn fmt_string(str: String) -> Option<String> {
    let len = str.len();
    let source = Source::from_string("pebble-fmt", str);
    let buf = Lexer::new(source).lex().ok()?;
    let items = pebblec_parse::parse(&buf).ok()?;
    let arena = BlobArena::default();

    let mut nodes = Vec::new();
    for item in items.iter() {
        match &item.kind {
            ItemKind::Func(func) => {
                check_whitespace(&buf, func.name, &mut nodes);
                nodes.extend([nodify_func(&buf, &arena, func), Node::nl()])
            }
            ItemKind::Struct(strukt) => {
                check_whitespace(&buf, strukt.name, &mut nodes);
                nodes.extend([nodify_struct(&buf, &arena, strukt), Node::nl()])
            }
            ItemKind::Const(konst) => {
                check_whitespace(&buf, konst.name, &mut nodes);
                nodes.extend([nodify_const(&buf, &arena, konst), Node::nl()])
            }
            ItemKind::Attr(attr) => {
                check_whitespace(&buf, attr.pound, &mut nodes);
                nodes.extend([nodify_attr(&buf, &arena, attr), Node::nl()]);
            }
            ItemKind::Extern(exturn) => {
                check_whitespace(&buf, exturn.exturn, &mut nodes);
                nodes.extend([nodify_extern(&buf, &arena, exturn), Node::nl()]);
            }
            ItemKind::Use(uze) => {
                check_whitespace(&buf, uze.uze, &mut nodes);
                nodes.extend([nodify_use(&buf, &arena, uze), Node::nl()]);
            }
            _ => {} //_ => todo!(),
        }
    }

    let mut output = String::with_capacity(len);
    let node = Node::Group(arena.alloc_slice(&nodes));
    fmt_tree(&node, &mut 0, true, &mut output);
    while output.ends_with("\n") {
        output.pop();
    }

    // TODO: This is a dirty solution, this should really be possible during the formatting itself.
    let mut cleaned = String::with_capacity(output.len());
    let mut prev_blank = false;

    for line in output.lines() {
        let is_blank = line.trim().is_empty();

        if is_blank && prev_blank {
            continue;
        }

        if !cleaned.is_empty() {
            cleaned.push('\n');
        }
        cleaned.push_str(line);
        prev_blank = is_blank;
    }

    Some(cleaned)
}

fn fmt_tree(node: &Node<'_>, indent: &mut usize, flat: bool, output: &mut String) {
    match node {
        Node::Group(nodes) => {
            let flat = node.flat_width() + *indent <= COLS;
            nodes.iter().for_each(|n| fmt_tree(n, indent, flat, output))
        }
        Node::Indent(node, add_indent) => {
            *indent += add_indent;
            fmt_tree(node, indent, flat, output);
            *indent -= add_indent;
        }
        Node::Text(text) => {
            if output.ends_with("\n") {
                (0..*indent).map(|_| ' ').for_each(|c| output.push(c));
            }
            output.push_str(text);
        }
        Node::FlatText(text) => {
            if flat {
                if output.ends_with("\n") {
                    (0..*indent).map(|_| ' ').for_each(|c| output.push(c));
                }
                output.push_str(text);
            }
        }
        Node::BreakText(text) => {
            if !flat {
                if output.ends_with("\n") {
                    (0..*indent).map(|_| ' ').for_each(|c| output.push(c));
                }
                output.push_str(text);
            }
        }
    }
}
