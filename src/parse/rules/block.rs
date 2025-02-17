use super::{Next, ParserRule, RResult};
use crate::lex::{buffer::*, kind::*};
use crate::parse::stream::TokenStream;
use crate::parse::{combinator::prelude::*, matc::*, rules::prelude::*};

/// Collection of [`Stmt`]s.
#[derive(Debug)]
pub struct Block {
    pub span: Span,
    pub stmts: Vec<Stmt>,
}

/// `{ <[stmt]>[, ...] }`
#[derive(Default)]
pub struct BlockRules;

impl<'a> ParserRule<'a> for BlockRules {
    type Output = Block;

    fn parse(
        buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        stack: &mut Vec<TokenId>,
    ) -> RResult<'a, Self::Output> {
        match Spanned::<(
            // `{ <[stmt]> }`
            //  ^
            Next<OpenCurly>,
            // `{ <[stmt]> }`
            //     ^^^^^^
            StmtSeqRules,
            // `{ <[stmt]> }`
            //             ^
            Next<CloseCurly>,
        )>::parse(buffer, stream, stack)
        {
            Ok(block) => {
                let span = block.span();
                let (_, stmts, _) = block.into_inner();
                Ok(Block { span, stmts })
            }
            Err(e) => {
                stream.eat_until_consume::<CloseCurly>();
                Err(e)
            }
        }
    }
}

/// `{ let x = 2; ret x; ... }`
/// 1--^^^^^^^^^^
/// 2-------------^^^^^^
/// n
#[derive(Debug, Default)]
pub struct StmtSeqRules;

impl<'a> ParserRule<'a> for StmtSeqRules {
    type Output = Vec<Stmt>;

    fn parse(
        buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        stack: &mut Vec<TokenId>,
    ) -> RResult<'a, Self::Output> {
        let mut instrs = Vec::new();

        while !stream.match_peek::<CloseCurly>() {
            let instr = match stream.peek_kind().unwrap() {
                TokenKind::Let => LetRule::parse(buffer, stream, stack)?,
                _ => StmtRule::parse(buffer, stream, stack)?,
            };

            instrs.push(instr);
        }

        Ok(instrs)
    }
}
