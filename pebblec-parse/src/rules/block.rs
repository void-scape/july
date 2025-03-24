use super::{Next, ParserRule, RResult};
use crate::lex::buffer::*;
use crate::lex::kind::*;
use crate::stream::TokenStream;
use crate::{combinator::prelude::*, matc::*, rules::prelude::*};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    pub span: Span,
    pub stmts: Vec<Stmt>,
}

#[derive(Default)]
pub struct BlockRules;

impl<'a, 's> ParserRule<'a, 's> for BlockRules {
    type Output = Block;

    fn parse(stream: &mut TokenStream<'a, 's>) -> RResult<'s, Self::Output> {
        if !stream.match_peek::<OpenCurly>() {
            return Err(stream.recover("expected `{`"));
        }

        let chk = *stream;
        match Spanned::<(Next<OpenCurly>, StmtSeqRules, Next<CloseCurly>)>::parse(stream) {
            Ok(block) => {
                let span = block.span();
                let (_open, stmts, _close) = block.into_inner();
                Ok(Block { span, stmts })
            }
            Err(e) => {
                *stream = chk;
                stream.consume_matched_delimiters_inclusive::<Curly>();
                Err(e.fail())
            }
        }
    }
}

#[derive(Debug, Default)]
pub struct StmtSeqRules;

impl<'a, 's> ParserRule<'a, 's> for StmtSeqRules {
    type Output = Vec<Stmt>;

    fn parse(stream: &mut TokenStream<'a, 's>) -> RResult<'s, Self::Output> {
        let mut instrs = Vec::new();

        while !stream.match_peek::<CloseCurly>() {
            instrs.push(StmtRule::parse(stream)?);
        }

        Ok(instrs)
    }
}
