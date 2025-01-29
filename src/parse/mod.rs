use crate::diagnostic::{Diag, Msg};
use crate::ir::prelude::*;
use crate::lex::{buffer::*, kind::*};
use combinator::prelude::*;
use instr::*;
use matc::*;
use rule::*;

pub mod combinator;
pub mod instr;
pub mod matc;
pub mod rule;

pub const PARSE_ERR: &'static str = "failed to parse";

pub type JulyRules = Loop<FnRules>;

/// fn <ident>() [:: <type>] <block>
#[derive(Default)]
pub struct FnRules {
    rules: StackTrack<
        Seq<(
            // `fn <ident>() [:: <type>] <block>`
            //  ^^
            Next<Fn>,
            // `fn <ident>() [:: <type>] <block>`
            //      ^^^^^
            Next<Ident, Push>,
            // `fn <ident>() [:: <type>] <block>`
            //            ^
            Next<OpenParen>,
            // `fn <ident>() [:: <type>] <block>`
            //             ^
            Next<CloseParen>,
            // `fn <ident>() [:: <type>] <block>`
            //                ^^^^^^^^^
            Opt<Seq<(Next<Colon>, Next<Colon>, Next<Ident, Push>)>>,
            // `fn <ident>() [:: <type>] <block>`
            //                            ^^^^^
            BlockRules,
        )>,
    >,
    start: Option<Span>,
    sig: Option<Sig>,
}

#[derive(Debug, Clone, Copy)]
struct Sig {
    name: TokenId,
    ret: Ty,
}

impl<'a> ParserRule<'a> for FnRules {
    fn apply(
        &mut self,
        buffer: &'a TokenBuffer<'a>,
        token: TokenId,
        stack: &mut Vec<TokenId>,
        ctx: &mut Ctx<'a>,
    ) -> RuleResult<'a> {
        if self.start.is_none() {
            self.start = Some(buffer.span(token));
        }

        let result = self
            .rules
            .apply(buffer, token, stack, ctx)
            .inspect_diag(|diag| diag.msg(Msg::note(self.start.unwrap(), "while parsing this fn")));

        if self.rules.index() == 5 && !result.failed() && self.sig.is_none() {
            match self.rules.pushed() {
                // `fn <ident>() [:: <type>] <block>`
                //      ^^^^^
                1 => {
                    self.sig = Some(Sig {
                        name: stack.pop().unwrap(),
                        ret: Ty::Void,
                    });
                }
                // `fn <ident>() [:: <type>] <block>`
                //      ^^^^^         ^^^^
                2 => {
                    let ret = ctx.ty(stack.pop().unwrap());
                    self.sig = Some(Sig {
                        name: stack.pop().unwrap(),
                        ret,
                    })
                }
                n => {
                    panic!("unexpected number of stack elements {n}");
                }
            }
        }

        if self.rules.finished() {
            let block = &self.rules.rules().5;
            assert!(block.finished());

            let sig = self.sig.expect("func sig");
            let name = ctx.store_ident(sig.name);
            let _ = ctx.store_func(Func::new(name, Vec::new(), sig.ret, block.id()));
            println!("{:#?}", ctx);
        }

        result
    }

    fn finished(&self) -> bool {
        self.rules.finished()
    }

    fn reset(&mut self) {
        *self = Self::default();
    }
}

/// `{ <[stmt]> }`
#[derive(Default)]
pub struct BlockRules {
    rules: Seq<(
        // `{ <[stmt]> }`
        //  ^
        Next<OpenCurly>,
        // `{ <[stmt]> }`
        //     ^^^^^^
        While<NextToken<Not<CloseCurly>>, Alt<(LetStmtRules, ReturnRules)>>,
        // `{ <[stmt]> }`
        //             ^
        Next<CloseCurly>,
    )>,
    start: Option<Span>,
    inner: Option<(Span, BlockId)>,
}

impl BlockRules {
    pub fn span(&self) -> Span {
        self.inner
            .map(|(span, _)| span)
            .expect("checked rules finished")
    }

    pub fn id(&self) -> BlockId {
        self.inner
            .map(|(_, id)| id)
            .expect("checked rules finished")
    }

    pub fn finished(&self) -> bool {
        self.rules.finished()
    }
}

impl<'a> ParserRule<'a> for BlockRules {
    fn apply(
        &mut self,
        buffer: &'a TokenBuffer<'a>,
        token: TokenId,
        stack: &mut Vec<TokenId>,
        ctx: &mut Ctx<'a>,
    ) -> RuleResult<'a> {
        if self.start.is_none() {
            self.start = Some(buffer.span(token));
        }

        let result = self
            .rules
            .apply(buffer, token, stack, ctx)
            .inspect_diag(|diag| {
                diag.msg(Msg::note(self.start.unwrap(), "while parsing this block"))
            });

        if self.rules.finished() {
            let end = buffer.span(token);
            let span = Span::from_range(self.start.unwrap().start as usize..end.end as usize);
            let id = ctx.store_block(Block::new(span, Vec::new(), Ty::Void));
            self.inner = Some((span, id));
        }

        result
    }

    fn finished(&self) -> bool {
        self.rules.finished()
    }

    fn reset(&mut self) {
        *self = Self::default();
    }
}

/// `return <expr>;`
pub type ReturnRules = Seq<(
    Next<Ret>,
    Alt<(Next<Ident>, ExprRule, Next<Int>)>,
    Next<Semi>,
)>;

/// `let <ident>[: <type>] = <expr>;`
pub type LetStmtRules = Seq<(
    // `let x = 1;`
    //  ^
    Next<Let>,
    // `let x = 1;`
    //      ^
    Expect<Let, Ident>,
    // `let x: i32 = 1;`
    //       ^^^^^
    Opt<Seq<(Rule<Ident, Colon, Push>, Rule<Colon, Ident, (Push, Swap)>)>>,
    // `let x = 1;`
    //        ^
    Next<Equals>,
    ExprRule,
    // `let x = 1;`
    //           ^
    Next<Semi>,
)>;

/// `let x = 1 + 2;`
///          ^^^^^
pub type ExprRule = While<NextToken<Not<Semi>>, Stateful<PushBinaryOp>>;

#[derive(Default)]
pub struct PushBinaryOp;

impl StatefulParserRule for PushBinaryOp {
    type State = Vec<TokenId>;

    fn init_state(&mut self) -> Self::State {
        Vec::new()
    }

    fn apply_with<'a>(
        &mut self,
        operators: &mut Self::State,
        buffer: &'a TokenBuffer<'a>,
        token: TokenId,
        stack: &mut Vec<TokenId>,
        _ctx: &mut Ctx<'a>,
    ) -> RuleResult<'a> {
        let kind = buffer.kind(token);
        match kind {
            TokenKind::Int => stack.push(token),
            TokenKind::Plus | TokenKind::Asterisk => {
                while operators
                    .last()
                    .is_some_and(|t| buffer.kind(*t).precedence() > kind.precedence())
                {
                    stack.push(operators.pop().unwrap());
                }
                operators.push(token)
            }
            t => {
                return RuleResult::Failed(Diag::sourced(
                    PARSE_ERR,
                    buffer.source(),
                    Msg::error(
                        buffer.span(token),
                        format!("cannot use {:?} in binary op", t),
                    ),
                ));
            }
        }

        if buffer.next(token).is_some_and(|t| buffer.is_terminator(t)) {
            stack.extend(operators.drain(..).rev());
        }

        RuleResult::Consume
    }
}

trait Precedence {
    fn precedence(&self) -> usize;
}

impl Precedence for TokenKind {
    fn precedence(&self) -> usize {
        match self {
            Self::Asterisk => 1,
            Self::Plus => 0,
            _ => 0,
        }
    }
}

pub struct Parser<'a> {
    pub post_order: Vec<TokenId>,
    ctx: Ctx<'a>,
}

impl<'a> Parser<'a> {
    pub fn new(buffer: &'a TokenBuffer<'a>, ctx: &mut Ctx<'a>) -> Result<Self, Diag<'a>> {
        Self::build_post_order(buffer, ctx)
    }

    fn build_post_order(buffer: &'a TokenBuffer<'a>, ctx: &mut Ctx<'a>) -> Result<Self, Diag<'a>> {
        let mut stack = Vec::with_capacity(buffer.len());
        let mut token_stream = buffer.tokens();
        let mut rules = JulyRules::default();

        loop {
            let Some(token) = token_stream.next() else {
                println!("finished parsing without finishing ruleset");
                break;
            };

            Self::parse_inner(
                &mut rules,
                &buffer,
                &mut token_stream,
                &mut stack,
                token,
                ctx,
            )?;

            if rules.finished() {
                if token_stream.next().is_some() {
                    panic!("more tokens to parse");
                }

                break;
            }
        }

        //println!(
        //    "stack: {:#?}",
        //    stack.iter().map(|t| buffer.kind(*t)).collect::<Vec<_>>()
        //);

        //assert_eq!(
        //    stack.len(),
        //    buffer.len(),
        //    "input token buffer len is not equal to output tree len"
        //);

        Ok(Self {
            post_order: stack,
            ctx: Ctx::new(buffer),
        })
    }

    fn parse_inner<C: ParserRule<'a>>(
        rules: &mut C,
        buffer: &'a TokenBuffer<'a>,
        token_stream: &mut impl Iterator<Item = TokenId>,
        stack: &mut Vec<TokenId>,
        token: TokenId,
        ctx: &mut Ctx<'a>,
    ) -> Result<(), Diag<'a>> {
        match rules.apply(buffer, token, stack, ctx) {
            RuleResult::Continue => {
                if rules.finished() {
                    return Ok(());
                }

                Self::parse_inner(rules, buffer, token_stream, stack, token, ctx)
            }
            RuleResult::Failed(diag) => Err(diag),
            _ => Ok(()),
        }
    }
}
