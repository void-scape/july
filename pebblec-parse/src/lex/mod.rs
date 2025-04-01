use buffer::{Span, Token, TokenBuffer};
use kind::TokenKind;
use source::Source;
use winnow::ascii::float;
use winnow::combinator::{delimited, opt, preceded};
use winnow::error::{ContextError, ErrMode};
use winnow::stream::Stream;
use winnow::token::any;
use winnow::{
    LocatingSlice, ModalResult, Parser, combinator::alt, stream::AsChar, token::take_while,
};

pub mod buffer;
pub mod io;
pub mod kind;
pub mod source;

pub struct Lexer {
    source: Source,
}

impl Lexer {
    pub fn new(source: Source) -> Self {
        Self { source }
    }

    pub fn lex<'a>(self) -> ModalResult<TokenBuffer> {
        let mut tokens = Vec::new();
        let mut input = LocatingSlice::new(self.source.source.as_str());

        while !input.is_empty() {
            take_while(.., |c| c == ' ' || c == '\n').parse_next(&mut input)?;
            if input.is_empty() {
                break;
            }

            match any_token(&mut input) {
                Some(token) => tokens.push(token?),
                None => {}
            }
        }

        Ok(TokenBuffer::new(tokens, self.source))
    }
}

fn comment<'a>(input: &mut LocatingSlice<&'a str>) -> ModalResult<()> {
    loop {
        take_while(.., |c| c == ' ' || c == '\n').parse_next(input)?;
        match alt(("//", "/*")).parse_next(input)? {
            "//" => {
                while input.peek_token().is_some_and(|(_, c)| c != '\n') {
                    _ = input.next_token();
                }
            }
            "/*" => {
                while input.peek_slice(2).1 != "*/" {
                    _ = input.next_token();
                }
                _ = (input.next_token(), input.next_token());
            }
            _ => unreachable!(),
        }
    }
}

fn any_token<'a>(input: &mut LocatingSlice<&'a str>) -> Option<ModalResult<Token>> {
    _ = comment(input);
    if input.is_empty() {
        return None;
    }

    let token = alt((str_lit, symbols, delim, int_lit, keyword_ident)).parse_next(input);
    Some(token)
}

fn str_lit<'a>(input: &mut LocatingSlice<&'a str>) -> ModalResult<Token> {
    let ((_, inner_span), _span) = delimited("\"", take_while(.., |c| c != '\"').with_span(), "\"")
        .with_span()
        .parse_next(input)?;
    // TODO: include the `"`
    Ok(Token::new(TokenKind::Str, Span::from_range(inner_span)))
}

const SYMBOL_TABLE: [TokenKind; 256] = symbol_table();
pub const SYMBOLS: [(char, TokenKind); 14] = [
    (';', TokenKind::Semi),
    (':', TokenKind::Colon),
    ('=', TokenKind::Equals),
    ('+', TokenKind::Plus),
    ('*', TokenKind::Asterisk),
    ('%', TokenKind::Percent),
    ('/', TokenKind::Slash),
    ('-', TokenKind::Hyphen),
    (',', TokenKind::Comma),
    ('#', TokenKind::Pound),
    ('&', TokenKind::Ampersand),
    ('!', TokenKind::Bang),
    ('^', TokenKind::Caret),
    ('|', TokenKind::Pipe),
];
const DUMMY_SYM: TokenKind = TokenKind::Ident;

const fn symbol_table() -> [TokenKind; 256] {
    let mut table = [DUMMY_SYM; 256];
    let mut i = 0;
    while i < SYMBOLS.len() {
        table[SYMBOLS[i].0 as usize] = SYMBOLS[i].1;
        i += 1;
    }
    table
}

fn symbols<'a>(input: &mut LocatingSlice<&'a str>) -> ModalResult<Token> {
    if input.peek_token().is_some_and(|(_, t)| {
        SYMBOL_TABLE
            .get(t as usize)
            .is_some_and(|s| *s != DUMMY_SYM)
    }) {
        let (sym, span) = any.with_span().parse_next(input)?;
        Ok(Token::new(
            SYMBOL_TABLE[sym as usize],
            Span::from_range(span),
        ))
    } else {
        if let Some((_, span)) = opt('.'.with_span()).parse_next(input)? {
            if input.peek_token().is_some_and(|(_, t)| t == '.') {
                _ = input.next_token();
                return Ok(Token::new(
                    TokenKind::DoubleDot,
                    Span::from_range(span.start..span.end + 1),
                ));
            } else {
                Ok(Token::new(TokenKind::Dot, Span::from_range(span)))
            }
        } else {
            ModalResult::Err(ErrMode::Backtrack(ContextError::new()))
        }
    }
}

const DELIM_TABLE: [TokenKind; 256] = delim_table();
const DELIMS: [(char, TokenKind); 8] = [
    ('<', TokenKind::OpenAngle),
    ('>', TokenKind::CloseAngle),
    ('{', TokenKind::OpenCurly),
    ('}', TokenKind::CloseCurly),
    ('(', TokenKind::OpenParen),
    (')', TokenKind::CloseParen),
    ('[', TokenKind::OpenBracket),
    (']', TokenKind::CloseBracket),
];
const DUMMY_DELIM: TokenKind = TokenKind::Ident;

const fn delim_table() -> [TokenKind; 256] {
    let mut table = [DUMMY_DELIM; 256];
    let mut i = 0;
    while i < DELIMS.len() {
        table[DELIMS[i].0 as usize] = DELIMS[i].1;
        i += 1;
    }
    table
}

fn delim<'a>(input: &mut LocatingSlice<&'a str>) -> ModalResult<Token> {
    if input.peek_token().is_some_and(|(_, t)| {
        DELIM_TABLE
            .get(t as usize)
            .is_some_and(|d| *d != DUMMY_DELIM)
    }) {
        let (sym, span) = any.with_span().parse_next(input)?;
        Ok(Token::new(
            DELIM_TABLE[sym as usize],
            Span::from_range(span),
        ))
    } else {
        ModalResult::Err(ErrMode::Backtrack(ContextError::new()))
    }
}

fn int_lit<'a>(input: &mut LocatingSlice<&'a str>) -> ModalResult<Token> {
    let other = *input;
    let (mut kind, span) = alt((
        preceded("0x", take_while(1.., AsChar::is_hex_digit)).map(|_| TokenKind::Int),
        preceded("0b", take_while(1.., AsChar::is_hex_digit)).map(|_| TokenKind::Int),
        float.map(|_: f64| TokenKind::Float),
    ))
    .with_span()
    .parse_next(input)?;

    if matches!(kind, TokenKind::Float) {
        let slice = other.get(..span.len());

        // TODO: get rid of winnow and parse manually, this is annoying
        if slice == Some("infinity") || slice == Some("nan") {
            kind = TokenKind::Ident;
        } else {
            if slice.is_some_and(|s| s.parse::<i64>().is_ok()) {
                return Ok(Token::new(TokenKind::Int, Span::from_range(span)));
            } else if slice.is_some_and(|s| s.ends_with('.'))
                && input.peek_token().is_some_and(|(_, t)| t == '.')
            {
                *input = other;
                take_while(1.., AsChar::is_hex_digit).parse_next(input)?;

                return Ok(Token::new(
                    TokenKind::Int,
                    Span::from_range(span.start..span.end - 1),
                ));
            }
        }
    }

    Ok(Token::new(kind, Span::from_range(span)))
}

// This is 4% slower than linear search with fxhash
//
//static KEYWORD_TABLE: LazyLock<HashMap<&'static str, TokenKind>> =
//    LazyLock::new(|| KEYWORDS.iter().cloned().collect());
//const KEYWORDS: [(&str, TokenKind); 19] = [
//    ("let", TokenKind::Let),
//    ("if", TokenKind::If),
//    ("else", TokenKind::Else),
//    ("true", TokenKind::True),
//    ("false", TokenKind::False),
//    ("for", TokenKind::For),
//    ("in", TokenKind::In),
//    ("fn", TokenKind::Fn),
//    ("self", TokenKind::Slf),
//    ("return", TokenKind::Ret),
//    ("continue", TokenKind::Continue),
//    ("break", TokenKind::Break),
//    ("as", TokenKind::As),
//    ("struct", TokenKind::Struct),
//    ("impl", TokenKind::Impl),
//    ("enum", TokenKind::Enum),
//    ("loop", TokenKind::Loop),
//    ("const", TokenKind::Const),
//    ("extern", TokenKind::Extern),
//];

fn keyword_ident<'a>(input: &mut LocatingSlice<&'a str>) -> ModalResult<Token> {
    let (result, span) = take_while(1.., |c: char| {
        !c.is_whitespace()
            && c != '.'
            && SYMBOL_TABLE
                .get(c as usize)
                .is_some_and(|s| *s == DUMMY_SYM)
            && DELIM_TABLE
                .get(c as usize)
                .is_some_and(|d| *d == DUMMY_DELIM)
    })
    .with_span()
    .parse_next(input)?;

    let token = match result {
        "let" => TokenKind::Let,
        "if" => TokenKind::If,
        "else" => TokenKind::Else,
        "true" => TokenKind::True,
        "false" => TokenKind::False,
        "for" => TokenKind::For,
        "in" => TokenKind::In,
        "fn" => TokenKind::Fn,
        "self" => TokenKind::Slf,
        "return" => TokenKind::Ret,
        "continue" => TokenKind::Continue,
        "break" => TokenKind::Break,
        "as" => TokenKind::As,
        "struct" => TokenKind::Struct,
        "impl" => TokenKind::Impl,
        "enum" => TokenKind::Enum,
        "loop" => TokenKind::Loop,
        "const" => TokenKind::Const,
        "use" => TokenKind::Use,
        "extern" => TokenKind::Extern,
        _ => TokenKind::Ident,
    };

    Ok(Token::new(token, Span::from_range(span)))
}
