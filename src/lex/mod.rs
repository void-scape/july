use crate::unit::source::Source;
use buffer::{Span, Token, TokenBuffer};
use kind::TokenKind;
use winnow::combinator::{delimited, preceded};
use winnow::stream::Stream;
use winnow::{combinator::alt, stream::AsChar, token::take_while, LocatingSlice, PResult, Parser};

pub mod buffer;
pub mod kind;

pub struct Lexer<'a> {
    source: &'a Source,
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a Source) -> Self {
        Self { source }
    }

    pub fn lex(self) -> PResult<TokenBuffer<'a>> {
        let mut tokens = Vec::new();
        let mut input = LocatingSlice::new(self.source.raw());

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

fn comment<'a>(input: &mut LocatingSlice<&'a str>) -> PResult<()> {
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

fn any_token<'a>(input: &mut LocatingSlice<&'a str>) -> Option<PResult<Token>> {
    _ = comment(input);
    if input.is_empty() {
        return None;
    }

    let token = alt((str_lit, symbols, delim, int_lit, keyword_ident)).parse_next(input);
    Some(token)
}

fn str_lit<'a>(input: &mut LocatingSlice<&'a str>) -> PResult<Token> {
    let ((_, inner_span), _span) = delimited("\"", take_while(.., |c| c != '\"').with_span(), "\"")
        .with_span()
        .parse_next(input)?;
    // TODO: include the `"`
    Ok(Token::new(TokenKind::Str, Span::from_range(inner_span)))
}

fn symbols<'a>(input: &mut LocatingSlice<&'a str>) -> PResult<Token> {
    let (sym, span) = alt((".", ",", ";", "+", "*", "=", ":", "-", ">", "#", "&"))
        .with_span()
        .parse_next(input)?;
    let token = match sym {
        "." => TokenKind::Dot,
        "," => TokenKind::Comma,
        ";" => TokenKind::Semi,
        "+" => TokenKind::Plus,
        "*" => TokenKind::Asterisk,
        "=" => TokenKind::Equals,
        ":" => TokenKind::Colon,
        "-" => TokenKind::Hyphen,
        ">" => TokenKind::Greater,
        "#" => TokenKind::Pound,
        "&" => TokenKind::Ampersand,
        _ => unreachable!(),
    };

    Ok(Token::new(token, Span::from_range(span)))
}

fn delim<'a>(input: &mut LocatingSlice<&'a str>) -> PResult<Token> {
    let (delim, span) = alt(('{', '}', '(', ')', '[', ']'))
        .with_span()
        .parse_next(input)?;
    let token = match delim {
        '{' => TokenKind::OpenCurly,
        '}' => TokenKind::CloseCurly,
        '(' => TokenKind::OpenParen,
        ')' => TokenKind::CloseParen,
        '[' => TokenKind::OpenBracket,
        ']' => TokenKind::CloseBracket,
        _ => unreachable!(),
    };

    Ok(Token::new(token, Span::from_range(span)))
}

fn int_lit<'a>(input: &mut LocatingSlice<&'a str>) -> PResult<Token> {
    let (_, span) = alt((
        preceded("0x", take_while(1.., AsChar::is_dec_digit)),
        take_while(1.., AsChar::is_dec_digit),
    ))
    .with_span()
    .parse_next(input)?;

    Ok(Token::new(TokenKind::Int, Span::from_range(span)))
}

fn keyword_ident<'a>(input: &mut LocatingSlice<&'a str>) -> PResult<Token> {
    let (result, span) = take_while(1.., |c: char| {
        !c.is_whitespace()
            && c != '.'
            && c != ','
            && c != ':'
            && c != ';'
            && c != '('
            && c != ')'
            && c != '{'
            && c != '}'
            && c != '['
            && c != ']'
    })
    .with_span()
    .parse_next(input)?;

    let token = match result {
        "let" => TokenKind::Let,
        "if" => TokenKind::If,
        "else" => TokenKind::Else,
        "true" => TokenKind::True,
        "false" => TokenKind::False,
        "return" => TokenKind::Ret,
        "fn" => TokenKind::Fn,
        "struct" => TokenKind::Struct,
        "enum" => TokenKind::Enum,
        "loop" => TokenKind::Loop,
        "const" => TokenKind::Const,
        "extern" => TokenKind::Extern,
        _ => TokenKind::Ident,
    };

    Ok(Token::new(token, Span::from_range(span)))
}
