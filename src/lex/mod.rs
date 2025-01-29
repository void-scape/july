use crate::source::Source;
use buffer::{Span, Token, TokenBuffer};
use kind::TokenKind;
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
        let mut input = LocatingSlice::new(self.source.as_str());

        while !input.is_empty() {
            take_while(.., |c| c == ' ' || c == '\n').parse_next(&mut input)?;
            if input.is_empty() {
                break;
            }

            tokens.push(any_token(&mut input)?);
        }

        Ok(TokenBuffer::new(tokens, self.source))
    }
}

fn any_token<'a>(input: &mut LocatingSlice<&'a str>) -> PResult<Token> {
    alt((
        ret, semi, plus, asterisk, equals, delim, colon, keyword, int_lit, ident,
    ))
    .parse_next(input)
}

macro_rules! simple_token {
    ($name:ident, $str:tt, $kind:expr) => {
        fn $name<'a>(input: &mut LocatingSlice<&'a str>) -> PResult<Token> {
            let span = $str.span().parse_next(input)?;
            Ok(Token::new($kind, Span::from_range(span)))
        }
    };
}

simple_token!(ret, "return", TokenKind::Ret);
simple_token!(semi, ';', TokenKind::Semi);
simple_token!(plus, '+', TokenKind::Plus);
simple_token!(asterisk, '*', TokenKind::Asterisk);
simple_token!(equals, '=', TokenKind::Equals);
simple_token!(colon, ':', TokenKind::Colon);

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
    let (_, span) = take_while(1.., AsChar::is_dec_digit)
        .with_span()
        .parse_next(input)?;

    Ok(Token::new(TokenKind::Int, Span::from_range(span)))
}

fn ident<'a>(input: &mut LocatingSlice<&'a str>) -> PResult<Token> {
    let (_, span) = take_while(1.., |c: char| {
        !c.is_whitespace()
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

    Ok(Token::new(TokenKind::Ident, Span::from_range(span)))
}

fn keyword<'a>(input: &mut LocatingSlice<&'a str>) -> PResult<Token> {
    let (keyword, span) = alt((
        "fn".with_span(),
        "struct".with_span(),
        "ret".with_span(),
        "let".with_span(),
    ))
    .parse_next(input)?;
    let token = match keyword {
        "fn" => TokenKind::Fn,
        //"struct" => TokenKind::Struct,
        //"ret" => TokenKind::Ret,
        "let" => TokenKind::Let,
        _ => unreachable!(),
    };

    Ok(Token::new(token, Span::from_range(span)))
}
