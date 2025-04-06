use crate::matc::TokenKindType;
use pebblec_macros::MatchTokens;

#[derive(MatchTokens, Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    // keywords
    Slf,
    Use,
    Impl,
    Const,
    Loop,
    While,
    If,
    Else,
    For,
    In,
    True,
    False,
    Break,
    Continue,
    Extern,
    Struct,
    Enum,
    Let,
    Fn,
    Ret,
    As,

    // delims
    OpenParen,
    CloseParen,
    OpenBracket,
    CloseBracket,
    OpenCurly,
    CloseCurly,
    OpenAngle,
    CloseAngle,

    // literals
    Int,
    Float,
    Str,
    Ident,

    // symbols
    Semi,
    Colon,
    Equals,
    Plus,
    Asterisk,
    Percent,
    Slash,
    Hyphen,
    Comma,
    Dot,
    DoubleDot,
    Pound,
    Ampersand,
    Bang,
    Caret,
    Pipe,
}

impl TokenKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            // keywords
            Self::Slf => "self",
            Self::Use => "use",
            Self::Impl => "impl",
            Self::Const => "const",
            Self::Loop => "loop",
            Self::While => "while",
            Self::If => "if",
            Self::Else => "else",
            Self::For => "for",
            Self::In => "in",
            Self::True => "true",
            Self::False => "false",
            Self::Break => "break",
            Self::Continue => "continue",
            Self::Extern => "extern",
            Self::Struct => "struct",
            Self::Enum => "enum",
            Self::Let => "let",
            Self::Fn => "fn",
            Self::Ret => "return",
            Self::As => "as",

            // delims
            Self::OpenParen => "(",
            Self::CloseParen => ")",
            Self::OpenBracket => "[",
            Self::CloseBracket => "]",
            Self::OpenCurly => "{",
            Self::CloseCurly => "}",
            Self::OpenAngle => "<",
            Self::CloseAngle => ">",

            // literals
            Self::Int => "integer",
            Self::Float => "float",
            Self::Str => "string",
            Self::Ident => "identifier",

            // symbols
            Self::Semi => ";",
            Self::Colon => ":",
            Self::Equals => "=",
            Self::Plus => "+",
            Self::Asterisk => "*",
            Self::Percent => "%",
            Self::Slash => "/",
            Self::Hyphen => "-",
            Self::Comma => ",",
            Self::Dot => ".",
            Self::DoubleDot => "..",
            Self::Pound => "#",
            Self::Ampersand => "&",
            Self::Bang => "!",
            Self::Caret => "^",
            Self::Pipe => "|",
        }
    }
}
