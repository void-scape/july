/// Definition of all kinds of tokens found within a source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    /// `if`
    If,
    /// `else`
    Else,
    /// `true`
    True,
    /// `false`
    False,

    /// `struct`
    Struct,
    /// `enum`
    Enum,
    /// `let`
    Let,
    /// `fn`
    Fn,
    /// `return`
    Ret,
    /// `;`
    Semi,
    /// `:`
    Colon,
    /// `=`
    Equals,
    /// `(`
    OpenParen,
    /// `)`
    CloseParen,
    /// `[`
    OpenBracket,
    /// `]`
    CloseBracket,
    /// `{`
    OpenCurly,
    /// `}`
    CloseCurly,
    /// `0`..`9`
    Int,
    /// `<a..z | A..Z>[a..z | A..Z | 0..9]`
    Ident,
    /// '+'
    Plus,
    /// `*`
    Asterisk,
    /// `/`
    Slash,
    /// `>`
    Greater,
    /// `-`
    Hyphen,
    /// `,`
    Comma,
    /// `.`
    Dot,
}

impl TokenKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Enum => "enum",
            Self::Struct => "struct",
            Self::Let => "let",
            Self::Fn => "fn",
            Self::Ret => "return",
            Self::Semi => ";",
            Self::Colon => ":",
            Self::Equals => "=",
            Self::OpenParen => "(",
            Self::CloseParen => ")",
            Self::OpenBracket => "[",
            Self::CloseBracket => "]",
            Self::OpenCurly => "{",
            Self::CloseCurly => "}",
            Self::Plus => "+",
            Self::Asterisk => "*",
            Self::Slash => "/",
            Self::Greater => ">",
            Self::Hyphen => "-",
            Self::Comma => ",",
            Self::Dot => ".",

            Self::If => "if",
            Self::Else => "else",
            Self::True => "true",
            Self::False => "false",

            Self::Int => "int literal",
            Self::Ident => "identifier",
        }
    }

    pub fn is_terminator(&self) -> bool {
        matches!(self, Self::Semi)
    }

    pub fn is_keyword(&self) -> bool {
        match self {
            Self::Ret | Self::Struct | Self::Fn | Self::Let => true,
            _ => false,
        }
    }
}
