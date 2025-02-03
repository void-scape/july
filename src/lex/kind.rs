/// Definition of all kinds of tokens found within a source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    /// `struct`
    Struct,
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
}

impl TokenKind {
    pub fn as_str(&self) -> &'static str {
        match self {
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
