/// Definition of all kinds of tokens found within a source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
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
}

impl TokenKind {
    pub fn is_terminator(&self) -> bool {
        matches!(self, Self::Semi)
    }
}
