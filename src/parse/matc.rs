use crate::lex::kind::TokenKind;
use std::marker::PhantomData;

pub trait MatchTokenKind {
    fn matches(kind: Option<TokenKind>) -> bool;
    fn expect() -> String;
}

impl<T> MatchTokenKind for T
where
    T: TokenKindType,
{
    fn matches(kind: Option<TokenKind>) -> bool {
        kind.is_some_and(|kind| T::kind() == kind)
    }

    fn expect() -> String {
        let kind = T::kind();
        match kind {
            TokenKind::Let
            | TokenKind::Enum
            | TokenKind::Struct
            | TokenKind::Fn
            | TokenKind::Ret
            | TokenKind::Semi
            | TokenKind::Colon
            | TokenKind::Equals
            | TokenKind::OpenParen
            | TokenKind::CloseParen
            | TokenKind::OpenBracket
            | TokenKind::CloseBracket
            | TokenKind::OpenCurly
            | TokenKind::CloseCurly
            | TokenKind::Plus
            | TokenKind::Slash
            | TokenKind::Hyphen
            | TokenKind::Greater
            | TokenKind::Comma
            | TokenKind::Dot
            | TokenKind::If
            | TokenKind::Else
            | TokenKind::True
            | TokenKind::False
            | TokenKind::Pound
            | TokenKind::Asterisk => format!("expected `{}`", kind.as_str()),
            TokenKind::Int => format!("expected {} (e.g. `14`)", kind.as_str()),
            TokenKind::Ident => format!("expected {}", kind.as_str()),
        }
    }
}

pub struct Not<T>(PhantomData<T>);

impl<T> MatchTokenKind for Not<T>
where
    T: MatchTokenKind,
{
    fn matches(kind: Option<TokenKind>) -> bool {
        !T::matches(kind)
    }

    fn expect() -> String {
        format!("Not<`{}`>", T::expect())
    }
}

/// Matches any input token when `T` is `()`. Otherwise, matches any
/// [`TokenKindType`] in `T`: e.g. (Ident, Colon, Type).
#[derive(Debug, Default)]
pub struct Any<T = ()>(PhantomData<T>);

impl MatchTokenKind for Any<()> {
    fn matches(_: Option<TokenKind>) -> bool {
        true
    }

    fn expect() -> String {
        "expected token".into()
    }
}

macro_rules! impl_match_token {
    ($($T:ident),*) => {
        impl<$($T,)*> MatchTokenKind for Any<($($T,)*)>
        where
            $($T: TokenKindType,)*
        {
            fn matches(kind: Option<TokenKind>) -> bool {
                $($T::matches(kind) ||)* false
            }

            fn expect() -> String {
                let mut e = String::new();
                $(
                    e.push_str($T::kind().as_str());
                    e.push_str(", ");
                )*
                _ = e.pop();
                _ = e.pop();

                format!("expected one of [ {} ]", e)
            }
        }
    };
}

variadics_please::all_tuples!(impl_match_token, 1, 5, T);

pub trait DelimPair {
    fn matches_open(kind: Option<TokenKind>) -> bool;
    fn matches_close(kind: Option<TokenKind>) -> bool;
}

pub struct Paren;

impl DelimPair for Paren {
    fn matches_open(kind: Option<TokenKind>) -> bool {
        OpenParen::matches(kind)
    }

    fn matches_close(kind: Option<TokenKind>) -> bool {
        CloseParen::matches(kind)
    }
}

pub struct Curly;

impl DelimPair for Curly {
    fn matches_open(kind: Option<TokenKind>) -> bool {
        OpenCurly::matches(kind)
    }

    fn matches_close(kind: Option<TokenKind>) -> bool {
        CloseCurly::matches(kind)
    }
}

pub struct Bracket;

impl DelimPair for Bracket {
    fn matches_open(kind: Option<TokenKind>) -> bool {
        OpenBracket::matches(kind)
    }

    fn matches_close(kind: Option<TokenKind>) -> bool {
        CloseBracket::matches(kind)
    }
}

/// Associates a [`TokenKind`] with the implementer.
pub trait TokenKindType {
    fn kind() -> TokenKind;
}

macro_rules! impl_tkt {
    ($ty:ident, $kind:expr) => {
        #[derive(Debug, Default)]
        pub struct $ty;

        impl TokenKindType for $ty {
            fn kind() -> TokenKind {
                $kind
            }
        }
    };
}

impl_tkt!(If, TokenKind::If);
impl_tkt!(Else, TokenKind::Else);
impl_tkt!(True, TokenKind::True);
impl_tkt!(False, TokenKind::False);

impl_tkt!(Enum, TokenKind::Enum);
impl_tkt!(Fn, TokenKind::Fn);
impl_tkt!(Struct, TokenKind::Struct);
impl_tkt!(Ret, TokenKind::Ret);
impl_tkt!(Let, TokenKind::Let);
impl_tkt!(Colon, TokenKind::Colon);
impl_tkt!(Ident, TokenKind::Ident);
impl_tkt!(Equals, TokenKind::Equals);
impl_tkt!(Int, TokenKind::Int);
impl_tkt!(Semi, TokenKind::Semi);
impl_tkt!(Plus, TokenKind::Plus);
impl_tkt!(Asterisk, TokenKind::Asterisk);
impl_tkt!(OpenCurly, TokenKind::OpenCurly);
impl_tkt!(CloseCurly, TokenKind::CloseCurly);
impl_tkt!(OpenParen, TokenKind::OpenParen);
impl_tkt!(CloseParen, TokenKind::CloseParen);
impl_tkt!(OpenBracket, TokenKind::OpenBracket);
impl_tkt!(CloseBracket, TokenKind::CloseBracket);
impl_tkt!(Hyphen, TokenKind::Hyphen);
impl_tkt!(Greater, TokenKind::Greater);
impl_tkt!(Comma, TokenKind::Comma);
impl_tkt!(Dot, TokenKind::Dot);
impl_tkt!(Pound, TokenKind::Pound);
