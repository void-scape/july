use crate::lex::kind::*;
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
        format!("expected `{}`", kind.as_str())
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

variadics_please::all_tuples!(impl_match_token, 1, 10, T);

pub trait DelimPair {
    fn matches_open(kind: Option<TokenKind>) -> bool;
    fn matches_close(kind: Option<TokenKind>) -> bool;
    fn open() -> TokenKind;
    fn close() -> TokenKind;
}

pub struct Paren;

impl DelimPair for Paren {
    fn matches_open(kind: Option<TokenKind>) -> bool {
        OpenParen::matches(kind)
    }

    fn matches_close(kind: Option<TokenKind>) -> bool {
        CloseParen::matches(kind)
    }

    fn open() -> TokenKind {
        TokenKind::OpenParen
    }

    fn close() -> TokenKind {
        TokenKind::CloseParen
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

    fn open() -> TokenKind {
        TokenKind::OpenCurly
    }

    fn close() -> TokenKind {
        TokenKind::CloseCurly
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

    fn open() -> TokenKind {
        TokenKind::OpenBracket
    }

    fn close() -> TokenKind {
        TokenKind::CloseBracket
    }
}

/// Associates a [`TokenKind`] with the implementer.
pub trait TokenKindType {
    fn kind() -> TokenKind;
}
