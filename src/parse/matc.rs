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

macro_rules! impl_tkt_for {
    ($variant:ident) => {
        impl_tkt!($variant, TokenKind::$variant);
    };
}

impl_tkt_for!(If);
impl_tkt_for!(Else);
impl_tkt_for!(True);
impl_tkt_for!(False);
impl_tkt_for!(Enum);
impl_tkt_for!(Fn);
impl_tkt_for!(Struct);
impl_tkt_for!(Ret);
impl_tkt_for!(Let);
impl_tkt_for!(Colon);
impl_tkt_for!(Ident);
impl_tkt_for!(Equals);
impl_tkt_for!(Int);
impl_tkt_for!(Semi);
impl_tkt_for!(Plus);
impl_tkt_for!(Asterisk);
impl_tkt_for!(OpenCurly);
impl_tkt_for!(CloseCurly);
impl_tkt_for!(OpenParen);
impl_tkt_for!(CloseParen);
impl_tkt_for!(OpenBracket);
impl_tkt_for!(CloseBracket);
impl_tkt_for!(Hyphen);
impl_tkt_for!(Greater);
impl_tkt_for!(Comma);
impl_tkt_for!(Dot);
impl_tkt_for!(DoubleDot);
impl_tkt_for!(Pound);
impl_tkt_for!(Ampersand);
impl_tkt_for!(Str);
impl_tkt_for!(Extern);
impl_tkt_for!(Loop);
impl_tkt_for!(For);
impl_tkt_for!(In);
impl_tkt_for!(Const);
impl_tkt_for!(Caret);
impl_tkt_for!(As);
