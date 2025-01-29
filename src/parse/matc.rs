use crate::lex::kind::TokenKind;
use std::marker::PhantomData;

/// [`TokenKind`] constraints used by [`crate::parse::rule::Rule`].
pub trait MatchTokenKind {
    fn matches(kind: Option<TokenKind>) -> bool;
}

impl<T> MatchTokenKind for T
where
    T: TokenKindType,
{
    fn matches(kind: Option<TokenKind>) -> bool {
        kind.is_some_and(|kind| T::kind() == kind)
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
}

macro_rules! impl_match_token {
    ($($T:ident),*) => {
        impl<$($T,)*> MatchTokenKind for Any<($($T,)*)>
        where
            $($T: MatchTokenKind,)*
        {
            fn matches(kind: Option<TokenKind>) -> bool {
                $($T::matches(kind) ||)* false
            }
        }
    };
}

variadics_please::all_tuples!(impl_match_token, 1, 5, T);

/// Matches true if no input token is supplied.
#[derive(Debug, Default)]
pub struct Empty;

impl MatchTokenKind for Empty {
    fn matches(kind: Option<TokenKind>) -> bool {
        kind.is_none()
    }
}

/// Matches true for any [`TokenKindType`] that is not `T`.
#[derive(Debug, Default)]
pub struct Not<T>(PhantomData<T>);

impl<T> MatchTokenKind for Not<T>
where
    T: MatchTokenKind,
{
    fn matches(kind: Option<TokenKind>) -> bool {
        !T::matches(kind)
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

impl_tkt!(Fn, TokenKind::Fn);
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
