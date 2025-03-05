use crate::diagnostic::Diag;
use crate::lex::buffer::{TokenBuffer, TokenId};
use crate::parse::{rules::*, stream::TokenStream};

pub trait Report {
    fn report<'a>(buffer: &'a TokenBuffer<'a>, stream: &TokenStream<'a>, err: Diag<'a>)
        -> Diag<'a>;
}

// TODO: parsing errors need to be brought up into the title so they are more concise
#[derive(Debug, Default)]
pub struct Reported<T, R>(T, R);

impl<'a, T, R> ParserRule<'a> for Reported<T, R>
where
    T: ParserRule<'a>,
    R: Report,
{
    type Output = <T as ParserRule<'a>>::Output;

    fn parse(
        buffer: &'a TokenBuffer<'a>,
        stream: &mut TokenStream<'a>,
        stack: &mut Vec<TokenId>,
    ) -> RResult<'a, Self::Output> {
        T::parse(buffer, stream, stack).map_err(|err| R::report(buffer, stream, err))
    }
}

#[macro_export]
macro_rules! parse_help {
    ($ty:ident, $title:expr, $msg:expr) => {
        #[derive(Debug, Default)]
        pub struct $ty;

        impl Report for $ty {
            fn report<'a>(
                buffer: &'a TokenBuffer<'a>,
                stream: &TokenStream<'a>,
                err: Diag<'a>,
            ) -> Diag<'a> {
                Diag::sourced(
                    $title,
                    buffer.source(),
                    crate::diagnostic::Msg::help(buffer.span(stream.prev()), $msg),
                )
                .level(annotate_snippets::Level::Help)
                .wrap(err)
            }
        }
    };
}
