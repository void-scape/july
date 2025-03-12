//use crate::diagnostic::Diag;
//use crate::parse::{rules::*, stream::TokenStream};
//
//pub trait Report {
//    fn report<'a, 's>(stream: &'a TokenStream<'a, 's>, err: Diag<'s>) -> Diag<'s>;
//}
//
//// TODO: parsing errors need to be brought up into the title so they are more concise
//#[derive(Debug, Default)]
//pub struct Reported<T, R>(T, R);
//
//impl<'a, 's, T, R> ParserRule<'a, 's> for Reported<T, R>
//where
//    T: ParserRule<'a, 's>,
//    R: Report,
//{
//    type Output = <T as ParserRule<'a, 's>>::Output;
//
//    fn parse(stream: &mut TokenStream<'a, 's>) -> RResult<'s, Self::Output> {
//        T::parse(stream).map_err(|err| R::report(stream, err))
//    }
//}
//
//#[macro_export]
//macro_rules! parse_help {
//    ($ty:ident, $title:expr, $msg:expr) => {
//        #[derive(Debug, Default)]
//        pub struct $ty;
//
//        impl Report for $ty {
//            fn report<'a>(
//                buffer: &'a TokenBuffer<'a>,
//                stream: &'a TokenStream<'a, 's>,
//                err: Diag<'a>,
//            ) -> Diag<'a> {
//                Diag::sourced(
//                    $title,
//                    buffer.source(),
//                    crate::diagnostic::Msg::help(buffer.span(stream.prev()), $msg),
//                )
//                .level(annotate_snippets::Level::Help)
//                .wrap(err)
//            }
//        }
//    };
//}
