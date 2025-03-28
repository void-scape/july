use crate::{rules::*, stream::TokenStream};

/// Return `Some` `T` output if `Ok`, otherwise `None`.
#[derive(Debug, Default)]
pub struct Opt<T>(T);

impl<'a, 's, T> ParserRule<'a> for Opt<T>
where
    T: ParserRule<'a>,
{
    type Output = Option<<T as ParserRule<'a>>::Output>;

    #[track_caller]
    fn parse(stream: &mut TokenStream<'a>) -> RResult<Self::Output> {
        let chck = *stream;

        match T::parse(stream) {
            Ok(v) => Ok(Some(v)),
            Err(_) => {
                *stream = chck;
                Ok(None)
            }
        }
    }
}

/// Succeeds if and only if all inputs fail or all inputs pass.
#[derive(Debug, Default)]
pub struct XNor<T>(T);

macro_rules! impl_xnor {
    ($(($n:tt, $T:ident)),*) => {
        impl<'a, 's, $($T,)*> ParserRule<'a> for XNor<($($T,)*)>
        where
            $($T: ParserRule<'a> + Default,)*
        {
            type Output = Option<($(<$T as ParserRule<'a>>::Output,)*)>;

            #[track_caller]
            fn parse(
                stream: &mut TokenStream<'a>,
            ) -> RResult<Self::Output> {
                let chck = *stream;

                let results = ($({
                    let res = $T::parse(stream);
                    if res.is_err() {
                        *stream = chck;
                    }
                    res
                },)*);
                let test_for = results.0.is_ok();

                if $(results.$n.is_ok() == test_for &&)* true {
                    if test_for {
                        Ok(Some(($(results.$n.unwrap(),)*)))
                    } else {
                        *stream = chck;
                        Ok(None)
                    }
                } else {
                    *stream = chck;

                    $(if let Err(e) = results.$n {
                        return Err(e);
                    })*

                    unreachable!();
                }
            }
        }
    };
}

variadics_please::all_tuples_enumerated!(impl_xnor, 1, 10, T);
