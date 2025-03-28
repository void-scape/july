use crate::{rules::*, stream::TokenStream};

/// Returns the first successful rule from `T`.
///
/// Fails if all rules failed.
#[derive(Debug, Default)]
pub struct Alt<T>(T);

impl<'a,  O, A, B> ParserRule<'a> for Alt<(A, B)>
where
    A: ParserRule<'a, Output = O>,
    B: ParserRule<'a, Output = O>,
{
    type Output = O;

    fn parse(stream: &mut TokenStream<'a>) -> RResult<Self::Output> {
        let str = *stream;
        match A::parse(stream) {
            Err(err) => {
                if err.recoverable() {
                    *stream = str;
                    B::parse(stream)
                } else {
                    Err(err)
                }
            }
            Ok(val) => Ok(val),
        }
    }
}

impl<'a,  O, A, B, C> ParserRule<'a> for Alt<(A, B, C)>
where
    A: ParserRule<'a, Output = O>,
    B: ParserRule<'a, Output = O>,
    C: ParserRule<'a, Output = O>,
{
    type Output = O;

    fn parse(stream: &mut TokenStream<'a>) -> RResult<Self::Output> {
        let str = *stream;
        match A::parse(stream) {
            Err(err) => {
                if err.recoverable() {
                    *stream = str;
                    match B::parse(stream) {
                        Err(err) => {
                            if err.recoverable() {
                                *stream = str;
                                C::parse(stream)
                            } else {
                                Err(err)
                            }
                        }
                        Ok(val) => Ok(val),
                    }
                } else {
                    Err(err)
                }
            }
            Ok(val) => Ok(val),
        }
    }
}

impl<'a,  O, A, B, C, D> ParserRule<'a> for Alt<(A, B, C, D)>
where
    A: ParserRule<'a, Output = O>,
    B: ParserRule<'a, Output = O>,
    C: ParserRule<'a, Output = O>,
    D: ParserRule<'a, Output = O>,
{
    type Output = O;

    fn parse(stream: &mut TokenStream<'a>) -> RResult<Self::Output> {
        let str = *stream;
        match A::parse(stream) {
            Err(err) => {
                if err.recoverable() {
                    *stream = str;
                    match B::parse(stream) {
                        Err(err) => {
                            if err.recoverable() {
                                *stream = str;
                                match C::parse(stream) {
                                    Err(err) => {
                                        if err.recoverable() {
                                            *stream = str;
                                            D::parse(stream)
                                        } else {
                                            Err(err)
                                        }
                                    }
                                    Ok(val) => Ok(val),
                                }
                            } else {
                                Err(err)
                            }
                        }
                        Ok(val) => Ok(val),
                    }
                } else {
                    Err(err)
                }
            }
            Ok(val) => Ok(val),
        }
    }
}

impl<'a,  O, A, B, C, D, E> ParserRule<'a> for Alt<(A, B, C, D, E)>
where
    A: ParserRule<'a, Output = O>,
    B: ParserRule<'a, Output = O>,
    C: ParserRule<'a, Output = O>,
    D: ParserRule<'a, Output = O>,
    E: ParserRule<'a, Output = O>,
{
    type Output = O;

    fn parse(stream: &mut TokenStream<'a>) -> RResult<Self::Output> {
        let str = *stream;
        match A::parse(stream) {
            Err(err) => {
                if err.recoverable() {
                    *stream = str;
                    match B::parse(stream) {
                        Err(err) => {
                            if err.recoverable() {
                                *stream = str;
                                match C::parse(stream) {
                                    Err(err) => {
                                        if err.recoverable() {
                                            *stream = str;
                                            match D::parse(stream) {
                                                Err(err) => {
                                                    if err.recoverable() {
                                                        *stream = str;
                                                        E::parse(stream)
                                                    } else {
                                                        Err(err)
                                                    }
                                                }
                                                Ok(val) => Ok(val),
                                            }
                                        } else {
                                            Err(err)
                                        }
                                    }
                                    Ok(val) => Ok(val),
                                }
                            } else {
                                Err(err)
                            }
                        }
                        Ok(val) => Ok(val),
                    }
                } else {
                    Err(err)
                }
            }
            Ok(val) => Ok(val),
        }
    }
}

impl<'a,  O, A, B, C, D, E, F> ParserRule<'a> for Alt<(A, B, C, D, E, F)>
where
    A: ParserRule<'a, Output = O>,
    B: ParserRule<'a, Output = O>,
    C: ParserRule<'a, Output = O>,
    D: ParserRule<'a, Output = O>,
    E: ParserRule<'a, Output = O>,
    F: ParserRule<'a, Output = O>,
{
    type Output = O;

    fn parse(stream: &mut TokenStream<'a>) -> RResult<Self::Output> {
        let str = *stream;
        match A::parse(stream) {
            Err(err) => {
                if err.recoverable() {
                    *stream = str;
                    match B::parse(stream) {
                        Err(err) => {
                            if err.recoverable() {
                                *stream = str;
                                match C::parse(stream) {
                                    Err(err) => {
                                        if err.recoverable() {
                                            *stream = str;
                                            match D::parse(stream) {
                                                Err(err) => {
                                                    if err.recoverable() {
                                                        *stream = str;
                                                        match E::parse(stream) {
                                                            Err(err) => {
                                                                if err.recoverable() {
                                                                    *stream = str;
                                                                    F::parse(stream)
                                                                } else {
                                                                    Err(err)
                                                                }
                                                            }
                                                            Ok(val) => Ok(val),
                                                        }
                                                    } else {
                                                        Err(err)
                                                    }
                                                }
                                                Ok(val) => Ok(val),
                                            }
                                        } else {
                                            Err(err)
                                        }
                                    }
                                    Ok(val) => Ok(val),
                                }
                            } else {
                                Err(err)
                            }
                        }
                        Ok(val) => Ok(val),
                    }
                } else {
                    Err(err)
                }
            }
            Ok(val) => Ok(val),
        }
    }
}
