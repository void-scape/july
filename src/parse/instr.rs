use crate::lex::buffer::{TokenBuffer, TokenId};

/// Manipulate the token stack with a given input token.
pub trait Instr {
    fn apply(
        buffer: &TokenBuffer,
        token: TokenId,
        stack: &mut Vec<TokenId>,
    ) -> Result<(), InstrError>;
}

pub struct InstrError;

impl Instr for () {
    fn apply(
        _buffer: &TokenBuffer,
        _token: TokenId,
        _stack: &mut Vec<TokenId>,
    ) -> Result<(), InstrError> {
        Ok(())
    }
}

/// Swap the top two elements of the stack.
#[derive(Debug, Default)]
pub struct Swap;

impl Instr for Swap {
    fn apply(_: &TokenBuffer, _: TokenId, stack: &mut Vec<TokenId>) -> Result<(), InstrError> {
        if stack.len() > 1 {
            let len = stack.len();
            let tmp = stack[len - 1];
            stack[len - 1] = stack[len - 2];
            stack[len - 2] = tmp;
            Ok(())
        } else {
            Err(InstrError)
        }
    }
}

/// Push the current token onto the stack.
#[derive(Debug, Default)]
pub struct Push;

//impl AsRule for Push {}

impl Instr for Push {
    fn apply(_: &TokenBuffer, token: TokenId, stack: &mut Vec<TokenId>) -> Result<(), InstrError> {
        stack.push(token);
        Ok(())
    }
}

/// Pop the top token off of the stack.
#[derive(Debug, Default)]
pub struct Pop;

//impl AsRule for Pop {}

impl Instr for Pop {
    fn apply(_: &TokenBuffer, _: TokenId, stack: &mut Vec<TokenId>) -> Result<(), InstrError> {
        let _ = stack.pop();
        Ok(())
    }
}

macro_rules! impl_instr {
    ($($T:ident),*) => {
        impl<$($T,)*> Instr for ($($T,)*)
        where
            $($T: Instr,)*
        {
            fn apply(
                buffer: &TokenBuffer,
                token: TokenId,
                stack: &mut Vec<TokenId>,
            ) -> Result<(), InstrError> {
                $($T::apply(buffer, token, stack)?;)*
                Ok(())
            }
        }
    };
}

variadics_please::all_tuples!(impl_instr, 1, 15, T);
