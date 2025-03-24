use proc_macro::TokenStream;
use quote::quote;
use syn::spanned::Spanned;
use syn::{DeriveInput, parse_macro_input};

#[proc_macro_derive(MatchTokens)]
pub fn impl_match_tokens(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match_tokens(input)
        .unwrap_or_else(|err| err.into_compile_error())
        .into()
}

fn match_tokens(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let mut variant_types = Vec::new();
    match input.data {
        syn::Data::Enum(data) => {
            for variant in data.variants.iter() {
                if !variant.fields.is_empty() {
                    return Err(syn::Error::new(variant.span(), "input must be an enum"));
                }

                let ident = &variant.ident;
                variant_types.push(quote! {
                    #[derive(Debug, Default)]
                    pub struct #ident;

                    impl TokenKindType for #ident {
                        fn kind() -> TokenKind {
                            TokenKind::#ident
                        }
                    }
                });
            }
        }
        _ => {
            return Err(syn::Error::new(input.span(), "input must be an enum"));
        }
    }

    Ok(quote! {
        #(#variant_types)*,
    })
}
