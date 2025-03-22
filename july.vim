" Vim syntax file
" Language:     July
" Maintainer:   Nic Ball <balln13572@gmail.com>

if exists("b:current_syntax")
  finish
endif

syn keyword julyKeyword struct const extern return if for while loop break continue let
syn keyword julyKeyword impl as in

syn keyword julyBoolean true false
syn keyword julyType i32 i64 u8 u16 u32 u64 f32 f64 bool str

syn match julyOperator "\v\*"
syn match julyOperator "\v\+"
syn match julyOperator "\v-"
syn match julyOperator "\v/"
syn match julyOperator "\v\="
syn match julyOperator "\v\!"
syn match julyOperator "\v\&"
syn match julyOperator "\v\|"
syn match julyOperator "\v\>"
syn match julyOperator "\v\<"
syn match julyOperator "\v\^\="

syn match julyComment "//.*$"

syn match julyFunction "\v<\w+\ze\("

syn region julyString start=/"/ skip=/\\"/ end=/"/
syn region julyString start=/'/ skip=/\\'/ end=/'/

syn match julyNumber "\v<\d+>"
syn match julyNumber "\v<\d+\.\d+>"
syn match julyNumber "\v<0x[0-9a-fA-F]+>"

syn region julyAttribute start=/#\[/ end=/\]/
syn match julyType "\v<[A-Z]\w*>"
syn match julyConstant "\v<[A-Z][A-Z0-9_]+>"
syn match julyFunction "\v<\w+\ze\s*:\s*\("
syn match julyAttribute "#\[.*\]"

hi def link julyKeyword     Keyword
hi def link julyBoolean     Boolean
hi def link julyType        Type
hi def link julyOperator    Operator
hi def link julyComment     Comment
hi def link julyFunction    Function
hi def link julyString      String
hi def link julyNumber      Number
hi def link julyAttribute   PreProc
hi def link julyConstant    Number

let b:current_syntax = "july"
