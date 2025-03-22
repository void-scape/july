" Vim syntax file
" Language:     July
" Maintainer:   Nic Ball <balln13572@gmail.com>

if exists("b:current_syntax")
  finish
endif

syn keyword pebbleKeyword struct const extern return if for while loop break continue let
syn keyword pebbleKeyword impl as in

syn keyword pebbleBoolean true false
syn keyword pebbleType i32 i64 u8 u16 u32 u64 f32 f64 bool str

syn match pebbleOperator "\v\*"
syn match pebbleOperator "\v\+"
syn match pebbleOperator "\v-"
syn match pebbleOperator "\v/"
syn match pebbleOperator "\v\="
syn match pebbleOperator "\v\!"
syn match pebbleOperator "\v\&"
syn match pebbleOperator "\v\|"
syn match pebbleOperator "\v\>"
syn match pebbleOperator "\v\<"
syn match pebbleOperator "\v\^\="

syn match pebbleComment "//.*$"

syn match pebbleFunction "\v<\w+\ze\("

syn region pebbleString start=/"/ skip=/\\"/ end=/"/
syn region pebbleString start=/'/ skip=/\\'/ end=/'/

syn match pebbleNumber "\v<\d+>"
syn match pebbleNumber "\v<\d+\.\d+>"
syn match pebbleNumber "\v<0x[0-9a-fA-F]+>"

syn region pebbleAttribute start=/#\[/ end=/\]/
syn match pebbleType "\v<[A-Z]\w*>"
syn match pebbleConstant "\v<[A-Z][A-Z0-9_]+>"
syn match pebbleFunction "\v<\w+\ze\s*:\s*\("
syn match pebbleAttribute "#\[.*\]"

hi def link pebbleKeyword     Keyword
hi def link pebbleBoolean     Boolean
hi def link pebbleType        Type
hi def link pebbleOperator    Operator
hi def link pebbleComment     Comment
hi def link pebbleFunction    Function
hi def link pebbleString      String
hi def link pebbleNumber      Number
hi def link pebbleAttribute   PreProc
hi def link pebbleConstant    Number

let b:current_syntax = "pebble"
