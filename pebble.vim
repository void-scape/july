" Vim syntax file
" Language:   Pebble
" Maintainer: Nic Ball <balln13572@gmail.com>

if exists("b:current_syntax")
  finish
endif

syn keyword pebbleKeyword   struct const extern return if else for while loop break continue let
syn keyword pebbleKeyword   use impl as in

syn keyword pebbleBoolean   true false
syn keyword pebbleType      i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 bool str

syn match   pebbleOperator  "\v\^"
syn match   pebbleOperator  "\v\%"
syn match   pebbleOperator  "\v\*"
syn match   pebbleOperator  "\v\+"
syn match   pebbleOperator  "\v-"
syn match   pebbleOperator  "\v/"
syn match   pebbleOperator  "\v\="
syn match   pebbleOperator  "\v\!"
syn match   pebbleOperator  "\v\&"
syn match   pebbleOperator  "\v\|"
syn match   pebbleOperator  "\v\>"
syn match   pebbleOperator  "\v\<"
syn match   pebbleOperator  "\v\^\="

syn match   pebbleComment   "//.*$"

syn match   pebbleFunction  "\v<\w+\ze\("

syn region  pebbleString    start=/"/ skip=/\\"/ end=/"/
syn region  pebbleString    start=/'/ skip=/\\'/ end=/'/

syn match   pebbleNumber    "\v<\d+>"
syn match   pebbleNumber    "\v<\d+\.\d+>"
syn match   pebbleNumber    "\v<0x[0-9a-fA-F]+>"
syn match   pebbleNumber    "\v<0b[0-9]+>"

syn region  pebbleAttribute start=/#\[/ end=/\]/
syn match   pebbleType      "\v<[A-Z]\w*>"
syn match   pebbleConstant  "\v<[A-Z][A-Z0-9_]+>"
syn match   pebbleFunction  "\v<\w+\ze\s*:\s*\("
syn match   pebbleAttribute "#\[.*\]"

syn match   pebblePathSep   "::"

hi def link pebbleKeyword   Keyword
hi def link pebbleBoolean   Boolean
hi def link pebbleType      Type
hi def link pebbleOperator  Operator
hi def link pebblePathSep   Operator
hi def link pebbleComment   Comment
hi def link pebbleFunction  Function
hi def link pebbleString    String
hi def link pebbleNumber    Number
hi def link pebbleAttribute PreProc
hi def link pebbleConstant  Number

let b:current_syntax = "pebble"
