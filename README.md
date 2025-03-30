# ![Pebble](logo.svg)

[![ci](https://github.com/void-scape/pebble/actions/workflows/ci.yml/badge.svg)](https://github.com/void-scape/pebble/actions/workflows/ci.yml)

**Pebble** is a statically typed programming language with **arbitrary code execution at compile time**.

> [!WARNING]
> At the moment, there is no compiler backend. Everything is interpreted _at compile time_. This means the compiler does not produce a binary. I deem basic language functionality more important than binaries at the moment. See below...

```rust
use core::io;

main: () {
    println("Hello, World!");
    println("This % the number %!", "is", 42);
}
```
> NOTE
> See [invaders.peb](https://github.com/void-scape/pebble/blob/master/demo/invaders.peb) for an in-depth example.

To use **Pebble**, you will need to install the [rust toolchain](https://www.rust-lang.org/tools/install) and compile the latest code.

First, clone the repo:

```console 
$ git clone https://github.com/void-scape/pebble.git
$ cd pebble
```

Then build the compiler, `pebblec`:

```console 
$ cargo build --release -p pebblec
```

The compiler binary will be located in `pebble/target/release/pebblec`. Be aware that pebblec relies on its location to find the `core` library, which is located in `pebble/core/`.

Finally, run `pebblec` and voila!

```console
$ target/release/pebblec -f myfile.peb
```

# Road Map

### Short Term (in no particular order)

- Name-spacing
- Methods
- Enums
- Switching (later transitioned into pattern matching)
- Compiler backend

### Long Term

- Pattern matching
- Traits & generics
