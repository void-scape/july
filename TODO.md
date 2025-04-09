# TODO

- Remove all of the `[item]Rules` structs and implement `ParserRule` for the data structures themselves.
- Backtraces: Much more information needs to be passed to the bytecode. Unfortunately, this will mean that static analysis for the formatter is no longer possible, since span information is required.
- Formatting `pebblec/tests/hosted/general.peb` makes tests fail!
