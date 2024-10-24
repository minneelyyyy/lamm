
## Internal Details

- arrays as iterators instead of a vector
- a virtual machine and bytecode

## Language Features

- tuples
- `extern "C"` functions
- modules (`import` function)
- structs
- data types (need an IO object for stateful functions to return)
- unpacking type parameters (`(x:xs)` in Haskell for example)
- type variables in function parameters and data types
- automatic Int to Float casting if a parameter expects a float
- `[x..y]` array generators
- `(+)` = `;.x y + x y`

## Maybe Add

- `/` for float division and `//` for integer division