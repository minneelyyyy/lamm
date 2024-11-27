# Lamm

A small, functional programming language.

# Syntax

Lamm uses [Polish Notation](https://en.wikipedia.org/wiki/Polish_notation).
That means that instead of writing `5 + 6`, you would instead write `+ 5 6`.

Since we're here, we might as well cover some operators.

## Math Operators

```
+ 5 6   # => 11
- 5 6   # => -1
* 5 6   # => 30
/ 5 6   # => 0 (integer division)
** 5 6  # => 15625
% 6 5   # => 1
```

There is no order of operations to worry about, you essentially write your code in the order it should be evaluated in.

## Variables

Variables are **constant** in Lamm, there is no mutation. Here are some examples of defining variables.

```
= pi 3.1415926    # immediately evaluated
. sqrt2 ** 2 0.5  # lazy evaluated
```

Variables are **scoped** in Lamm, meaning they only exist in the single expression that they are defined for. That means that the following code is an **error**.

```
= pi 3.1415926
= r 16
* pi ** r 2       # OK

= deg 60
* deg / pi 360.0  # ERROR: `pi` was undefined
```

## Scope

Scope in Lamm consists of a single expression, such as `sqrt + ** a 2 ** b 2`. So then, what do I do when I need a variable for more than a single expression? There are multiple solutions depending on your needs.

### Multi-Statement Expression

You can create a multi-statement expression using either `()` syntax or the `~` operator, which `()` is simple syntactic sugar for. In these, only the value of the last expression is returned, the rest get ignored. This is the perfect place to put stateful function calls.

```
. x 12 (
	print + "My favorite number is " string x
	print + "Auf Wiedersehen! Ich werde aber meine Lieblingsnummer " + string x " vermissen."
)
```

### Global Scope

You can introduce a variable to global scope using the `export` builtin function.

```
# A very useful constant
= pi 3.1415926
export pi

# Some more useful constants
= e 2.71828
= phi 1.6180339887
export (e phi)
```

## Functions

All functions in Lamm are **scoped** similarly to variables. Functions are declared using the `:` operator, which can be extended with more `:` and `.` characters to let Lamm know how many arguments the function takes.

```
: inc x + x 1
	(inc 24)  # => 25

:. pythag a b sqrt + ** a 2.0 ** b 2.0
	(pythag 3 4)  # => 5

:::::. ten'args a b c d e f g h i j
	[a b c d e f g h i j]
```

The parameter types and return type of functions can be declared using a special syntax unique to function and lambda definitions.
Calling a function requires parenthises around the call.

```
# Takes an x of `Any` type
: inc x
	+ x 1
(inc 12)  # => 13

# Takes an x of `Int` and returns an `Int`
: inc ?. x Int -> Int
	+ x 1
(inc 9)  # => 10
```

The `?.` operator is unique to function declarations and is used to specify the type of an argument. There are also first class functions, here is the syntax for it.

```
# Applies a function to any value
:. apply : f x
	(f x)
(apply sqrt 9)  # => 3

# Applies a function f which maps an Int to an Int to x
:. apply'int ?: f Int -> Int ?. x Int -> Int
	(f x)
(apply'int sqrt 36)  # => 6
```

The `:` operator inside of a function prototype tells Lamm that this argument must be a function where every argument and it's return type are all `Any`. This means that `: f` is essentially syntactic sugar for `?: f Any -> Any`. You can pass a function with just it's identifier.

And of course, `:` and `?:` in function prototypes can also be extended depending on the number of arguments the function must take.

## Branching

Lamm has the following boolean expressions

```
== 1 2         # => false
!= 1 2         # => true
>  1 2         # => false
<  1 2         # => true
>= 1 2         # => false
<= 1 2         # => true
!true          # => false
true && false  # => false
true || false  # => true
```

These can be used inside of `?` (if) and `??` (if-else) statements.

```
. n 12
	?? < 12 10
		print "n is less than 10"
		print "n is greater than 10"
```

An `?` if statement where it's condition is false simply returns `nil`, as do `print` and other functions without a return value. `?` is mostly useful inside of blocks.

```
: times'twelve ?. n Int -> Int (
	? == n 0
		print "n is 0"

	* n 12
)
```

## Arrays

Lamm offers a few fundamental array operations.

```
+ 1 [2 3 4]     # => [1 2 3 4]
+ [1 2 3] 4     # => [1 2 3 4]
+ [1 2] [3 4]   # => [1 2 3 4]
head [1 2 3 4]  # => 1
tail [1 2 3 4]  # => [2 3 4]
init [1 2 3 4]  # => [1 2 3]
fini [1 2 3 4]  # => 4
bool [1 2 3 4]  # => true
bool empty      # => false
```

Using these, you can build a lot of fundamental functional paradigm functions.

```
:. map : f ?. x [] -> []
  ?? bool x
    [+ (f head x) (map f tail x)
    empty
(map ;x ** x 2 [1 2 3 4 5 6 7 8 9 10])  # => [1 4 9 16 25 36 49 64 81 100]

:: iterate : f i count -> []
  ?? > count 0
    [+ i (iterate f (f i) - count 1)
    empty
(iterate (+ 1) 0 10)  # => [0 1 2 3 4 5 6 7 8 9]

:. take ?. n Int ?. x [] -> []
	?? > n 0
		[+ head x (take - n 1 tail x)
		empty
(take 3 [1 2 3 4 5])  # => [1 2 3]

:. take'while ?: pred Any -> Bool ?. x [] -> []
	?? && bool x (pred head x)
		[+ head x (take'while pred tail x)
		empty
(take'while (> 10) [1 3 5 7 9 11 13 15 16])  # => [1 3 5 7 9]
```

## Lambdas

Lambdas are created using the `;` operator, and they are always passed as a value, so no `'` is necessary.


```
map ;x * x 12 [1 2 3]  # => [12 24 36]
```

They follow the same prototype syntax as regular functions, with the notable lack of an identifier.