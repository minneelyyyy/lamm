use super::{Value, Type, Function, FunctionType};
use super::tokenizer::{Token, TokenizeError, Op};

use std::error;
use std::collections::HashMap;
use std::fmt::Display;
use std::borrow::Cow;

#[derive(Debug)]
pub enum ParseError {
    NoInput,
    UnexpectedEndInput,
    IdentifierUndefined(String),
    InvalidIdentifier,
    FunctionUndefined(String),
    VariableUndefined(String),
    UnmatchedArrayClose,
    UnwantedToken(Token),
    TokenizeError(TokenizeError),
    ImmutableError(String),
}

impl Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::UnexpectedEndInput => write!(f, "Input ended unexpectedly"),
            ParseError::IdentifierUndefined(name) => write!(f, "Undefined variable `{name}`"),
            ParseError::InvalidIdentifier => write!(f, "Invalid identifier"),
            ParseError::FunctionUndefined(name) => write!(f, "Undefined function `{name}`"),
            ParseError::VariableUndefined(name) => write!(f, "Undefined variable `{name}`"),
            ParseError::NoInput => write!(f, "No input given"),
            ParseError::UnmatchedArrayClose => write!(f, "there was an unmatched array closing operator `]`"),
            ParseError::TokenizeError(e) => write!(f, "Tokenizer Error: {e}"),
            ParseError::ImmutableError(i) => write!(f, "attempt to redeclare {i} met with force"),
            ParseError::UnwantedToken(_t) => write!(f, "unexpected token"),
        }
    }
}

impl error::Error for ParseError {}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ParseTree {
    // Mathematical Operators
    Add(Box<ParseTree>, Box<ParseTree>),
    Sub(Box<ParseTree>, Box<ParseTree>),
    Mul(Box<ParseTree>, Box<ParseTree>),
    Div(Box<ParseTree>, Box<ParseTree>),
    Exp(Box<ParseTree>, Box<ParseTree>),
    Mod(Box<ParseTree>, Box<ParseTree>),

    // Boolean Operations
    EqualTo(Box<ParseTree>, Box<ParseTree>),
    NotEqualTo(Box<ParseTree>, Box<ParseTree>),
    GreaterThan(Box<ParseTree>, Box<ParseTree>),
    GreaterThanOrEqualTo(Box<ParseTree>, Box<ParseTree>),
    LessThan(Box<ParseTree>, Box<ParseTree>),
    LessThanOrEqualTo(Box<ParseTree>, Box<ParseTree>),
    Not(Box<ParseTree>),
    And(Box<ParseTree>, Box<ParseTree>),
    Or(Box<ParseTree>, Box<ParseTree>),

    // Defining Objects
    Equ(String, Box<ParseTree>, Box<ParseTree>),
    LazyEqu(String, Box<ParseTree>, Box<ParseTree>),
    FunctionDefinition(Function, Box<ParseTree>),
    FunctionDeclaration(Function, Box<ParseTree>),
    LambdaDefinition(Function),

    // Functional Operations
    Compose(Box<ParseTree>, Box<ParseTree>),
    Id(Box<ParseTree>),

    // Branching
    If(Box<ParseTree>, Box<ParseTree>),
    IfElse(Box<ParseTree>, Box<ParseTree>, Box<ParseTree>),

    // Evaluations
    FunctionCall(String, Vec<ParseTree>),
    Variable(String),
    Constant(Value),

    // Type Casts
    IntCast(Box<ParseTree>),
    FloatCast(Box<ParseTree>),
    BoolCast(Box<ParseTree>),
    StringCast(Box<ParseTree>),

    // Misc
    Print(Box<ParseTree>),
}

macro_rules! one_arg {
    ($op:ident, $tokens:ident, $globals:ident, $locals:ident) => {
        Ok(ParseTree::$op(
            Box::new(ParseTree::parse($tokens, $globals, $locals)?)
    ))}
}

macro_rules! two_arg {
    ($op:ident, $tokens:ident, $globals:ident, $locals:ident) => {
        Ok(ParseTree::$op(
            Box::new(ParseTree::parse($tokens, $globals, $locals)?),
            Box::new(ParseTree::parse($tokens, $globals, $locals)?)
    ))}
}

macro_rules! three_arg {
    ($op:ident, $tokens:ident, $globals:ident, $locals:ident) => {
        Ok(ParseTree::$op(
            Box::new(ParseTree::parse($tokens, $globals, $locals)?),
            Box::new(ParseTree::parse($tokens, $globals, $locals)?),
            Box::new(ParseTree::parse($tokens, $globals, $locals)?)
    ))}
}

impl ParseTree {
    fn parse<I>(
        tokens: &mut I,
        globals: &HashMap<String, Function>,
        locals: &mut Cow<HashMap<String, Function>>) -> Result<Self, ParseError>
    where
        I: Iterator<Item = Result<Token, TokenizeError>>,
    {
        match tokens.next() {
            Some(Ok(token)) => {
                match token {
                    Token::Constant(c) => Ok(Self::Constant(c)),
                    Token::Identifier(ident) => {
                        // If it is found to be a function, get its argument count.
                        // During parsing, we only keep track of function definitions
                        // so that we know how many arguments it takes
                        if let Some(f) = locals.clone().get(&ident).or(globals.clone().get(&ident)) {
                            let args = f.t.1.iter()
                                .map(|_| ParseTree::parse(tokens, globals, locals)).collect::<Result<Vec<_>, ParseError>>()?;

                            Ok(ParseTree::FunctionCall(ident.clone(), args))
                        } else {
                            Ok(ParseTree::Variable(ident.clone()))
                        }
                    }
                    Token::Operator(op) => {
                        match op {
                            Op::Add => two_arg!(Add, tokens, globals, locals),
                            Op::Sub => two_arg!(Sub, tokens, globals, locals),
                            Op::Mul => two_arg!(Mul, tokens, globals, locals),
                            Op::Div => two_arg!(Div, tokens, globals, locals),
                            Op::Exp => two_arg!(Exp, tokens, globals, locals),
                            Op::Mod => two_arg!(Mod, tokens, globals, locals),
                            Op::Equ | Op::LazyEqu => {
                                let token = tokens.next()
                                    .ok_or(ParseError::UnexpectedEndInput)?
                                    .map_err(|e| ParseError::TokenizeError(e))?;

                                if let Token::Identifier(ident) = token {
                                    match op {
                                        Op::Equ => Ok(ParseTree::Equ(ident.clone(),
                                            Box::new(ParseTree::parse(tokens, globals, locals)?),
                                            Box::new(ParseTree::parse(tokens, globals, locals)?)
                                        )),
                                        Op::LazyEqu => Ok(ParseTree::LazyEqu(ident.clone(),
                                            Box::new(ParseTree::parse(tokens, globals, locals)?),
                                            Box::new(ParseTree::parse(tokens, globals, locals)?)
                                        )),
                                        _ => panic!("Operator literally changed under your nose"),
                                    }
                                } else {
                                    Err(ParseError::InvalidIdentifier)
                                }
                            }
                            Op::FunctionDefine(nargs) => {
                                let token = tokens.next()
                                    .ok_or(ParseError::UnexpectedEndInput)?
                                    .map_err(|e| ParseError::TokenizeError(e))?;

                                if let Token::Identifier(ident) = token {
                                    let args: Vec<String> = tokens.take(nargs)
                                        .map(|token| match token {
                                            Ok(Token::Identifier(ident)) => Ok(ident),
                                            Ok(_) => Err(ParseError::InvalidIdentifier),
                                            Err(e) => Err(ParseError::TokenizeError(e)),
                                        })
                                        .collect::<Result<Vec<_>, ParseError>>()?;

                                    let f = if locals.contains_key(&ident) {
                                        let locals = locals.to_mut();
                                        let f = locals.get(&ident).unwrap();
                                        let f = f.clone();

                                        // iterate over f's types and push them
                                        for (t, name) in std::iter::zip(f.t.1.clone(), args.clone()) {
                                            match t {
                                                Type::Function(finner) => {
                                                    locals.insert(name.clone(), Function::named(&name, finner, None, None));
                                                }
                                                _ => (),
                                            }
                                        }

                                        Function::named(
                                            &ident,
                                            f.t.clone(),
                                            Some(args),
                                            Some(Box::new(ParseTree::parse(tokens, globals, &mut Cow::Borrowed(&locals))?)))
                                    } else {
                                        let f = Function::named(
                                            &ident,
                                            FunctionType(Box::new(Type::Any), args.iter().map(|_| Type::Any).collect()),
                                            Some(args),
                                            Some(Box::new(ParseTree::parse(tokens, globals, &mut Cow::Borrowed(&locals))?)));

                                        let locals = locals.to_mut();

                                        locals.insert(ident.clone(), f.clone());

                                        f
                                    };

                                    Ok(ParseTree::FunctionDefinition(f,
                                        Box::new(ParseTree::parse(tokens, globals, locals)?)))
                                } else {
                                    Err(ParseError::InvalidIdentifier)
                                }
                            }
                            Op::Compose => two_arg!(Compose, tokens, globals, locals),
                            Op::Id => one_arg!(Id, tokens, globals, locals),
                            Op::If => two_arg!(If, tokens, globals, locals),
                            Op::IfElse => three_arg!(IfElse, tokens, globals, locals),
                            Op::EqualTo => two_arg!(EqualTo, tokens, globals, locals),
                            Op::GreaterThan => two_arg!(GreaterThan, tokens, globals, locals),
                            Op::LessThan => two_arg!(LessThan, tokens, globals, locals),
                            Op::GreaterThanOrEqualTo => two_arg!(GreaterThanOrEqualTo, tokens, globals, locals),
                            Op::LessThanOrEqualTo => two_arg!(LessThanOrEqualTo, tokens, globals, locals),
                            Op::Not => one_arg!(Not, tokens, globals, locals),
                            Op::IntCast => one_arg!(IntCast, tokens, globals, locals),
                            Op::FloatCast => one_arg!(FloatCast, tokens, globals, locals),
                            Op::BoolCast => one_arg!(BoolCast, tokens, globals, locals),
                            Op::StringCast => one_arg!(StringCast, tokens, globals, locals),
                            Op::Print => one_arg!(Print, tokens, globals, locals),
                            Op::OpenArray => {
                                let mut depth = 1;

                                // take tokens until we reach the end of this array
                                // if we don't collect them here it causes rust to overflow computing the types
                                let array_tokens = tokens.by_ref().take_while(|t| match t {
                                    Ok(Token::Operator(Op::OpenArray)) => {
                                        depth += 1;
                                        true
                                    },
                                    Ok(Token::Operator(Op::CloseArray)) => {
                                        depth -= 1;
                                        depth > 0
                                    }
                                    _ => true,
                                }).collect::<Result<Vec<_>, TokenizeError>>().map_err(|e| ParseError::TokenizeError(e))?;

                                let array_tokens: Vec<Result<Token, TokenizeError>> = array_tokens.into_iter().map(|t| Ok(t)).collect();

                                let trees: Vec<ParseTree> = Parser::new(array_tokens.into_iter())
                                    .globals(globals.clone())
                                    .locals(locals.to_mut().to_owned())
                                    .collect::<Result<_, ParseError>>()?;

                                let tree = trees.into_iter().fold(
                                    ParseTree::Constant(Value::Array(Type::Any, vec![])),
                                    |acc, x| ParseTree::Add(Box::new(acc), Box::new(x.clone())),
                                );

                                Ok(tree)
                            }
                            Op::Empty => Ok(ParseTree::Constant(Value::Array(Type::Any, vec![]))),
                            Op::CloseArray => Err(ParseError::UnmatchedArrayClose),
                            Op::NotEqualTo => two_arg!(NotEqualTo, tokens, globals, locals),
                            Op::And => two_arg!(And, tokens, globals, locals),
                            Op::Or => two_arg!(Or, tokens, globals, locals),
                            Op::FunctionDeclare(arg_count) => {
                                let name = match tokens.next()
                                    .ok_or(ParseError::UnexpectedEndInput)?
                                    .map_err(|e| ParseError::TokenizeError(e))?
                                {
                                        Token::Identifier(x) => x,
                                        _ => return Err(ParseError::InvalidIdentifier),
                                };

                                let args: Vec<Type> = (0..arg_count)
                                    .map(|_| Self::parse_type(tokens))
                                    .collect::<Result<_, ParseError>>()?;

                                let rett = Self::parse_type(tokens)?;

                                if locals.contains_key(&name) {
                                    println!("{name} already found: {locals:?}");
                                    return Err(ParseError::ImmutableError(name.clone()));
                                }

                                let f = Function::named(
                                    &name,
                                    FunctionType(Box::new(rett), args),
                                    None,
                                    None);

                                let locals = locals.to_mut();

                                locals.insert(name, f.clone());

                                Ok(ParseTree::FunctionDeclaration(
                                    f,
                                    Box::new(ParseTree::parse(tokens, globals, &mut Cow::Borrowed(&*locals))?)))
                            }
                            Op::LambdaDefine(arg_count) => {
                                let args: Vec<String> = tokens.take(arg_count)
                                    .map(|token| match token {
                                        Ok(Token::Identifier(ident)) => Ok(ident),
                                        Ok(_) => Err(ParseError::InvalidIdentifier),
                                        Err(e) => Err(ParseError::TokenizeError(e)),
                                    })
                                    .collect::<Result<Vec<_>, ParseError>>()?;

                                Ok(ParseTree::LambdaDefinition(
                                    Function::lambda(
                                        FunctionType(Box::new(Type::Any), args.clone().into_iter().map(|_| Type::Any).collect()),
                                        args,
                                        Some(Box::new(ParseTree::parse(tokens, globals, &mut Cow::Borrowed(&*locals))?)))))
                            }
                            Op::NonCall => {
                                let ident = match tokens.next().ok_or(ParseError::UnexpectedEndInput)?
                                    .map_err(|e| ParseError::TokenizeError(e))?
                                {
                                    Token::Identifier(x) => x,
                                    _ => return Err(ParseError::InvalidIdentifier),
                                };

                                if let Some(f) = locals.clone().get(&ident).or(globals.clone().get(&ident)).cloned() {
                                    Ok(ParseTree::Constant(Value::Function(f)))
                                } else {
                                    Err(ParseError::FunctionUndefined(ident.clone()))
                                }
                            }
                        }
                    }
                    t => Err(ParseError::UnwantedToken(t)),
                }
            },
            Some(Err(e)) => Err(ParseError::TokenizeError(e)),
            None => Err(ParseError::NoInput),
        }
    }

    fn parse_type<I>(tokens: &mut I) -> Result<Type, ParseError>
    where
        I: Iterator<Item = Result<Token, TokenizeError>>,
    {
        match tokens.next() {
            Some(Ok(Token::Type(t))) => Ok(t),
            Some(Ok(Token::Operator(Op::FunctionDefine(n)))) => {
                let args: Vec<Type> = (0..n)
                    .map(|_| Self::parse_type(tokens))
                    .collect::<Result<_, ParseError>>()?;

                let rett = Self::parse_type(tokens)?;

                Ok(Type::Function(FunctionType(Box::new(rett), args.clone())))
            },
            Some(Ok(Token::Operator(Op::OpenArray))) => {
                let t = Self::parse_type(tokens)?;
                let _ = match tokens.next() {
                    Some(Ok(Token::Operator(Op::CloseArray))) => (),
                    _ => return Err(ParseError::UnmatchedArrayClose),
                };

                Ok(Type::Array(Box::new(t)))
            }
            Some(Ok(t)) => Err(ParseError::UnwantedToken(t.clone())),
            Some(Err(e)) => Err(ParseError::TokenizeError(e)),
            None => Err(ParseError::UnexpectedEndInput),
        }
    }
}

/// Parses input tokens and produces ParseTrees for an Executor
pub(crate) struct Parser<I: Iterator<Item = Result<Token, TokenizeError>>> {
    tokens: I,

    // These are used to keep track of functions in the current context
    // by the parser. otherwise the parser would have no way to tell
    // if the program `* a b 12` is supposed to be ((* a b) (12)) or (* (a b) 12)
    globals: HashMap<String, Function>,
    locals: HashMap<String, Function>,
}

impl<I: Iterator<Item = Result<Token, TokenizeError>>> Parser<I> {
    pub fn new(tokens: I) -> Self {
        Self {
            tokens,
            globals: HashMap::new(),
            locals: HashMap::new()
        }
    }

    pub fn globals(self, globals: HashMap<String, Function>) -> Self {
        Self {
            tokens: self.tokens,
            globals,
            locals: self.locals,
        }
    }

    pub fn locals(self, locals: HashMap<String, Function>) -> Self {
        Self {
            tokens: self.tokens,
            globals: self.globals,
            locals,
        }
    }
}

impl<I: Iterator<Item = Result<Token, TokenizeError>>> Iterator for Parser<I> {
    type Item = Result<ParseTree, ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        let tree = ParseTree::parse(&mut self.tokens, &self.globals, &mut Cow::Borrowed(&self.locals));

        match tree {
            Ok(tree) => Some(Ok(tree)),
            Err(e) => {
                match e {
                    ParseError::NoInput => None,
                    _ => Some(Err(e)),
                }
            }
        }
    }
}
