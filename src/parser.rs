use super::{Type, Value, FunctionDeclaration};
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
    TokenizeError(TokenizeError),
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
            ParseError::TokenizeError(e) => write!(f, "{e}"),
        }
    }
}

impl error::Error for ParseError {}

#[derive(Clone, Debug)]
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
    GreaterThan(Box<ParseTree>, Box<ParseTree>),
    GreaterThanOrEqualTo(Box<ParseTree>, Box<ParseTree>),
    LessThan(Box<ParseTree>, Box<ParseTree>),
    LessThanOrEqualTo(Box<ParseTree>, Box<ParseTree>),
    Not(Box<ParseTree>),

    // Defining Objects
    Equ(String, Box<ParseTree>, Box<ParseTree>),
    LazyEqu(String, Box<ParseTree>, Box<ParseTree>),
    FunctionDefinition(String, Vec<(String, Type)>, Type, Box<ParseTree>, Box<ParseTree>),

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

impl ParseTree {
    fn parse<I>(
        tokens: &mut I,
        globals: &HashMap<String, FunctionDeclaration>,
        locals: &mut Cow<HashMap<String, FunctionDeclaration>>) -> Result<Self, ParseError>
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
                        if let Some(decl) = locals.clone().get(&ident).or(globals.clone().get(&ident)) {
                            let args = decl.args.iter()
                                .map(|_| ParseTree::parse(tokens, globals, locals)).collect::<Result<Vec<_>, ParseError>>()?;

                            Ok(ParseTree::FunctionCall(ident.clone(), args))
                        } else {
                            Ok(ParseTree::Variable(ident.clone()))
                        }
                    }
                    Token::Operator(op) => {
                        match op {
                            Op::Add => Ok(ParseTree::Add(
                                Box::new(ParseTree::parse(tokens, globals, locals)?),
                                Box::new(ParseTree::parse(tokens, globals, locals)?)
                            )),
                            Op::Sub => Ok(ParseTree::Sub(
                                Box::new(ParseTree::parse(tokens, globals, locals)?),
                                Box::new(ParseTree::parse(tokens, globals, locals)?)
                            )),
                            Op::Mul => Ok(ParseTree::Mul(
                                Box::new(ParseTree::parse(tokens, globals, locals)?),
                                Box::new(ParseTree::parse(tokens, globals, locals)?)
                            )),
                            Op::Div => Ok(ParseTree::Div(
                                Box::new(ParseTree::parse(tokens, globals, locals)?),
                                Box::new(ParseTree::parse(tokens, globals, locals)?)
                            )),
                            Op::Exp => Ok(ParseTree::Exp(
                                Box::new(ParseTree::parse(tokens, globals, locals)?),
                                Box::new(ParseTree::parse(tokens, globals, locals)?)
                            )),
                            Op::Mod => Ok(ParseTree::Mod(
                                Box::new(ParseTree::parse(tokens, globals, locals)?),
                                Box::new(ParseTree::parse(tokens, globals, locals)?)
                            )),
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
                            Op::FunctionDeclare(nargs) => {
                                let token = tokens.next()
                                    .ok_or(ParseError::UnexpectedEndInput)?
                                    .map_err(|e| ParseError::TokenizeError(e))?;

                                    if let Token::Identifier(ident) = token {
                                        let args: Vec<(String, Type)> = tokens.take(nargs)
                                            .map(|token| match token {
                                                Ok(Token::Identifier(ident)) => Ok((ident, Type::Any)),
                                                Ok(_) => Err(ParseError::InvalidIdentifier),
                                                Err(e) => Err(ParseError::TokenizeError(e)),
                                            })
                                            .collect::<Result<Vec<_>, ParseError>>()?;

                                        let locals = locals.to_mut();

                                        locals.insert(ident.clone(), FunctionDeclaration {
                                            _name: ident.clone(),
                                            _r: Type::Any,
                                            args: args.clone(),
                                        });

                                        Ok(ParseTree::FunctionDefinition(
                                            ident,
                                            args,
                                            Type::Any,
                                            Box::new(ParseTree::parse(tokens, globals, &mut Cow::Borrowed(&*locals))?),
                                            Box::new(ParseTree::parse(tokens, globals, &mut Cow::Borrowed(&*locals))?)))
                                    } else {
                                        Err(ParseError::InvalidIdentifier)
                                    }
                            }
                            Op::Compose => Ok(ParseTree::Compose(
                                Box::new(ParseTree::parse(tokens, globals, locals)?),
                                Box::new(ParseTree::parse(tokens, globals, locals)?)
                            )),
                            Op::Id => Ok(ParseTree::Id(
                                Box::new(ParseTree::parse(tokens, globals, locals)?)
                            )),
                            Op::If => Ok(ParseTree::If(
                                Box::new(ParseTree::parse(tokens, globals, locals)?),
                                Box::new(ParseTree::parse(tokens, globals, locals)?)
                            )),
                            Op::IfElse => Ok(ParseTree::IfElse(
                                Box::new(ParseTree::parse(tokens, globals, locals)?),
                                Box::new(ParseTree::parse(tokens, globals, locals)?),
                                Box::new(ParseTree::parse(tokens, globals, locals)?)
                            )),
                            Op::EqualTo => Ok(ParseTree::EqualTo(
                                Box::new(ParseTree::parse(tokens, globals, locals)?),
                                Box::new(ParseTree::parse(tokens, globals, locals)?)
                            )),
                            Op::GreaterThan => Ok(ParseTree::GreaterThan(
                                Box::new(ParseTree::parse(tokens, globals, locals)?),
                                Box::new(ParseTree::parse(tokens, globals, locals)?)
                            )),
                            Op::LessThan => Ok(ParseTree::LessThan(
                                Box::new(ParseTree::parse(tokens, globals, locals)?),
                                Box::new(ParseTree::parse(tokens, globals, locals)?)
                            )),
                            Op::GreaterThanOrEqualTo => Ok(ParseTree::GreaterThanOrEqualTo(
                                Box::new(ParseTree::parse(tokens, globals, locals)?),
                                Box::new(ParseTree::parse(tokens, globals, locals)?)
                            )),
                            Op::LessThanOrEqualTo => Ok(ParseTree::LessThanOrEqualTo(
                                Box::new(ParseTree::parse(tokens, globals, locals)?),
                                Box::new(ParseTree::parse(tokens, globals, locals)?)
                            )),
                            Op::Not => Ok(ParseTree::Not(Box::new(ParseTree::parse(tokens, globals, locals)?))),
                            Op::IntCast => Ok(ParseTree::IntCast(Box::new(ParseTree::parse(tokens, globals, locals)?))),
                            Op::FloatCast => Ok(ParseTree::FloatCast(Box::new(ParseTree::parse(tokens, globals, locals)?))),
                            Op::BoolCast => Ok(ParseTree::BoolCast(Box::new(ParseTree::parse(tokens, globals, locals)?))),
                            Op::StringCast => Ok(ParseTree::StringCast(Box::new(ParseTree::parse(tokens, globals, locals)?))),
                            Op::Print => Ok(ParseTree::Print(Box::new(ParseTree::parse(tokens, globals, locals)?)))
                        }
                    }
                }
            },
            Some(Err(e)) => Err(ParseError::TokenizeError(e)),
            None => Err(ParseError::NoInput),
        }
    }
}

/// Parses input tokens and produces ParseTrees for an Executor
pub(crate) struct Parser<I: Iterator<Item = Result<Token, TokenizeError>>> {
    tokens: I,

    // These are used to keep track of functions in the current context
    // by the parser. otherwise the parser would have no way to tell
    // if the program `* a b 12` is supposed to be ((* a b) (12)) or (* (a b) 12)
    globals: HashMap<String, FunctionDeclaration>,
    locals: HashMap<String, FunctionDeclaration>,
}

impl<I: Iterator<Item = Result<Token, TokenizeError>>> Parser<I> {
    pub fn new(tokens: I) -> Self {
        Self {
            tokens,
            globals: HashMap::new(),
            locals: HashMap::new()
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
