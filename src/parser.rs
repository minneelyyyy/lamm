
use crate::executor::Executor;

use super::{Value, Type, Function, FunctionType};
use super::tokenizer::{Token, TokenizeError, Op};

use std::borrow::{BorrowMut, Cow};
use std::error;
use std::collections::HashMap;
use std::fmt::Display;
use std::iter::Peekable;

#[derive(Debug)]
pub enum ParseError {
    NoInput,
    UnexpectedEndInput,
    IdentifierUndefined(String),
    InvalidIdentifier(Token),
    UnmatchedArrayClose,
    UnwantedToken(Token),
    TokenizeError(TokenizeError),
    ImmutableError(String),
}

impl Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::UnexpectedEndInput => write!(f, "Input ended unexpectedly"),
            ParseError::IdentifierUndefined(name) => write!(f, "Undefined identifier `{name}`"),
            ParseError::InvalidIdentifier(t) => write!(f, "Invalid identifier `{t:?}`"),
            ParseError::NoInput => write!(f, "No input given"),
            ParseError::UnmatchedArrayClose => write!(f, "there was an unmatched array closing operator `]`"),
            ParseError::TokenizeError(e) => write!(f, "Tokenizer Error: {e}"),
            ParseError::ImmutableError(i) => write!(f, "attempt to redeclare {i} met with force"),
            ParseError::UnwantedToken(t) => write!(f, "unexpected token {t:?}"),
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
    LambdaDefinition(Function),

    // Functional Operations
    Compose(Box<ParseTree>, Box<ParseTree>),
    Id(Box<ParseTree>),
    Head(Box<ParseTree>),
    Tail(Box<ParseTree>),
    Init(Box<ParseTree>),
    Fini(Box<ParseTree>),

    // Branching
    If(Box<ParseTree>, Box<ParseTree>),
    IfElse(Box<ParseTree>, Box<ParseTree>, Box<ParseTree>),

    // Evaluations
    FunctionCall(String, Vec<ParseTree>),
    Variable(String),
    Constant(Value),
    NonCall(String),

    // Type Casts
    IntCast(Box<ParseTree>),
    FloatCast(Box<ParseTree>),
    BoolCast(Box<ParseTree>),
    StringCast(Box<ParseTree>),

    // Misc
    Print(Box<ParseTree>),
    Nop,
    Export(Vec<String>),
}

/// Parses input tokens and produces ParseTrees for an Executor
pub(crate) struct Parser<'a, I: Iterator<Item = Result<Token, TokenizeError>>> {
    tokens: &'a mut Peekable<I>,
    globals: &'a mut HashMap<String, Type>,
    locals: HashMap<String, Type>,
}

impl<'a, I: Iterator<Item = Result<Token, TokenizeError>>> Parser<'a, I> {
    pub fn new(tokens: &'a mut Peekable<I>, globals: &'a mut HashMap<String, Type>) -> Self {
        Self {
            tokens: tokens,
            globals,
            locals: HashMap::new()
        }
    }

    pub fn add_global(self, k: String, v: Type) -> Self {
        self.globals.insert(k, v);
        self
    }

    pub fn add_globals<Items: Iterator<Item = (String, Type)>>(self, items: Items) -> Self {
        items.for_each(|(name, t)| _ = self.globals.insert(name, t));
        self
    }

    pub fn locals(mut self, locals: HashMap<String, Type>) -> Self {
        self.locals = locals;
        self
    }

    pub fn add_local(mut self, k: String, v: Type) -> Self {
        self.locals.insert(k, v);
        self
    }

    pub fn add_locals<Items: Iterator<Item = (String, Type)>>(mut self, items: Items) -> Self {
        items.for_each(|(name, t)| _ = self.locals.insert(name, t));
        self
    }

    fn get_object_type(&self, ident: &String) -> Result<&Type, ParseError> {
        self.locals.get(ident).or(self.globals.get(ident))
            .ok_or(ParseError::IdentifierUndefined(ident.clone()))
    }

    fn get_object_types<Names: Iterator<Item = String>>(&self, items: Names) -> impl Iterator<Item = Result<&Type, ParseError>> {
        items.map(|x| self.get_object_type(&x))
    }

    fn parse(&mut self) -> Result<ParseTree, ParseError> {
        match self.tokens.next().ok_or(ParseError::NoInput)?.map_err(|e| ParseError::TokenizeError(e))? {
            Token::Constant(c) => Ok(ParseTree::Constant(c)),
            Token::Identifier(ident) => {
                match self.get_object_type(&ident)? {
                    Type::Function(f) => {
                        let args = f.1.clone().iter()
                            .map(|_| self.parse()).collect::<Result<Vec<_>, ParseError>>()?;

                        Ok(ParseTree::FunctionCall(ident, args))
                    }
                    _ => Ok(ParseTree::Variable(ident)),
                }
            }
            Token::Operator(op) => {
                match op {
                    Op::Add => Ok(ParseTree::Add(Box::new(self.parse()?), Box::new(self.parse()?))),
                    Op::Sub => Ok(ParseTree::Sub(Box::new(self.parse()?), Box::new(self.parse()?))),
                    Op::Mul => Ok(ParseTree::Mul(Box::new(self.parse()?), Box::new(self.parse()?))),
                    Op::Div => Ok(ParseTree::Div(Box::new(self.parse()?), Box::new(self.parse()?))),
                    Op::Exp => Ok(ParseTree::Exp(Box::new(self.parse()?), Box::new(self.parse()?))),
                    Op::Mod => Ok(ParseTree::Mod(Box::new(self.parse()?), Box::new(self.parse()?))),
                    Op::Equ | Op::LazyEqu => {
                        let token = self.tokens.next()
                            .ok_or(ParseError::UnexpectedEndInput)?
                            .map_err(|e| ParseError::TokenizeError(e))?;

                        let body = Box::new(self.parse()?);

                        if let Token::Identifier(ident) = token {
                            match op {
                                Op::Equ => Ok(ParseTree::Equ(ident.clone(),
                                    body,
                                    Box::new(Parser::new(self.tokens.by_ref(), self.globals.borrow_mut())
                                        .locals(self.locals.clone())
                                        .add_local(ident, Type::Any)
                                        .parse()?))
                                ),
                                Op::LazyEqu => Ok(ParseTree::LazyEqu(ident.clone(),
                                    body,
                                    Box::new(Parser::new(self.tokens.by_ref(), self.globals.borrow_mut())
                                        .locals(self.locals.clone())
                                        .add_local(ident, Type::Any)
                                        .parse()?))
                                ),
                                _ => unreachable!(),
                            }
                        } else {
                            Err(ParseError::InvalidIdentifier(token))
                        }
                    }
                    Op::FunctionDefine(arg_count) => {
                        let f = self.parse_function(arg_count)?;

                        Ok(ParseTree::FunctionDefinition(f.clone(),
                            Box::new(
                                Parser::new(self.tokens, self.globals.borrow_mut())
                                .locals(self.locals.clone())
                                .add_local(f.name().unwrap().to_string(), Type::Function(f.get_type()))
                                .parse()?
                            )))
                    },
                    Op::Compose => Ok(ParseTree::Compose(Box::new(self.parse()?), Box::new(self.parse()?))),
                    Op::Id => Ok(ParseTree::Id(Box::new(self.parse()?))),
                    Op::IfElse => Ok(ParseTree::IfElse(Box::new(self.parse()?), Box::new(self.parse()?), Box::new(self.parse()?))),
                    Op::If => Ok(ParseTree::If(Box::new(self.parse()?), Box::new(self.parse()?))),
                    Op::EqualTo => Ok(ParseTree::EqualTo(Box::new(self.parse()?), Box::new(self.parse()?))),
                    Op::GreaterThan => Ok(ParseTree::GreaterThan(Box::new(self.parse()?), Box::new(self.parse()?))),
                    Op::LessThan => Ok(ParseTree::LessThan(Box::new(self.parse()?), Box::new(self.parse()?))),
                    Op::GreaterThanOrEqualTo => Ok(ParseTree::GreaterThanOrEqualTo(Box::new(self.parse()?), Box::new(self.parse()?))),
                    Op::LessThanOrEqualTo => Ok(ParseTree::LessThanOrEqualTo(Box::new(self.parse()?), Box::new(self.parse()?))),
                    Op::Not => Ok(ParseTree::Not(Box::new(self.parse()?))),
                    Op::IntCast => Ok(ParseTree::IntCast(Box::new(self.parse()?))),
                    Op::FloatCast => Ok(ParseTree::FloatCast(Box::new(self.parse()?))),
                    Op::BoolCast => Ok(ParseTree::BoolCast(Box::new(self.parse()?))),
                    Op::StringCast => Ok(ParseTree::StringCast(Box::new(self.parse()?))),
                    Op::Print => Ok(ParseTree::Print(Box::new(self.parse()?))),
                    Op::OpenArray => {
                        let mut depth = 1;

                        // take tokens until we reach the end of this array
                        // if we don't collect them here it causes rust to overflow computing the types
                        let array_tokens = self.tokens.by_ref().take_while(|t| match t {
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

                        let mut array_tokens = array_tokens
                            .into_iter()
                            .map(|t| Ok(t))
                            .collect::<Vec<Result<Token, TokenizeError>>>()
                            .into_iter()
                            .peekable();

                        let trees: Vec<ParseTree> = Parser::new(&mut array_tokens, self.globals.borrow_mut())
                            .locals(self.locals.to_owned())
                            .collect::<Result<_, ParseError>>()?;

                        let tree = trees.into_iter().fold(
                            ParseTree::Constant(Value::Array(Type::Any, vec![])),
                            |acc, x| ParseTree::Add(Box::new(acc), Box::new(x.clone())),
                        );

                        Ok(tree)
                    }
                    Op::OpenStatement => {
                        let mut depth = 1;

                        // take tokens until we reach the end of this array
                        // if we don't collect them here it causes rust to overflow computing the types
                        let tokens = self.tokens.by_ref().take_while(|t| match t {
                            Ok(Token::Operator(Op::OpenStatement)) => {
                                depth += 1;
                                true
                            },
                            Ok(Token::Operator(Op::CloseStatement)) => {
                                depth -= 1;
                                depth > 0
                            }
                            _ => true,
                        }).collect::<Result<Vec<_>, TokenizeError>>().map_err(|e| ParseError::TokenizeError(e))?;

                        let mut tokens = tokens
                            .into_iter()
                            .map(|t| Ok(t))
                            .collect::<Vec<Result<Token, TokenizeError>>>()
                            .into_iter()
                            .peekable();

                        let trees: Vec<ParseTree> = Parser::new(&mut tokens, self.globals.borrow_mut())
                            .locals(self.locals.to_owned())
                            .collect::<Result<_, ParseError>>()?;

                        let tree = trees.into_iter().fold(
                            ParseTree::Nop,
                            |acc, x| ParseTree::Compose(Box::new(acc), Box::new(x.clone())),
                        );

                        Ok(tree)
                    }
                    Op::Empty => Ok(ParseTree::Constant(Value::Array(Type::Any, vec![]))),
                    Op::CloseArray => Err(ParseError::UnmatchedArrayClose),
                    Op::NotEqualTo => Ok(ParseTree::NotEqualTo(Box::new(self.parse()?), Box::new(self.parse()?))),
                    Op::And => Ok(ParseTree::And(Box::new(self.parse()?), Box::new(self.parse()?))),
                    Op::Or => Ok(ParseTree::Or(Box::new(self.parse()?), Box::new(self.parse()?))),
                    Op::LambdaDefine(arg_count) => {
                        let f = self.parse_lambda(arg_count)?;
                        Ok(ParseTree::LambdaDefinition(f))
                    }
                    Op::NonCall => {
                        let name = Self::get_identifier(self.tokens.next())?;
                        Ok(ParseTree::NonCall(name))
                    },
                    Op::Head => Ok(ParseTree::Head(Box::new(self.parse()?))),
                    Op::Tail => Ok(ParseTree::Tail(Box::new(self.parse()?))),
                    Op::Init => Ok(ParseTree::Init(Box::new(self.parse()?))),
                    Op::Fini => Ok(ParseTree::Fini(Box::new(self.parse()?))),
                    Op::Export => {
                        let list = self.parse()?;
                        let mut g = HashMap::new();
                        let list = Executor::new(&mut vec![Ok(list)].into_iter(), &mut g).next().unwrap().map_err(|_| ParseError::NoInput)?;

                        if let Value::Array(Type::String, items) = list {
                            let names = items.into_iter().map(|x| match x {
                                Value::String(s) => s,
                                _ => unreachable!(),
                            });

                            for name in names.clone() {
                                let t = self.locals.remove(&name).ok_or(ParseError::IdentifierUndefined(name.clone()))?;
                                self.globals.insert(name, t);
                            }

                            Ok(ParseTree::Export(names.collect()))
                        } else {
                            Err(ParseError::NoInput)
                        }
                    }
                    op => Err(ParseError::UnwantedToken(Token::Operator(op))),
                }
            }
            t => Err(ParseError::UnwantedToken(t)),
        }
    }

    fn parse_lambda(&mut self, arg_count: usize) -> Result<Function, ParseError> {
        let (t, args) = Self::parse_function_declaration(self.tokens, arg_count)?;

        let mut locals = self.locals.clone();

        for (name, t) in std::iter::zip(args.iter(), t.1.iter()) {
            locals.insert(name.clone(), t.clone());
        }

        Ok(Function::lambda(t, args, Box::new(
            Parser::new(self.tokens, &mut self.globals)
                .locals(locals).parse()?)))
    }

    fn parse_function(&mut self, arg_count: usize) -> Result<Function, ParseError> {
        let name = Self::get_identifier(self.tokens.next())?;
        let (t, args) = Self::parse_function_declaration(self.tokens, arg_count)?;

        let mut locals = self.locals.clone();

        for (name, t) in std::iter::zip(args.iter(), t.1.iter()) {
            locals.insert(name.clone(), t.clone());
        }

        locals.insert(name.clone(), Type::Function(t.clone()));

        Ok(Function::named(&name, t, args, Box::new(
            Parser::new(self.tokens, &mut self.globals)
                .locals(locals).parse()?)))
    }

    fn parse_function_declaration(tokens: &mut Peekable<I>, arg_count: usize) -> Result<(FunctionType, Vec<String>), ParseError> {
        let args: Vec<(Type, String)> = (0..arg_count)
            .map(|_| Self::parse_function_declaration_parameter(tokens))
            .collect::<Result<_, _>>()?;

        let (types, names): (Vec<_>, Vec<_>) = args.into_iter().unzip();
        let mut ret = Type::Any;

        if tokens.next_if(|x| matches!(x, Ok(Token::Operator(Op::Arrow)))).is_some() {
            ret = Self::parse_type(tokens)?;
        }

        Ok((FunctionType(Box::new(ret), types), names))
    }

    fn parse_function_declaration_parameter(mut tokens: &mut Peekable<I>) -> Result<(Type, String), ParseError> {
        match tokens.next() {
            // untyped variable
            Some(Ok(Token::Identifier(x))) => Ok((Type::Any, x)),

            // typed variable
            Some(Ok(Token::Operator(Op::TypeDeclaration))) => {
                let name = Self::get_identifier(tokens.next())?;
                let t = Self::parse_type(&mut tokens)?;

                Ok((t, name))
            }

            // untyped function (all args Any, return type Any)
            Some(Ok(Token::Operator(Op::FunctionDefine(n)))) => {
                let name = Self::get_identifier(tokens.next())?;
                let args = (0..n).map(|_| Type::Any).collect();

                Ok((Type::Function(FunctionType(Box::new(Type::Any), args)), name))
            }

            // typed function
            Some(Ok(Token::Operator(Op::FunctionDeclare(n)))) => {
                let name = Self::get_identifier(tokens.next())?;
                let args = (0..n).map(|_| Self::parse_type(&mut tokens)).collect::<Result<_, _>>()?;
                let mut ret = Type::Any;

                // this is annoying
                // inside of the next_if closure, we already can know that its an error
                // and return it, but we cannot return out of a closure
                if let Some(t) = tokens.next_if(|x| matches!(x, Ok(Token::Operator(Op::Arrow))))
                {
                    // so we just check for an error here. this is the only reason t exists.
                    if let Err(e) = t {
                        return Err(ParseError::TokenizeError(e));
                    }

                    ret = Self::parse_type(&mut tokens)?;
                }

                Ok((Type::Function(FunctionType(Box::new(ret), args)), name))
            }

            Some(Ok(t)) => Err(ParseError::UnwantedToken(t)),
            Some(Err(e)) => Err(ParseError::TokenizeError(e)),
            None => Err(ParseError::UnexpectedEndInput),
        }
    }

    fn parse_type(tokens: &mut Peekable<I>) -> Result<Type, ParseError> {
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

    fn get_identifier(t: Option<Result<Token, TokenizeError>>) -> Result<String, ParseError> {
        match t.ok_or(ParseError::UnexpectedEndInput)?
            .map_err(|e| ParseError::TokenizeError(e))
        {
            Ok(Token::Identifier(ident)) => Ok(ident),
            Ok(t) => Err(ParseError::InvalidIdentifier(t)),
            Err(e) => Err(e),
        }
    }
}

impl<'a, I: Iterator<Item = Result<Token, TokenizeError>>> Iterator for Parser<'a, I> {
    type Item = Result<ParseTree, ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        let tree = self.parse();

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
