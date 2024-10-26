
use crate::executor::Executor;

use super::{Value, Type, Function, FunctionType};
use super::tokenizer::{Token, TokenizeError, Op};

use std::borrow::BorrowMut;
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
    RuntimeError,
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
            ParseError::RuntimeError => write!(f, "Runtime Error"),
            ParseError::ImmutableError(i) => write!(f, "attempt to redeclare {i} met with force"),
            ParseError::UnwantedToken(t) => write!(f, "unexpected token {t:?}"),
        }
    }
}

impl error::Error for ParseError {}

#[derive(Clone, Debug)]
pub(crate) enum ParseTree {
    Operator(Op, Vec<ParseTree>),

    // Defining Objects
    Equ(String, Box<ParseTree>, Box<ParseTree>),
    LazyEqu(String, Box<ParseTree>, Box<ParseTree>),
    FunctionDefinition(Function, Box<ParseTree>),
    LambdaDefinition(Function),

    // Control Flow
    If(Box<ParseTree>, Box<ParseTree>),
    IfElse(Box<ParseTree>, Box<ParseTree>, Box<ParseTree>),

    // Evaluations
    FunctionCall(String, Vec<ParseTree>),
    _FunctionCallLocal(usize, Vec<ParseTree>),
    Variable(String),
    _Local(usize),
    Value(Value),
    GeneratedFunction(Function),

    Nop,
    NonCall(String),
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
            tokens,
            globals,
            locals: HashMap::new()
        }
    }


    pub fn _add_global(self, k: String, v: Type) -> Self {
        self.globals.insert(k, v);
        self
    }

    pub fn _add_globals<Items: Iterator<Item = (String, Type)>>(self, items: Items) -> Self {
        items.for_each(|(name, t)| {
            self.globals.insert(name, t);
        });
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

    pub fn _add_locals<Items: Iterator<Item = (String, Type)>>(mut self, items: Items) -> Self {
        items.for_each(|(name, t)| {
            self.locals.insert(name, t);
        });
        self
    }

    fn get_object_type(&self, ident: &String) -> Result<&Type, ParseError> {
        self.locals.get(ident).or(self.globals.get(ident))
            .ok_or(ParseError::IdentifierUndefined(ident.clone()))
    }

    fn _get_object_types<Names: Iterator<Item = String>>(&self, items: Names) -> impl Iterator<Item = Result<&Type, ParseError>> {
        items.map(|x| self.get_object_type(&x))
    }

    // get at most count arguments
    fn get_args(&mut self, count: usize) -> Result<Vec<ParseTree>, ParseError> {
        (0..count).map_while(|_| match self.parse() {
            Ok(r) => Some(Ok(r)),
            Err(ParseError::NoInput) => None,
            Err(e) => Some(Err(e)),
        }).collect()
    }

    fn parse_operator(&mut self, op: Op) -> Result<ParseTree, ParseError> {
        let operators: HashMap<Op, FunctionType> = HashMap::from([
            (Op::Add, FunctionType(Box::new(Type::Any), vec![Type::Any, Type::Any])),
            (Op::Sub, FunctionType(Box::new(Type::Any), vec![Type::Any, Type::Any])),
            (Op::Mul, FunctionType(Box::new(Type::Any), vec![Type::Any, Type::Any])),
            (Op::Div, FunctionType(Box::new(Type::Float), vec![Type::Any, Type::Any])),
            (Op::FloorDiv, FunctionType(Box::new(Type::Int), vec![Type::Any, Type::Any])),
            (Op::Exp, FunctionType(Box::new(Type::Any), vec![Type::Any, Type::Any])),
            (Op::Mod, FunctionType(Box::new(Type::Any), vec![Type::Any, Type::Any])),
            (Op::Id, FunctionType(Box::new(Type::Any), vec![Type::Any])),
            (Op::GreaterThan, FunctionType(Box::new(Type::Bool), vec![Type::Any, Type::Any])),
            (Op::LessThan, FunctionType(Box::new(Type::Bool), vec![Type::Any, Type::Any])),
            (Op::EqualTo, FunctionType(Box::new(Type::Bool), vec![Type::Any, Type::Any])),
            (Op::NotEqualTo, FunctionType(Box::new(Type::Bool), vec![Type::Any, Type::Any])),
            (Op::GreaterThanOrEqualTo, FunctionType(Box::new(Type::Bool), vec![Type::Any, Type::Any])),
            (Op::LessThanOrEqualTo, FunctionType(Box::new(Type::Bool), vec![Type::Any, Type::Any])),
            (Op::Not, FunctionType(Box::new(Type::Bool), vec![Type::Bool])),
            (Op::And, FunctionType(Box::new(Type::Bool), vec![Type::Bool, Type::Bool])),
            (Op::Or, FunctionType(Box::new(Type::Bool), vec![Type::Bool, Type::Bool])),
            (Op::Head, FunctionType(Box::new(Type::Any), vec![Type::Array(Box::new(Type::Any))])),
            (Op::Tail, FunctionType(Box::new(Type::Array(Box::new(Type::Any))), vec![Type::Array(Box::new(Type::Any))])),
            (Op::Init, FunctionType(Box::new(Type::Array(Box::new(Type::Any))), vec![Type::Array(Box::new(Type::Any))])),
            (Op::Fini, FunctionType(Box::new(Type::Any), vec![Type::Array(Box::new(Type::Any))])),
            (Op::Print, FunctionType(Box::new(Type::Nil), vec![Type::Any])),
            (Op::IntCast, FunctionType(Box::new(Type::Int), vec![Type::Any])),
            (Op::FloatCast, FunctionType(Box::new(Type::Float), vec![Type::Any])),
            (Op::BoolCast, FunctionType(Box::new(Type::Bool), vec![Type::Any])),
            (Op::StringCast, FunctionType(Box::new(Type::String), vec![Type::Any])),
        ]);

        let operator = operators.get(&op).expect("All operators should be accounted for");
        let args = self.get_args(operator.1.len())?;

        if args.len() == operator.1.len() {
            Ok(ParseTree::Operator(op, args))
        } else {
            let mut counter = 0;
            let func_args: Vec<Type> = operator.1.iter().skip(args.len()).cloned().collect();
            let (names, types): (Vec<String>, Vec<Type>) = func_args
                .into_iter()
                .map(|t| {
                    counter += 1;
                    (format!("{counter}"), t)
                }).unzip();
            let function_type = FunctionType(operator.0.clone(), types);

            Ok(ParseTree::GeneratedFunction(Function::lambda(
                function_type,
                names.clone(),
                Box::new(ParseTree::Operator(op,
                    vec![
                        args,
                        names.into_iter().map(|x| ParseTree::Variable(x)).collect()
                    ].concat())))))
        }
    }

    fn parse(&mut self) -> Result<ParseTree, ParseError> {
        let token = self.tokens.next()
            .ok_or(ParseError::NoInput)?
            .map_err(|e| ParseError::TokenizeError(e))?;

        match token {
            Token::Constant(c) => Ok(ParseTree::Value(c)),
            Token::Identifier(ident) => {
                match self.get_object_type(&ident)? {
                    Type::Function(f) => {
                        let f = f.clone();
                        let args = self.get_args(f.1.len())?;

                        if args.len() < f.1.len() {
                            let mut counter = 0;
                            let func_args: Vec<Type> = f.1.iter().skip(args.len()).cloned().collect();
                            let (names, types): (Vec<String>, Vec<Type>) = func_args
                                .into_iter()
                                .map(|t| {
                                    counter += 1;
                                    (format!("{counter}"), t)
                                }).unzip();
                            let function_type = FunctionType(f.0.clone(), types);

                            Ok(ParseTree::Value(Value::Function(Function::lambda(
                                function_type,
                                names.clone(),
                                Box::new(ParseTree::FunctionCall(ident,
                                    vec![
                                        args,
                                        names.into_iter().map(|x| ParseTree::Variable(x)).collect()
                                    ].concat()))))))
                        } else {
                            Ok(ParseTree::FunctionCall(ident, args))
                        }
                    }
                    _ => Ok(ParseTree::Variable(ident)),
                }
            },
            Token::Operator(op) => match op {
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
                        ParseTree::Value(Value::Array(Type::Any, vec![])),
                        |acc, x| ParseTree::Operator(Op::Add, vec![acc, x.clone()]),
                    );

                    Ok(tree)
                },
                Op::OpenStatement => {
                    let mut depth = 1;

                    // take tokens until we reach the end of this array
                    // if we don't collect them here it causes rust to overflow computing the types
                    let array_tokens = self.tokens.by_ref().take_while(|t| match t {
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
                        ParseTree::Nop,
                        |acc, x| ParseTree::Operator(Op::Compose, vec![acc, x.clone()]),
                    );

                    Ok(tree)
                },
                Op::Equ | Op::LazyEqu => {
                    let token = self.tokens.next().ok_or(ParseError::UnexpectedEndInput)?.map_err(|e| ParseError::TokenizeError(e))?;

                    let body = Box::new(self.parse()?);

                    if let Token::Identifier(ident) = token {
                        match op {
                            Op::Equ => Ok(ParseTree::Equ(
                                ident.clone(),
                                body,
                                Box::new(Parser::new(self.tokens.by_ref(), self.globals.borrow_mut())
                                    .locals(self.locals.clone())
                                    .add_local(ident, Type::Any)
                                    .parse()?))
                            ),
                            Op::LazyEqu => Ok(ParseTree::LazyEqu(
                                ident.clone(),
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
                },
                Op::FunctionDefine(arg_count) => {
                    let f = self.parse_function_definition(arg_count)?;

                    Ok(ParseTree::FunctionDefinition(
                        f.clone(),
                        Box::new(
                            Parser::new(self.tokens, self.globals.borrow_mut())
                                .locals(self.locals.clone())
                                .add_local(f.name().unwrap().to_string(), Type::Function(f.get_type()))
                                .parse()?
                        )))
                },
                Op::LambdaDefine(arg_count) => {
                    let f = self.parse_lambda_definition(arg_count)?;

                    Ok(ParseTree::LambdaDefinition(f))
                },
                Op::Export => {
                    let list = self.parse()?;

                    let mut g = HashMap::new();
                    let list = Executor::new(&mut vec![Ok(list)].into_iter(), &mut g)
                        .next().unwrap().map_err(|_| ParseError::RuntimeError)?;

                    if let Value::Array(Type::String, items) = list {
                        let names = items.into_iter().map(|x| match x {
                            Value::String(s) => s,
                            _ => unreachable!(),
                        });

                        for name in names.clone() {
                            let t = match self.locals.remove(&name).ok_or(ParseError::IdentifierUndefined(name.clone())) {
                                Ok(t) => t,
                                Err(e) => return Err(e),
                            };
                            self.globals.insert(name, t);
                        }

                        Ok(ParseTree::Export(names.collect()))
                    } else {
                        Err(ParseError::NoInput)
                    }
                }
                Op::Empty => Ok(ParseTree::Value(Value::Array(Type::Any, vec![]))),
                Op::NonCall => {
                    let name = Self::get_identifier(self.tokens.next())?;
                    Ok(ParseTree::NonCall(name))
                },
                Op::If => {
                    let cond = self.parse()?;
                    let truebranch = self.parse()?;
    
                    Ok(ParseTree::If(Box::new(cond), Box::new(truebranch)))
                },
                Op::IfElse => {
                    let cond = self.parse()?;
                    let truebranch = self.parse()?;
                    let falsebranch = self.parse()?;
    
                    Ok(ParseTree::IfElse(
                        Box::new(cond), Box::new(truebranch), Box::new(falsebranch)))
                },
                op => self.parse_operator(op),
            },
            t => Err(ParseError::UnwantedToken(t)),
        }
    }

    fn parse_lambda_definition(&mut self, arg_count: usize) -> Result<Function, ParseError> {
        let (t, args) = Self::parse_function_declaration(self.tokens, arg_count)?;

        let mut locals = self.locals.clone();

        for (name, t) in std::iter::zip(args.iter(), t.1.iter()) {
            locals.insert(name.clone(), t.clone());
        }

        Ok(Function::lambda(t, args, Box::new(
            Parser::new(self.tokens, &mut self.globals)
                .locals(locals).parse()?)))
    }

    fn parse_function_definition(&mut self, arg_count: usize) -> Result<Function, ParseError> {
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

    fn parse_function_declaration(
        tokens: &mut Peekable<I>,
        arg_count: usize) -> Result<(FunctionType, Vec<String>), ParseError>
    {
        let args: Vec<(Type, String)> = (0..arg_count)
            .map(|_| Self::parse_function_declaration_parameter(tokens))
            .collect::<Result<_, _>>()?;

        let (types, names): (Vec<_>, Vec<_>) = args.into_iter().unzip();
        let ret = if tokens.next_if(|x| matches!(x, Ok(Token::Operator(Op::Arrow)))).is_some() {
            Self::parse_type(tokens)?
        } else {
            Type::Any
        };

        Ok((FunctionType(Box::new(ret), types), names))
    }

    fn parse_function_declaration_parameter(
        mut tokens: &mut Peekable<I>) -> Result<(Type, String), ParseError>
    {
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
                // inside the next_if closure, we already can know that its an error
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

    // for some dumbass reason,
    // this is the only code that breaks if it doesn't take an impl Iterator instead of simply I ...
    fn parse_type(tokens: &mut Peekable<impl Iterator<Item = Result<Token, TokenizeError>>>) -> Result<Type, ParseError> {
        match tokens.next() {
            Some(Ok(Token::Type(t))) => Ok(t),
            Some(Ok(Token::Operator(Op::OpenArray))) => {
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

                // ... thanks to this conversion here. The compiler complains that the types don't
                // match. there is code elsewhere in this codebase that looks exactly like this and
                // still simply uses &mut Peekable<I> as the type. I don't understand why this code
                // is special, but we have to do horribleness for it to work.
                let mut array_tokens = array_tokens
                    .into_iter()
                    .map(|t| Ok(t))
                    .collect::<Vec<Result<Token, TokenizeError>>>()
                    .into_iter()
                    .peekable();

                let t = match Self::parse_type(&mut array_tokens) {
                    Ok(t) => t,
                    Err(ParseError::UnexpectedEndInput) => Type::Any,
                    Err(e) => return Err(e),
                };

                Ok(Type::Array(Box::new(t)))
            },
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
