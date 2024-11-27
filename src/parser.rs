use super::{Value, Type, Function, FunctionType};
use super::tokenizer::{Token, TokenType, Op};
use super::error::Error;

use std::collections::HashMap;
use std::iter::Peekable;
use std::cmp::Ordering;

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
    Export(Vec<String>),
}

/// Parses input tokens and produces ParseTrees for an Executor
#[derive(Clone)]
pub(crate) struct Parser {
    globals: HashMap<String, Type>,
    locals: HashMap<String, Type>,
}

impl Parser {
    pub(crate) fn new() -> Self {
        Self {
            globals: HashMap::new(),
            locals: HashMap::new()
        }
    }

    pub(crate) fn trees<I: Iterator<Item = Result<Token, Error>>>(mut self, mut tokens: Peekable<I>) -> impl Iterator<Item = Result<ParseTree, Error>> {
        std::iter::from_fn(move || {
            match self.parse(&mut tokens) {
                Ok(Some(tree)) => Some(Ok(tree)),
                Ok(None) => None,
                Err(e) => Some(Err(e)),
            }
        })
    }

    pub(crate) fn add_global(mut self, k: String, v: Type) -> Self {
        self.globals.insert(k, v);
        self
    }

    pub(crate) fn add_globals<Items: IntoIterator<Item = (String, Type)>>(self, items: Items) -> Self {
        items.into_iter().fold(self, |acc, (k, v)| acc.add_global(k, v))
    }

    pub(crate) fn _add_local(mut self, k: String, v: Type) -> Self {
        self.locals.insert(k, v);
        self
    }

    pub(crate) fn _add_locals<Items: Iterator<Item = (String, Type)>>(self, items: Items) -> Self {
        items.fold(self, |acc, (key, value)| acc._add_local(key, value))
    }

    fn add_local_mut(&mut self, k: String, v: Type) -> &mut Self {
        self.locals.insert(k, v);
        self
    }

    fn add_locals_mut<Items: IntoIterator<Item = (String, Type)>>(&mut self, items: Items) -> &mut Self {
        for (name, t) in items {
            self.locals.insert(name, t);
        }

        self
    }

    fn get_object_type(&self, ident: &String) -> Option<&Type> {
        self.locals.get(ident).or(self.globals.get(ident))
    }

    fn _get_object_types<Names: Iterator<Item = String>>(&self, items: Names) -> impl Iterator<Item = Option<&Type>> {
        items.map(|x| self.get_object_type(&x))
    }

    // get at most count arguments
    fn get_args<I: Iterator<Item = Result<Token, Error>>>(&mut self, tokens: &mut Peekable<I>, count: usize) -> Result<Vec<ParseTree>, Error> {
        (0..count).map_while(|_| match self.parse(tokens) {
            Ok(Some(tree)) => Some(Ok(tree)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }).collect()
    }

    fn parse_operator<I: Iterator<Item = Result<Token, Error>>>(&mut self, tokens: &mut Peekable<I>, op: Op) -> Result<ParseTree, Error> {
        let operators: HashMap<Op, FunctionType> = HashMap::from([
            (Op::Add, FunctionType(Box::new(Type::Any), vec![Type::Any, Type::Any])),
            (Op::Sub, FunctionType(Box::new(Type::Any), vec![Type::Any, Type::Any])),
            (Op::Neg, FunctionType(Box::new(Type::Any), vec![Type::Any])),
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
            (Op::Concat, FunctionType(Box::new(Type::Array(Box::new(Type::Any))), vec![Type::Array(Box::new(Type::Any)), Type::Array(Box::new(Type::Any))])),
            (Op::Prepend, FunctionType(Box::new(Type::Array(Box::new(Type::Any))), vec![Type::Any, Type::Array(Box::new(Type::Any))])),
            (Op::Append, FunctionType(Box::new(Type::Array(Box::new(Type::Any))), vec![Type::Array(Box::new(Type::Any)), Type::Any])),
            (Op::Insert, FunctionType(Box::new(Type::Array(Box::new(Type::Any))), vec![Type::Int, Type::Any, Type::Array(Box::new(Type::Any))])),
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
        let args = self.get_args(tokens, operator.1.len())?;

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

    pub(crate) fn parse<I: Iterator<Item = Result<Token, Error>>>(&mut self, tokens: &mut Peekable<I>) -> Result<Option<ParseTree>, Error> {
        let token = match tokens.next() {
            Some(Ok(t)) => t,
            Some(Err(e)) => return Err(e),
            None => return Ok(None),
        };

        match token.token() {
            TokenType::Constant(c) => Ok(Some(ParseTree::Value(c))),
            TokenType::Identifier(ident) => Ok(Some(ParseTree::Variable(ident))),
            TokenType::Operator(op) => match op {
                Op::OpenArray => {
                    let mut depth = 1;

                    // take tokens until we reach the end of this array
                    // if we don't collect them here it causes rust to overflow computing the types
                    let array_tokens = tokens.by_ref().take_while(|t| match t {
                        Ok(t) => match t.token() {
                            TokenType::Operator(Op::OpenArray) => {
                                depth += 1;
                                true
                            },
                            TokenType::Operator(Op::CloseArray) => {
                                depth -= 1;
                                depth > 0
                            }
                            _ => true,
                        }
                        _ => true,
                    }).collect::<Result<Vec<_>, Error>>()?;

                    let array_tokens = array_tokens
                        .into_iter()
                        .map(|t| Ok(t))
                        .collect::<Vec<Result<Token, Error>>>()
                        .into_iter()
                        .peekable();

                    let trees: Vec<ParseTree> = self.clone().trees(array_tokens)
                        .collect::<Result<_, Error>>()?;

                    let tree = trees.into_iter().fold(
                        ParseTree::Value(Value::Array(Type::Any, vec![])),
                        |acc, x| ParseTree::Operator(Op::Append, vec![acc, x.clone()]),
                    );

                    Ok(Some(tree))
                },
                Op::OpenStatement => {
                    let mut depth = 1;

                    // take tokens until we reach the end of this array
                    // if we don't collect them here it causes rust to overflow computing the types
                    let tokens = tokens.by_ref().take_while(|t| match t {
                        Ok(t) => match t.token() {
                            TokenType::Operator(Op::OpenStatement) => {
                                depth += 1;
                                true
                            },
                            TokenType::Operator(Op::CloseStatement) => {
                                depth -= 1;
                                depth > 0
                            }
                            _ => true,
                        }
                        _ => true,
                    }).collect::<Result<Vec<_>, Error>>()?;

                    let mut tokens = tokens
                        .into_iter()
                        .map(|t| Ok(t))
                        .collect::<Vec<Result<Token, Error>>>()
                        .into_iter()
                        .peekable();

                    if let Some(Ok(Some(Type::Function(f)))) = tokens.peek()
                        .map(|t| t.clone().and_then(|t| match t.token() {
                            TokenType::Identifier(ident) =>
                                Ok(Some(self.get_object_type(&ident).ok_or(
                                    Error::new(format!("undefined identifier {ident}"))
                                        .location(token.line, token.location))?)),
                            _ => Ok(None),
                        }))
                    {
                        let token = tokens.next().unwrap().unwrap();
                        let params: Vec<ParseTree> = self.clone().trees(tokens).collect::<Result<_, Error>>()?;

                        match params.len().cmp(&f.1.len()) {
                            Ordering::Equal => Ok(Some(ParseTree::FunctionCall(token.lexeme, params))),
                            Ordering::Greater => Err(Error::new(format!("too many arguments to {}", token.lexeme)).location(token.line, token.location)),
                            Ordering::Less => {
                                let mut counter = 0;
                                let func_args: Vec<Type> = f.1.iter().skip(params.len()).cloned().collect();
                                let (names, types): (Vec<String>, Vec<Type>) = func_args
                                    .into_iter()
                                    .map(|t| {
                                        counter += 1;
                                        (format!("{counter}"), t)
                                    }).unzip();
                                let function_type = FunctionType(f.0.clone(), types);
    
                                Ok(Some(ParseTree::Value(Value::Function(Function::lambda(
                                    function_type,
                                    names.clone(),
                                    Box::new(ParseTree::FunctionCall(token.lexeme,
                                        vec![
                                            params,
                                            names.into_iter().map(|x| ParseTree::Variable(x)).collect()
                                        ].concat())))))))
                            }
                        }
                    } else {
                        let trees: Vec<ParseTree> = self.clone().trees(tokens).collect::<Result<_, Error>>()?;

                        let tree = trees.into_iter().fold(
                            ParseTree::Nop,
                            |acc, x| ParseTree::Operator(Op::Compose, vec![acc, x.clone()]),
                        );

                        Ok(Some(tree))
                    }
                },
                Op::Equ => {
                    let token = tokens.next()
                        .ok_or(Error::new("no identifier given for = expression".into())
                            .location(token.line, token.location)
                            .note("expected an identifier after this token".into()))??;

                    if let TokenType::Identifier(ident) = token.token() {
                        let body = self.parse(tokens)?.ok_or(Error::new(format!("the variable `{ident}` has no value"))
                            .location(token.line, token.location.clone())
                            .note("expected a value after this identifier".into()))?;

                        let scope = self.add_local_mut(ident.clone(), Type::Any)
                            .parse(tokens)?
                            .ok_or(Error::new("variable declaration requires a scope defined after it".into())
                                .location(token.line, token.location)
                                .note(format!("this variable {ident} has no scope")))?;
                        
                        // temporary fix: just remove the identifier
                        // ignore errors removing, in the case that the symbol was already exported, it won't be present in locals
                        // this comes down to a basic architectural error. globals need to stick to the parser while locals need to be scoped.
                        self.locals.remove(&ident);

                        Ok(Some(ParseTree::Equ(
                            ident.clone(),
                            Box::new(body),
                            Box::new(scope))
                        ))
                    } else {
                        Err(Error::new(format!("`{}` is not a valid identifier", token.lexeme)).location(token.line, token.location))
                    }
                },
                Op::LazyEqu => {
                    let token = tokens.next()
                        .ok_or(Error::new("no identifier given for . expression".into())
                            .location(token.line, token.location)
                            .note("expected an identifier after this token".into()))??;

                    if let TokenType::Identifier(ident) = token.token() {
                        let body = Box::new(self.parse(tokens)?.ok_or(Error::new(format!("the variable `{ident}` has no value"))
                            .location(token.line, token.location.clone())
                            .note("expected a value after this identifier".into()))?);

                        let scope = self.add_local_mut(ident.clone(), Type::Any)
                            .parse(tokens)?
                            .ok_or(Error::new("variable declaration requires a scope defined after it".into())
                                .location(token.line, token.location)
                                .note(format!("this variable {ident} has no scope")))?;
                        
                        // temporary fix: just remove the identifier
                        // ignore errors removing, in the case that the symbol was already exported, it won't be present in locals
                        self.locals.remove(&ident);

                        Ok(Some(ParseTree::LazyEqu(
                            ident.clone(),
                            body,
                            Box::new(scope))
                        ))
                    } else {
                        Err(Error::new(format!("`{}` is not a valid identifier", token.lexeme)).location(token.line, token.location))
                    }
                },
                Op::FunctionDefine(arg_count) => {
                    let f = self.parse_function_definition(tokens, arg_count)?;

                    let scope = self.add_local_mut(f.name().unwrap().to_string(), Type::Function(f.get_type()))
                        .parse(tokens)?
                            .ok_or(Error::new("function declaration requires a scope defined after it".into())
                            .location(token.line, token.location)
                            .note(format!("this function {} has no scope", f.name().unwrap())))?;
                    
                    self.locals.remove(f.name().unwrap());

                    Ok(Some(ParseTree::FunctionDefinition( f.clone(), Box::new(scope))))
                },
                Op::LambdaDefine(arg_count) => Ok(Some(ParseTree::LambdaDefinition(self.parse_lambda_definition(tokens, arg_count)?))),
                Op::Empty => Ok(Some(ParseTree::Value(Value::Array(Type::Any, vec![])))),
                Op::If => {
                    let cond = self.parse(tokens)?
                        .ok_or(Error::new("? statement requires a condition".into())
                            .location(token.line, token.location.clone()))?;
                    let truebranch = self.parse(tokens)?
                        .ok_or(Error::new("? statement requires a branch".into())
                            .location(token.line, token.location))?;
    
                    Ok(Some(ParseTree::If(Box::new(cond), Box::new(truebranch))))
                },
                Op::IfElse => {
                    let cond = self.parse(tokens)?
                        .ok_or(Error::new("?? statement requires a condition".into())
                            .location(token.line, token.location.clone()))?;
                    let truebranch = self.parse(tokens)?
                        .ok_or(Error::new("?? statement requires a branch".into())
                            .location(token.line, token.location.clone()))?;
                    let falsebranch = self.parse(tokens)?
                        .ok_or(Error::new("?? statement requires a false branch".into())
                            .location(token.line, token.location))?;
    
                    Ok(Some(ParseTree::IfElse(
                        Box::new(cond), Box::new(truebranch), Box::new(falsebranch))))
                },
                Op::Export => {
                    let token = tokens.next()
                        .ok_or(Error::new("export expects an identifer or multiple inside of parens".into())
                            .location(token.line, token.location.clone()))??;

                    let names = match token.token() {
                        TokenType::Identifier(ident) => vec![ident],
                        TokenType::Operator(Op::OpenStatement) => {
                            tokens
                                .take_while(|token| !matches!(token.clone().map(|token| token.token()), Ok(TokenType::Operator(Op::CloseStatement))))
                                .map(|token| token.map(|token| match token.token() {
                                    TokenType::Identifier(ident) => Ok(ident),
                                    _ => Err(Error::new(format!("expected an identifier")).location(token.line, token.location))
                                })?)
                                .collect::<Result<_, Error>>()?
                        }
                        _ => return Err(Error::new("export expects one or more identifiers".into()).location(token.line, token.location)),
                    };

                    for name in &names {
                        let (name, t) = self.locals.remove_entry(name)
                            .ok_or(
                                Error::new(format!("attempt to export {name}, which is not in local scope"))
                                    .location(token.line, token.location.clone())
                            )?;

                        self.globals.insert(name, t);
                    }

                    Ok(Some(ParseTree::Export(names)))
                },
                op => self.parse_operator(tokens, op).map(|x| Some(x)),
            },
            _ => Err(Error::new(format!("the token {} was unexpected", token.lexeme)).location(token.line, token.location)),
        }
    }

    fn parse_lambda_definition<I: Iterator<Item = Result<Token, Error>>>(&mut self, tokens: &mut Peekable<I>, arg_count: usize) -> Result<Function, Error> {
        let (t, args) = Self::parse_function_declaration(tokens, arg_count)?;

        let mut locals = self.locals.clone();

        for (name, t) in std::iter::zip(args.iter(), t.1.iter()) {
            locals.insert(name.clone(), t.clone());
        }

        Ok(Function::lambda(t, args, Box::new(
            self.clone().add_locals_mut(locals).parse(tokens)?.ok_or(Error::new("lambda requires a body".into()))?)))
    }

    fn parse_function_definition<I: Iterator<Item = Result<Token, Error>>>(&mut self, tokens: &mut Peekable<I>, arg_count: usize) -> Result<Function, Error> {
        let name = Self::get_identifier(tokens.next())?;
        let (t, args) = Self::parse_function_declaration(tokens, arg_count)?;

        let mut locals = self.locals.clone();

        for (name, t) in std::iter::zip(args.iter(), t.1.iter()) {
            locals.insert(name.clone(), t.clone());
        }

        locals.insert(name.clone(), Type::Function(t.clone()));

        Ok(Function::named(&name, t, args, Box::new(
            self.clone().add_locals_mut(locals).parse(tokens)?.ok_or(Error::new("function requires a body".into()))?)))
    }

    fn parse_function_declaration<I: Iterator<Item = Result<Token, Error>>>(
        tokens: &mut Peekable<I>,
        arg_count: usize) -> Result<(FunctionType, Vec<String>), Error>
    {
        let args: Vec<(Type, String)> = (0..arg_count)
            .map(|_| Self::parse_function_declaration_parameter(tokens))
            .collect::<Result<_, _>>()?;

        let (types, names): (Vec<_>, Vec<_>) = args.into_iter().unzip();

        let ret = if tokens.next_if(|x| matches!(x.as_ref().unwrap().token(), TokenType::Operator(Op::Arrow))).is_some() {
            Self::parse_type(tokens)?
        } else {
            Type::Any
        };

        Ok((FunctionType(Box::new(ret), types), names))
    }

    fn parse_function_declaration_parameter<I: Iterator<Item = Result<Token, Error>>>(tokens: &mut Peekable<I>) -> Result<(Type, String), Error> {
        let token = tokens.next().ok_or(Error::new("function definition is incomplete".into()))??;

        match token.token() {
            // untyped variable
            TokenType::Identifier(x) => Ok((Type::Any, x)),

            // typed variable
            TokenType::Operator(Op::TypeDeclaration) => {
                let name = Self::get_identifier(tokens.next())?;
                let t = Self::parse_type(tokens)?;

                Ok((t, name))
            }

            // untyped function (all args Any, return type Any)
            TokenType::Operator(Op::FunctionDefine(n)) => {
                let name = Self::get_identifier(tokens.next())?;
                let args = (0..n).map(|_| Type::Any).collect();

                Ok((Type::Function(FunctionType(Box::new(Type::Any), args)), name))
            }

            // typed function
            TokenType::Operator(Op::FunctionDeclare(n)) => {
                let name = Self::get_identifier(tokens.next())?;
                let args = (0..n).map(|_| Self::parse_type(tokens)).collect::<Result<_, _>>()?;
                let mut ret = Type::Any;

                // this is annoying
                // inside the next_if closure, we already can know that its an error
                // and return it, but we cannot return out of a closure
                if let Some(t) = tokens.next_if(|x| matches!(x.as_ref().unwrap().token(), TokenType::Operator(Op::Arrow)))
                {
                    // so we just check for an error here. this is the only reason t exists.
                    if let Err(e) = t {
                        return Err(e);
                    }

                    ret = Self::parse_type(tokens)?;
                }

                Ok((Type::Function(FunctionType(Box::new(ret), args)), name))
            }
            _ => Err(Error::new(format!("unexpected token {}", token.lexeme))),
        }
    }

    fn parse_type<I: Iterator<Item = Result<Token, Error>>>(tokens: &mut Peekable<I>) -> Result<Type, Error> {
        let token = tokens.next().ok_or(Error::new("type is incomplete".into()))??;

        match token.token() {
            TokenType::Type(t) => Ok(t),
            TokenType::Operator(Op::OpenArray) => {
                let mut depth = 1;

                // take tokens until we reach the end of this array
                // if we don't collect them here it causes rust to overflow computing the types
                let array_tokens = tokens.by_ref().take_while(|t| match t {
                    Ok(t) => match t.token() {
                        TokenType::Operator(Op::OpenArray) => {
                            depth += 1;
                            true
                        },
                        TokenType::Operator(Op::CloseArray) => {
                            depth -= 1;
                            depth > 0
                        }
                        _ => true,
                    }
                    _ => true,
                }).collect::<Result<Vec<_>, Error>>()?;

                let mut array_tokens = array_tokens
                    .into_iter()
                    .map(|t| Ok(t))
                    .collect::<Vec<_>>()
                    .into_iter();

                let t = if array_tokens.len() == 0 {
                    Type::Any
                } else {
                    Parser::parse_type(&mut array_tokens.by_ref().peekable())?
                };

                Ok(Type::Array(Box::new(t)))
            },
            _ => Err(Error::new(format!("unexpected token {}", token.lexeme))),
        }
    }

    fn get_identifier(t: Option<Result<Token, Error>>) -> Result<String, Error> {
        let token = t.ok_or(Error::new(format!("expected an identifier, found nothing")))??;

        match token.token() {
            TokenType::Identifier(ident) => Ok(ident),
            _ => Err(Error::new(format!("the identifier {} is invalid", token.lexeme))),
        }
    }
}
