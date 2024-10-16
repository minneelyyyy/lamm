mod tokenizer;
mod parser;
mod executor;

use executor::{Executor, RuntimeError};
use parser::{ParseTree, Parser};
use tokenizer::Tokenizer;

use std::fmt::Display;
use std::io::{Write, Read, BufRead};
use std::fmt;

#[derive(Clone, Debug)]
pub enum Type {
    Float,
    Int,
    Bool,
    String,
    Array(Box<Type>),
    Function(FunctionType),
    Nil,
    Any,
}

impl PartialEq for Type {
    fn eq(&self, other: &Type) -> bool {
        match (self, other) {
            (Self::Any, _) => true,
            (_, Self::Any) => true,
            (Self::Array(l0), Self::Array(r0)) => l0 == r0,
            (Self::Function(l0), Self::Function(r0)) => l0 == r0,
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", match self {
            Self::Float => "Float".into(),
            Self::Int => "Int".into(),
            Self::Bool => "Bool".into(),
            Self::String => "String".into(),
            Self::Array(t) => format!("[{t}]"),
            Self::Function(r) => format!("{r}"),
            Self::Nil => "Nil".into(),
            Self::Any => "Any".into(),
        })
    }
}

/// Represents the result of executing a ParseTree with an Executor
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Float(f64),
    Int(i64),
    Bool(bool),
    String(String),
    Array(Type, Vec<Value>),
    Function(Function),
    Nil,
}

impl Value {
    pub(crate) fn get_type(&self) -> Type {
        match self {
            Self::Float(_) => Type::Float,
            Self::Int(_) => Type::Int,
            Self::Bool(_) => Type::Bool,
            Self::String(_) => Type::String,
            Self::Array(t, _) => Type::Array(Box::new(t.clone())),
            Self::Nil => Type::Nil,
            Self::Function(f) => Type::Function(f.t.clone()),
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Float(x) => write!(f, "{x}"),
            Self::Int(x) => write!(f, "{x}"),
            Self::Bool(x) => write!(f, "{}", if *x { "true" } else { "false" }),
            Self::String(x) => write!(f, "\"{x}\""),
            Self::Array(_t, v) => write!(f, "[{}]", v.iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(" ")),
            Self::Function(func) => {
                if let Some(name) = &func.name {
                    write!(f, "Function({}, {}, {})", name, func.t.0, func.t.1.iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(", "))
                } else {
                    write!(f, "{}", func.t)
                }
            }
            Self::Nil => write!(f, "nil"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FunctionType(Box<Type>, Vec<Type>);

impl Display for FunctionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Function({}, {})", self.0, self.1.iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(", "))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Function {
    name: Option<String>,
    t: FunctionType,
    arg_names: Option<Vec<String>>,
    body: Option<Box<ParseTree>>,
}

impl Function {
    pub fn lambda(t: FunctionType, arg_names: Vec<String>, body: Option<Box<ParseTree>>) -> Self {
        Self {
            name: None,
            t,
            arg_names: Some(arg_names),
            body
        }
    }

    pub fn named(name: &str, t: FunctionType, arg_names: Option<Vec<String>>, body: Option<Box<ParseTree>>) -> Self {
        Self {
            name: Some(name.to_string()),
            t,
            arg_names,
            body
        }
    }
}

pub struct Runtime<'a, R: BufRead> {
    inner: executor::Executor<'a, parser::Parser<tokenizer::Tokenizer<R>>>
}

impl<'a, R: BufRead + 'a> Runtime<'a, R> {
    pub fn new(reader: R) -> Self {
        Self {
            inner: Executor::new(Parser::new(Tokenizer::new(reader)))
        }
    }

    pub fn stdout(self, stdout: impl Write + 'a) -> Self {
        Self {
            inner: self.inner.stdout(stdout)
        }
    }

    pub fn stdin(self, stdin: impl Read + 'a) -> Self {
        Self {
            inner: self.inner.stdin(stdin)
        }
    }

    pub fn values(self) -> impl Iterator<Item = Result<Value, RuntimeError>> + 'a {
        self.inner
    }
}
