mod tokenizer;
mod parser;
mod executor;

use executor::{Executor, RuntimeError};
use parser::Parser;
use tokenizer::Tokenizer;

use std::fmt::Display;
use std::io::{Write, Read, BufRead};

#[derive(Clone, Debug, PartialEq)]
pub enum Type {
    Float,
    Int,
    Bool,
    String,
    Array,
    _Function(Box<Type>, Vec<Type>),
    Nil,
    Any,
}

impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", match self {
            Self::Float => "Float".into(),
            Self::Int => "Int".into(),
            Self::Bool => "Bool".into(),
            Self::String => "String".into(),
            Self::Array => format!("Array"),
            Self::_Function(r, _) => format!("Function -> {}", *r),
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
    Array(Vec<Value>),
    Nil,
}

impl Value {
    pub(crate) fn get_type(&self) -> Type {
        match self {
            Self::Float(_) => Type::Float,
            Self::Int(_) => Type::Int,
            Self::Bool(_) => Type::Bool,
            Self::String(_) => Type::String,
            Self::Array(_) => Type::Array,
            Self::Nil => Type::Nil,
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
            Self::Array(v) => write!(f, "[{}]", v.iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(" ")),
            Self::Nil => write!(f, "nil"),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct FunctionDeclaration {
    _name: String,
    args: Vec<String>,
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
