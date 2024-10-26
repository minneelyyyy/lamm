mod tokenizer;
mod parser;
mod executor;
mod function;

use executor::{Executor, RuntimeError};
use parser::{ParseTree, Parser};
use tokenizer::Tokenizer;
use function::{FunctionType, Function};

use std::collections::HashMap;
use std::fmt::Display;
use std::io::BufRead;
use std::fmt;
use std::iter::Peekable;
use std::sync::{Arc, Mutex};

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
#[derive(Clone, Debug)]
pub enum Value {
    Float(f64),
    Int(i64),
    Bool(bool),
    String(String),
    Array(Type, Vec<Value>),
    Function(Function),
    Nil,
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Float(l0), Self::Float(r0)) => l0 == r0,
            (Self::Int(l0), Self::Int(r0)) => l0 == r0,
            (Self::Bool(l0), Self::Bool(r0)) => l0 == r0,
            (Self::String(l0), Self::String(r0)) => l0 == r0,
            (Self::Array(l0, l1), Self::Array(r0, r1)) => l0 == r0 && l1 == r1,
            (Self::Function(_), Self::Function(_)) => false,
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
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
            Self::Function(f) => Type::Function(f.get_type()),
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
            Self::Function(func) => write!(f, "{func}"),
            Self::Nil => write!(f, "nil"),
        }
    }
}

#[derive(Clone, Debug)]
enum Cache {
    Cached(Value),
    Uncached(ParseTree),
}

#[derive(Clone, Debug)]
struct Object {
    locals: HashMap<String, Arc<Mutex<Object>>>,
    globals: HashMap<String, Arc<Mutex<Object>>>,
    value: Cache,
}

impl Object {
    pub fn variable(tree: ParseTree, globals: HashMap<String, Arc<Mutex<Object>>>, locals: HashMap<String, Arc<Mutex<Object>>>) -> Self {
        Self {
            locals,
            globals,
            value: Cache::Uncached(tree),
        }
    }

    pub fn value(v: Value, globals: HashMap<String, Arc<Mutex<Object>>>, locals: HashMap<String, Arc<Mutex<Object>>>) -> Self {
        Self {
            locals,
            globals,
            value: Cache::Cached(v),
        }
    }

    pub fn function(func: Function, globals: HashMap<String, Arc<Mutex<Object>>>, locals: HashMap<String, Arc<Mutex<Object>>>) -> Self {
        Self {
            locals,
            globals,
            value: Cache::Cached(Value::Function(func)),
        }
    }

    /// evaluate the tree inside an object if it isn't evaluated yet, returns the value
    pub fn eval(&mut self) -> Result<Value, RuntimeError> {
        match self.value.clone() {
            Cache::Cached(v) => Ok(v),
            Cache::Uncached(tree) => {
                let mut t = vec![Ok(tree.clone())].into_iter();

                let mut exec = Executor::new(&mut t, &mut self.globals)
                    .locals(self.locals.clone());

                let v = exec.exec(Box::new(tree))?;

                self.value = Cache::Cached(v.clone());

                Ok(v)
            }
        }
    }

    pub fn _locals(&self) -> HashMap<String, Arc<Mutex<Object>>> {
        self.locals.clone()
    }

    pub fn _globals(&self) -> HashMap<String, Arc<Mutex<Object>>> {
        self.globals.clone()
    }
}

pub struct Runtime<'a, R: BufRead> {
    tokenizer: Peekable<Tokenizer<R>>,
    global_types: HashMap<String, Type>,
    globals: HashMap<String, Arc<Mutex<Object>>>,
    parser: Option<Parser<'a, Tokenizer<R>>>,
}

impl<'a, R: BufRead> Runtime<'a, R> {
    pub fn new(reader: R) -> Self {
        Self {
            tokenizer: Tokenizer::new(reader).peekable(),
            global_types: HashMap::new(),
            globals: HashMap::new(),
            parser: None,
        }.add_global("version'", Value::String(
                format!("{} ({}/{})",
                        env!("CARGO_PKG_VERSION"),
                        env!("GIT_BRANCH"),
                        env!("GIT_HASH")
                )
        ))
    }

    pub fn add_global(mut self, name: &str, value: Value) -> Self {
        self.global_types.insert(name.to_string(), value.get_type());
        self.globals.insert(name.to_string(),
                            Arc::new(Mutex::new(Object::value(
                                value,
                                HashMap::new(),
                                HashMap::new()))));

        self
    }

    pub fn values(&'a mut self) -> impl Iterator<Item = Result<Value, RuntimeError>> + 'a {
        self.parser = Some(Parser::new(&mut self.tokenizer, &mut self.global_types));
        Executor::new(self.parser.as_mut().unwrap(), &mut self.globals)
    }
}
