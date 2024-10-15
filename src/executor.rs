use super::{Value, Type, FunctionDeclaration};
use super::parser::{ParseTree, ParseError};

use std::collections::HashMap;
use std::borrow::Cow;
use std::fmt::Display;
use std::error::Error;

#[derive(Debug)]
pub enum RuntimeError {
    ParseError(ParseError),
    NoOverloadForTypes(String, Vec<Value>),
    ImmutableError(String),
    VariableUndefined(String),
    FunctionUndeclared(String),
    FunctionUndefined(String),
    NotAVariable(String),
    ParseFail(String, Type),
}

impl Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ParseError(e) => write!(f, "{e}"),
            Self::NoOverloadForTypes(op, values)
                => write!(f, "No overload of `{op}` exists for the operands `[{}]`", 
                    values.iter().map(|x| format!("{}({x})", x.get_type())).collect::<Vec<_>>().join(", ")),
            Self::ImmutableError(ident) => write!(f, "`{ident}` already exists and cannot be redefined"),
            Self::VariableUndefined(ident) => write!(f, "variable `{ident}` was not defined"),
            Self::FunctionUndeclared(ident) => write!(f, "function `{ident}` was not declared"),
            Self::FunctionUndefined(ident) => write!(f, "function `{ident}` was not defined"),
            Self::NotAVariable(ident) => write!(f, "`{ident}` is a function but was attempted to be used like a variable"),
            Self::ParseFail(s, t) => write!(f, "`\"{s}\"` couldn't be parsed into {}", t),
        }
    }
}

impl Error for RuntimeError {}

#[derive(Clone, Debug)]
enum Evaluation {
    // at this point, it's type is set in stone
    Computed(Value),

    // at this point, it's type is unknown, and may contradict a variable's type
    // or not match the expected value of the expression, this is a runtime error
    Uncomputed(Box<ParseTree>),
}

#[derive(Clone, Debug)]
struct Function {
    decl: FunctionDeclaration,
    body: Option<Box<ParseTree>>,
}

#[derive(Clone, Debug)]
enum Object {
    Variable(Evaluation),
    Function(Function),
}

/// Executes an input of ParseTrees
pub struct Executor<I: Iterator<Item = Result<ParseTree, ParseError>>> {
    exprs: I,
    globals: HashMap<String, Object>,
}

impl<I: Iterator<Item = Result<ParseTree, ParseError>>> Executor<I> {
    pub fn new(exprs: I) -> Self {
        Self {
            exprs,
            globals: HashMap::new(),
        }
    }

    fn exec(
        &mut self,
        tree: Box<ParseTree>,
        locals: &mut Cow<Box<HashMap<String, Object>>>) -> Result<Value, RuntimeError>
    {
        match *tree {
            ParseTree::Add(x, y) => match (self.exec(x, locals)?, self.exec(y, locals)?) {
                (Value::Int(x), Value::Int(y)) => Ok(Value::Int(x + y)),
                (Value::Float(x), Value::Int(y)) => Ok(Value::Float(x + y as f64)),
                (Value::Int(x), Value::Float(y)) => Ok(Value::Float(x as f64 + y)),
                (Value::Float(x), Value::Float(y)) => Ok(Value::Float(x + y)),
                (Value::String(x), Value::String(y)) => Ok(Value::String(format!("{x}{y}"))),
                (x, y) => Err(RuntimeError::NoOverloadForTypes("+".into(), vec![x, y]))
            },
            ParseTree::Sub(x, y) => match (self.exec(x, locals)?, self.exec(y, locals)?) {
                (Value::Int(x), Value::Int(y)) => Ok(Value::Int(x - y)),
                (Value::Float(x), Value::Int(y)) => Ok(Value::Float(x - y as f64)),
                (Value::Int(x), Value::Float(y)) => Ok(Value::Float(x as f64 - y)),
                (Value::Float(x), Value::Float(y)) => Ok(Value::Float(x - y)),
                (x, y) => Err(RuntimeError::NoOverloadForTypes("-".into(), vec![x, y]))
            },
            ParseTree::Mul(x, y) => match (self.exec(x, locals)?, self.exec(y, locals)?) {
                (Value::Int(x), Value::Int(y)) => Ok(Value::Int(x * y)),
                (Value::Float(x), Value::Int(y)) => Ok(Value::Float(x * y as f64)),
                (Value::Int(x), Value::Float(y)) => Ok(Value::Float(x as f64 * y)),
                (Value::Float(x), Value::Float(y)) => Ok(Value::Float(x * y)),
                (Value::String(x), Value::Int(y)) => Ok(Value::String(x.repeat(y as usize))),
                (x, y) => Err(RuntimeError::NoOverloadForTypes("*".into(), vec![x, y]))
            },
            ParseTree::Div(x, y) => match (self.exec(x, locals)?, self.exec(y, locals)?) {
                (Value::Int(x), Value::Int(y)) => Ok(Value::Int(x / y)),
                (Value::Float(x), Value::Int(y)) => Ok(Value::Float(x / y as f64)),
                (Value::Int(x), Value::Float(y)) => Ok(Value::Float(x as f64 / y)),
                (Value::Float(x), Value::Float(y)) => Ok(Value::Float(x / y)),
                (x, y) => Err(RuntimeError::NoOverloadForTypes("*".into(), vec![x, y]))
            },
            ParseTree::Exp(x, y) => match (self.exec(x, locals)?, self.exec(y, locals)?) {
                (Value::Int(x), Value::Int(y)) => Ok(Value::Int(x.pow(y as u32))),
                (Value::Int(x), Value::Float(y)) => Ok(Value::Float((x as f64).powf(y))),
                (Value::Float(x), Value::Int(y)) => Ok(Value::Float(x.powf(y as f64))),
                (Value::Float(x), Value::Float(y)) => Ok(Value::Float(x.powf(y))),
                (x, y) => Err(RuntimeError::NoOverloadForTypes("**".into(), vec![x, y])),
            },
            ParseTree::Mod(x, y) => match (self.exec(x, locals)?, self.exec(y, locals)?) {
                (Value::Int(x), Value::Int(y)) => Ok(Value::Int(x % y)),
                (Value::Float(x), Value::Int(y)) => Ok(Value::Float(x % y as f64)),
                (Value::Int(x), Value::Float(y)) => Ok(Value::Float(x as f64 % y)),
                (Value::Float(x), Value::Float(y)) => Ok(Value::Float(x % y)),
                (x, y) => Err(RuntimeError::NoOverloadForTypes("%".into(), vec![x, y])),
            },
            ParseTree::EqualTo(x, y) => match (self.exec(x, locals)?, self.exec(y, locals)?) {
                (Value::Int(x), Value::Int(y)) => Ok(Value::Bool(x == y)),
                (Value::Int(x), Value::Float(y)) => Ok(Value::Bool(x as f64 == y)),
                (Value::Float(x), Value::Int(y)) => Ok(Value::Bool(x == y as f64)),
                (Value::Float(x), Value::Float(y)) => Ok(Value::Bool(x == y)),
                (Value::Bool(x), Value::Bool(y)) => Ok(Value::Bool(x == y)),
                (Value::String(x), Value::String(y)) => Ok(Value::Bool(x == y)),
                (x, y) => Err(RuntimeError::NoOverloadForTypes("==".into(), vec![x, y])),
            },
            ParseTree::GreaterThan(x, y) => match (self.exec(x, locals)?, self.exec(y, locals)?) {
                (Value::Int(x), Value::Int(y)) => Ok(Value::Bool(x > y)),
                (Value::Int(x), Value::Float(y)) => Ok(Value::Bool(x as f64 > y)),
                (Value::Float(x), Value::Int(y)) => Ok(Value::Bool(x > y as f64)),
                (Value::Float(x), Value::Float(y)) => Ok(Value::Bool(x > y)),
                (x, y) => Err(RuntimeError::NoOverloadForTypes(">".into(), vec![x, y])),
            },
            ParseTree::GreaterThanOrEqualTo(x, y) => match (self.exec(x, locals)?, self.exec(y, locals)?) {
                (Value::Int(x), Value::Int(y)) => Ok(Value::Bool(x >= y)),
                (Value::Int(x), Value::Float(y)) => Ok(Value::Bool(x as f64 >= y)),
                (Value::Float(x), Value::Int(y)) => Ok(Value::Bool(x >= y as f64)),
                (Value::Float(x), Value::Float(y)) => Ok(Value::Bool(x >= y)),
                (x, y) => Err(RuntimeError::NoOverloadForTypes(">=".into(), vec![x, y])),
            },
            ParseTree::LessThan(x, y) => match (self.exec(x, locals)?, self.exec(y, locals)?) {
                (Value::Int(x), Value::Int(y)) => Ok(Value::Bool(x < y)),
                (Value::Int(x), Value::Float(y)) => Ok(Value::Bool((x as f64) < y)),
                (Value::Float(x), Value::Int(y)) => Ok(Value::Bool(x < y as f64)),
                (Value::Float(x), Value::Float(y)) => Ok(Value::Bool(x < y)),
                (x, y) => Err(RuntimeError::NoOverloadForTypes("<".into(), vec![x, y])),
            },
            ParseTree::LessThanOrEqualTo(x, y) => match (self.exec(x, locals)?, self.exec(y, locals)?) {
                (Value::Int(x), Value::Int(y)) => Ok(Value::Bool(x <= y)),
                (Value::Int(x), Value::Float(y)) => Ok(Value::Bool(x as f64 <= y)),
                (Value::Float(x), Value::Int(y)) => Ok(Value::Bool(x <= y as f64)),
                (Value::Float(x), Value::Float(y)) => Ok(Value::Bool(x <= y)),
                (x, y) => Err(RuntimeError::NoOverloadForTypes("<=".into(), vec![x, y])),
            },
            ParseTree::Not(x) => match self.exec(x, locals)? {
                Value::Bool(x) => Ok(Value::Bool(!x)),
                x => Err(RuntimeError::NoOverloadForTypes("not".into(), vec![x]))
            },
            ParseTree::Equ(ident, body, scope) => {
                if self.globals.contains_key(&ident) || locals.contains_key(&ident) {
                    Err(RuntimeError::ImmutableError(ident.clone()))
                } else {
                    let locals = locals.to_mut();
                    let value = self.exec(body, &mut Cow::Borrowed(&locals))?;
                    locals.insert(ident.clone(), Object::Variable(Evaluation::Computed(value)));

                    self.exec(scope, &mut Cow::Borrowed(&locals))
                }
            },
            ParseTree::LazyEqu(ident, body, scope) => {
                if self.globals.contains_key(&ident) || locals.contains_key(&ident) {
                    Err(RuntimeError::ImmutableError(ident.clone()))
                } else {
                    let locals = locals.to_mut();
                    locals.insert(ident.clone(), Object::Variable(Evaluation::Uncomputed(body)));

                    self.exec(scope, &mut Cow::Borrowed(&locals))
                }
            },
            ParseTree::FunctionDefinition(ident, args, r, body, scope) => {
                let existing = locals.get(&ident).or(self.globals.get(&ident)).cloned();

                match existing {
                    Some(_) => Err(RuntimeError::ImmutableError(ident.clone())),
                    None => {
                        let locals = locals.to_mut();

                        locals.insert(ident.clone(), Object::Function(Function {
                            decl: FunctionDeclaration { _name: ident.clone(), _r: r, args },
                            body: Some(body)
                        }));

                        self.exec(scope, &mut Cow::Borrowed(&locals))
                    }
                }
            },
            ParseTree::Compose(x, y) => {
                self.exec(x, locals)?;
                self.exec(y, locals)
            },
            ParseTree::Id(x) => self.exec(x, locals),
            ParseTree::If(cond, body) => if match self.exec(cond, locals)? {
                    Value::Float(f) => f != 0.0,
                    Value::Int(i) => i != 0,
                    Value::Bool(b) => b,
                    Value::String(s) => !s.is_empty(),
                    Value::Nil => false,
                } {
                    self.exec(body, locals)
                } else {
                    Ok(Value::Nil)
                },
            ParseTree::IfElse(cond, istrue, isfalse) => if match self.exec(cond, locals)? {
                Value::Float(f) => f != 0.0,
                Value::Int(i) => i != 0,
                Value::Bool(b) => b,
                Value::String(s) => !s.is_empty(),
                Value::Nil => false,
            } {
                self.exec(istrue, locals)
            } else {
                self.exec(isfalse, locals)
            },
            ParseTree::FunctionCall(ident, args) => {
                let obj = locals.get(&ident).or(self.globals.get(&ident)).cloned();

                if let Some(Object::Function(f)) = obj {
                    let locals = locals.to_mut();
                    let body = f.body.ok_or(RuntimeError::FunctionUndefined(ident.clone()))?;

                    for ((name, _), tree) in std::iter::zip(f.decl.args, args) {
                        locals.insert(name.clone(), Object::Variable(Evaluation::Computed(self.exec(Box::new(tree), &mut Cow::Borrowed(locals))?)));
                    }

                    self.exec(body, &mut Cow::Borrowed(&locals))
                } else {
                    Err(RuntimeError::FunctionUndeclared(ident.clone()))
                }
            },
            ParseTree::Variable(ident) => {
                let locals = locals.to_mut();

                let obj = locals.get(&ident).or(self.globals.get(&ident)).cloned();

                if let Some(Object::Variable(eval)) = obj {
                    match eval {
                        Evaluation::Computed(v) => Ok(v),
                        Evaluation::Uncomputed(tree) => {
                            let v = self.exec(tree, &mut Cow::Borrowed(&locals))?;
                            locals.insert(ident, Object::Variable(Evaluation::Computed(v.clone())));

                            Ok(v)
                        }
                    }
                } else {
                    Err(RuntimeError::VariableUndefined(ident.clone()))
                }
            },
            ParseTree::Constant(value) => Ok(value),
            ParseTree::ToInt(x) => match self.exec(x, locals)? {
                Value::Int(x) => Ok(Value::Int(x)),
                Value::Float(x) => Ok(Value::Int(x as i64)),
                Value::Bool(x) => Ok(Value::Int(if x { 1 } else { 0 })),
                Value::String(x) => {
                    let r: i64 = x.parse().map_err(|_| RuntimeError::ParseFail(x.clone(), Type::Int))?;
                    Ok(Value::Int(r))
                }
                x => Err(RuntimeError::NoOverloadForTypes("int".into(), vec![x])),
            },
            ParseTree::ToFloat(x) => match self.exec(x, locals)? {
                Value::Int(x) => Ok(Value::Float(x as f64)),
                Value::Float(x) => Ok(Value::Float(x)),
                Value::Bool(x) => Ok(Value::Float(if x { 1.0 } else { 0.0 })),
                Value::String(x) => {
                    let r: f64 = x.parse().map_err(|_| RuntimeError::ParseFail(x.clone(), Type::Int))?;
                    Ok(Value::Float(r))
                }
                x => Err(RuntimeError::NoOverloadForTypes("float".into(), vec![x])),
            },
            ParseTree::ToBool(x) => match self.exec(x, locals)? {
                Value::Int(x) => Ok(Value::Bool(x != 0)),
                Value::Float(x) => Ok(Value::Bool(x != 0.0)),
                Value::Bool(x) => Ok(Value::Bool(x)),
                Value::String(x) => Ok(Value::Bool(!x.is_empty())),
                x => Err(RuntimeError::NoOverloadForTypes("bool".into(), vec![x])),
            },
            ParseTree::ToString(x) => Ok(Value::String(format!("{}", self.exec(x, locals)?))),
        }
    }
}

impl<I: Iterator<Item = Result<ParseTree, ParseError>>> Iterator for Executor<I> {
    type Item = Result<Value, RuntimeError>;

    fn next(&mut self) -> Option<Self::Item> {
        let expr = self.exprs.next();

        match expr {
            Some(Ok(expr)) => Some(self.exec(Box::new(expr), &mut Cow::Borrowed(&Box::new(HashMap::new())))),
            Some(Err(e)) => Some(Err(RuntimeError::ParseError(e))),
            None => None,
        }
    }
}