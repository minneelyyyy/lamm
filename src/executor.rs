use super::{Value, Type, Object};
use super::parser::{ParseTree, ParseError};
use super::tokenizer::Op;

use std::collections::HashMap;
use std::fmt::Display;
use std::error::Error;
use std::io;
use std::sync::{Arc, Mutex};

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
    TypeError(Type, Type),
    EmptyArray,
    IO(io::Error),
}

impl Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ParseError(e) => write!(f, "Parser Error: {e}"),
            Self::NoOverloadForTypes(op, values)
                => write!(f, "No overload of `{op}` exists for the operands `{}`", 
                    values.iter().map(|x| format!("{}({x})", x.get_type())).collect::<Vec<_>>().join(", ")),
            Self::ImmutableError(ident) => write!(f, "`{ident}` already exists and cannot be redefined"),
            Self::VariableUndefined(ident) => write!(f, "variable `{ident}` was not defined"),
            Self::FunctionUndeclared(ident) => write!(f, "function `{ident}` was not declared"),
            Self::FunctionUndefined(ident) => write!(f, "function `{ident}` was not defined"),
            Self::NotAVariable(ident) => write!(f, "`{ident}` is a function but was attempted to be used like a variable"),
            Self::ParseFail(s, t) => write!(f, "`\"{s}\"` couldn't be parsed into {}", t),
            Self::IO(e) => write!(f, "{e}"),
            Self::TypeError(left, right) => write!(f, "expected type `{left}` but got type `{right}`"),
            Self::EmptyArray => write!(f, "attempt to access element from an empty array"),
        }
    }
}

impl Error for RuntimeError {}

/// Executes an input of ParseTrees
pub struct Executor<'a, I>
where
    I: Iterator<Item = Result<ParseTree, ParseError>>
{
    exprs: &'a mut I,
    globals: &'a mut HashMap<String, Arc<Mutex<Object>>>,
    locals: HashMap<String, Arc<Mutex<Object>>>,
}

impl<'a, I> Executor<'a, I>
where
    I: Iterator<Item = Result<ParseTree, ParseError>>,
{
    pub fn new(exprs: &'a mut I, globals: &'a mut HashMap<String, Arc<Mutex<Object>>>) -> Self {
        Self {
            exprs,
            globals,
            locals: HashMap::new(),
        }
    }

    pub fn _add_global(self, k: String, v: Arc<Mutex<Object>>) -> Self {
        self.globals.insert(k, v);
        self
    }

    pub fn locals(mut self, locals: HashMap<String, Arc<Mutex<Object>>>) -> Self {
        self.locals = locals;
        self
    }

    pub fn add_local(mut self, k: String, v: Arc<Mutex<Object>>) -> Self {
        self.locals.insert(k, v);
        self
    }

    fn _get_object(&self, ident: &String) -> Result<&Arc<Mutex<Object>>, RuntimeError> {
        self.locals.get(ident).or(self.globals.get(ident))
            .ok_or(RuntimeError::VariableUndefined(ident.clone()))
    }

    fn get_object_mut(&mut self, ident: &String) -> Result<&mut Arc<Mutex<Object>>, RuntimeError> {
        self.locals.get_mut(ident).or(self.globals.get_mut(ident))
            .ok_or(RuntimeError::VariableUndefined(ident.clone()))
    }

    fn variable_exists(&self, ident: &String) -> bool {
        self.locals.contains_key(ident) || self.globals.contains_key(ident)
    }

    fn eval(obj: &mut Arc<Mutex<Object>>) -> Result<Value, RuntimeError> {
        let mut guard = obj.lock().unwrap();

        let v = guard.eval()?;

        Ok(v)
    }

    fn _obj_locals(obj: &Arc<Mutex<Object>>) -> HashMap<String, Arc<Mutex<Object>>> {
        let guard = obj.lock().unwrap();

        let locals = guard._locals();

        locals
    }

    fn _obj_globals(obj: &Arc<Mutex<Object>>) -> HashMap<String, Arc<Mutex<Object>>> {
        let guard = obj.lock().unwrap();

        let locals = guard._globals();

        locals
    }

    pub fn exec(&mut self, tree: Box<ParseTree>) -> Result<Value, RuntimeError> {
        match *tree {
            ParseTree::Operator(op, args) => {
                let args: Vec<Value> = args.into_iter()
                    .map(|x| self.exec(Box::new(x))).collect::<Result<_, _>>()?;

                match op {
                    Op::Add => match &args[..] {
                        [Value::Int(x), Value::Int(y)] => Ok(Value::Int(x + y)),
                        [Value::Float(x), Value::Int(y)] => Ok(Value::Float(x + *y as f64)),
                        [Value::Int(x), Value::Float(y)] => Ok(Value::Float(*x as f64 + y)),
                        [Value::Float(x), Value::Float(y)] => Ok(Value::Float(x + y)),
                        [Value::String(x), Value::String(y)] => Ok(Value::String(format!("{x}{y}"))),
                        [Value::Array(xtype, x), Value::Array(ytype, y)] => {
                            if xtype != ytype {
                                return Err(RuntimeError::TypeError(xtype.clone(), ytype.clone()));
                            }

                            Ok(Value::Array(xtype.clone(), [x.clone(), y.clone()].concat()))
                        },
                        [Value::Nil, x] => Ok(x.clone()),
                        [x, Value::Nil] => Ok(x.clone()),
                        [Value::Array(t, x), y] => {
                            let ytype = y.get_type();

                            if *t != ytype {
                                return Err(RuntimeError::TypeError(t.clone(), ytype));
                            }

                            // NOTE: use y's type instead of the arrays type.
                            // an `empty` array has Any type, but any value will have a fixed type.
                            // this converts the empty array into a typed array.
                            Ok(Value::Array(ytype, [x.clone(), vec![y.clone()]].concat()))
                        },
                        [x, Value::Array(t, y)] => {
                            let xtype = x.get_type();

                            if *t != xtype {
                                return Err(RuntimeError::TypeError(t.clone(), xtype));
                            }

                            // NOTE: read above
                            Ok(Value::Array(xtype, [vec![x.clone()], y.clone()].concat()))
                        },
                        _ => Err(RuntimeError::NoOverloadForTypes("+".into(), args)),
                    }
                    Op::Sub => match &args[..] {
                        [Value::Int(x), Value::Int(y)] => Ok(Value::Int(x - y)),
                        [Value::Int(x), Value::Float(y)] => Ok(Value::Float(*x as f64 - y)),
                        [Value::Float(x), Value::Int(y)] => Ok(Value::Float(x - *y as f64)),
                        [Value::Float(x), Value::Float(y)] => Ok(Value::Float(x - y)),
                        [Value::Nil, x] => Ok(x.clone()),
                        [x, Value::Nil] => Ok(x.clone()),
                        _ => Err(RuntimeError::NoOverloadForTypes("-".into(), args)),
                    }
                    Op::Mul => match &args[..] {
                        [Value::Int(x), Value::Int(y)] => Ok(Value::Int(x * y)),
                        [Value::Int(x), Value::Float(y)] => Ok(Value::Float(*x as f64 * y)),
                        [Value::Float(x), Value::Int(y)] => Ok(Value::Float(x * *y as f64)),
                        [Value::Float(x), Value::Float(y)] => Ok(Value::Float(x * y)),
                        [Value::Nil, x] => Ok(x.clone()),
                        [x, Value::Nil] => Ok(x.clone()),
                        _ => Err(RuntimeError::NoOverloadForTypes("*".into(), args)),
                    }
                    Op::Div => match &args[..] {
                        [Value::Int(x), Value::Int(y)] => Ok(Value::Float(*x as f64 / *y as f64)),
                        [Value::Float(x), Value::Int(y)] => Ok(Value::Float(x / *y as f64)),
                        [Value::Int(x), Value::Float(y)] => Ok(Value::Float(*x as f64 / y)),
                        [Value::Float(x), Value::Float(y)] => Ok(Value::Float(x / y)),
                        [Value::Nil, x] => Ok(x.clone()),
                        [x, Value::Nil] => Ok(x.clone()),
                        _ => Err(RuntimeError::NoOverloadForTypes("/".into(), args)),
                    }
                    Op::FloorDiv => match &args[..] {
                        [Value::Int(x), Value::Int(y)] => Ok(Value::Int(x / y)),
                        [Value::Float(x), Value::Int(y)] => Ok(Value::Int(*x as i64 / y)),
                        [Value::Int(x), Value::Float(y)] => Ok(Value::Int(x / *y as i64)),
                        [Value::Float(x), Value::Float(y)] => Ok(Value::Int(*x as i64 / *y as i64)),
                        [Value::Nil, x] => Ok(x.clone()),
                        [x, Value::Nil] => Ok(x.clone()),
                        _ => Err(RuntimeError::NoOverloadForTypes("//".into(), args)),
                    }
                    Op::Exp => match &args[..] {
                        [Value::Int(x), Value::Int(y)] => Ok(Value::Float((*x as f64).powf(*y as f64))),
                        [Value::Float(x), Value::Int(y)] => Ok(Value::Float(x.powf(*y as f64))),
                        [Value::Int(x), Value::Float(y)] => Ok(Value::Float((*x as f64).powf(*y))),
                        [Value::Float(x), Value::Float(y)] => Ok(Value::Float(x.powf(*y))),
                        [Value::Nil, x] => Ok(x.clone()),
                        [x, Value::Nil] => Ok(x.clone()),
                        _ => Err(RuntimeError::NoOverloadForTypes("**".into(), args)),
                    }
                    Op::Mod => match &args[..] {
                        [Value::Int(x), Value::Int(y)] => Ok(Value::Int(x % y)),
                        [Value::Float(x), Value::Int(y)] => Ok(Value::Float(x % *y as f64)),
                        [Value::Int(x), Value::Float(y)] => Ok(Value::Float(*x as f64 % y)),
                        [Value::Float(x), Value::Float(y)] => Ok(Value::Float(x % y)),
                        [Value::Nil, x] => Ok(x.clone()),
                        [x, Value::Nil] => Ok(x.clone()),
                        _ => Err(RuntimeError::NoOverloadForTypes("%".into(), args)),
                    }
                    Op::GreaterThan => match &args[..] {
                        [Value::Int(x), Value::Int(y)] => Ok(Value::Bool(x > y)),
                        [Value::Float(x), Value::Int(y)] => Ok(Value::Bool(*x > *y as f64)),
                        [Value::Int(x), Value::Float(y)] => Ok(Value::Bool(*x as f64 > *y)),
                        [Value::Float(x), Value::Float(y)] => Ok(Value::Bool(x > y)),
                        _ => Err(RuntimeError::NoOverloadForTypes(">".into(), args)),
                    }
                    Op::GreaterThanOrEqualTo => match &args[..] {
                        [Value::Int(x), Value::Int(y)] => Ok(Value::Bool(x >= y)),
                        [Value::Float(x), Value::Int(y)] => Ok(Value::Bool(*x >= *y as f64)),
                        [Value::Int(x), Value::Float(y)] => Ok(Value::Bool(*x as f64 >= *y)),
                        [Value::Float(x), Value::Float(y)] => Ok(Value::Bool(x >= y)),
                        _ => Err(RuntimeError::NoOverloadForTypes(">=".into(), args)),
                    }
                    Op::LessThan => match &args[..] {
                        [Value::Int(x), Value::Int(y)] => Ok(Value::Bool(x < y)),
                        [Value::Float(x), Value::Int(y)] => Ok(Value::Bool(*x < *y as f64)),
                        [Value::Int(x), Value::Float(y)] => Ok(Value::Bool((*x as f64) < *y)),
                        [Value::Float(x), Value::Float(y)] => Ok(Value::Bool(x < y)),
                        _ => Err(RuntimeError::NoOverloadForTypes("<".into(), args)),
                    }
                    Op::LessThanOrEqualTo => match &args[..] {
                        [Value::Int(x), Value::Int(y)] => Ok(Value::Bool(x <= y)),
                        [Value::Float(x), Value::Int(y)] => Ok(Value::Bool(*x <= *y as f64)),
                        [Value::Int(x), Value::Float(y)] => Ok(Value::Bool(*x as f64 <= *y)),
                        [Value::Float(x), Value::Float(y)] => Ok(Value::Bool(x <= y)),
                        _ => Err(RuntimeError::NoOverloadForTypes("<=".into(), args)),
                    }
                    Op::EqualTo => match &args[..] {
                        [Value::Int(x), Value::Int(y)] => Ok(Value::Bool(x == y)),
                        [Value::Float(x), Value::Int(y)] => Ok(Value::Bool(*x == *y as f64)),
                        [Value::Int(x), Value::Float(y)] => Ok(Value::Bool(*x as f64 == *y)),
                        [Value::Float(x), Value::Float(y)] => Ok(Value::Bool(x == y)),
                        _ => Err(RuntimeError::NoOverloadForTypes("==".into(), args)),
                    }
                    Op::NotEqualTo => match &args[..] {
                        [Value::Int(x), Value::Int(y)] => Ok(Value::Bool(x != y)),
                        [Value::Float(x), Value::Int(y)] => Ok(Value::Bool(*x != *y as f64)),
                        [Value::Int(x), Value::Float(y)] => Ok(Value::Bool(*x as f64 != *y)),
                        [Value::Float(x), Value::Float(y)] => Ok(Value::Bool(x != y)),
                        _ => Err(RuntimeError::NoOverloadForTypes("!=".into(), args)),
                    }
                    Op::Not => match &args[0] {
                        Value::Bool(b) => Ok(Value::Bool(!b)),
                        _ => Err(RuntimeError::NoOverloadForTypes("!".into(), args)),
                    }
                    Op::Or => match &args[..] {
                        [Value::Bool(x), Value::Bool(y)] => Ok(Value::Bool(*x || *y)),
                        [Value::Nil, x] => Ok(x.clone()),
                        [x, Value::Nil] => Ok(x.clone()),
                        _ => Err(RuntimeError::NoOverloadForTypes("||".into(), args)),
                    }
                    Op::And => match &args[..] {
                        [Value::Bool(x), Value::Bool(y)] => Ok(Value::Bool(*x && *y)),
                        [Value::Nil, x] => Ok(x.clone()),
                        [x, Value::Nil] => Ok(x.clone()),
                        _ => Err(RuntimeError::NoOverloadForTypes("&&".into(), args)),
                    }
                    Op::Compose => match &args[..] {
                        [_, v] => Ok(v.clone()),
                        _ => Err(RuntimeError::NoOverloadForTypes("~".into(), args)),
                    }
                    Op::Head => match &args[0] {
                        Value::Array(_, x) => Ok(x.first().ok_or(RuntimeError::EmptyArray)?.clone()),
                        _ => Err(RuntimeError::NoOverloadForTypes("head".into(), args)),
                    }
                    Op::Tail => match &args[0] {
                        Value::Array(t, x) => Ok(Value::Array(t.clone(), if x.len() > 0 { x[1..].to_vec() } else { vec![] })),
                        _ => Err(RuntimeError::NoOverloadForTypes("head".into(), args)),
                    }
                    Op::Init => match &args[0] {
                        Value::Array(t, x) => Ok(Value::Array(t.clone(), if x.len() > 0 { x[..x.len() - 1].to_vec() } else { vec![] })),
                        _ => Err(RuntimeError::NoOverloadForTypes("head".into(), args)),
                    }
                    Op::Fini => match &args[0] {
                        Value::Array(_, x) => Ok(x.last().ok_or(RuntimeError::EmptyArray)?.clone()),
                        _ => Err(RuntimeError::NoOverloadForTypes("head".into(), args)),
                    }
                    Op::Id => match &args[0] {
                        x => Ok(x.clone()),
                    }
                    Op::IntCast => match &args[0] {
                        Value::Int(x) => Ok(Value::Int(*x)),
                        Value::Float(x) => Ok(Value::Int(*x as i64)),
                        Value::Bool(x) => Ok(Value::Int(if *x { 1 } else { 0 })),
                        Value::String(x) => {
                            let r: i64 = x.parse().map_err(|_| RuntimeError::ParseFail(x.clone(), Type::Int))?;
                            Ok(Value::Int(r))
                        }
                            x => Err(RuntimeError::NoOverloadForTypes("int".into(), vec![x.clone()])),
                    }
                    Op::FloatCast => match &args[0] {
                        Value::Int(x) => Ok(Value::Float(*x as f64)),
                        Value::Float(x) => Ok(Value::Float(*x)),
                        Value::Bool(x) => Ok(Value::Float(if *x { 1.0 } else { 0.0 })),
                        Value::String(x) => {
                            let r: f64 = x.parse().map_err(|_| RuntimeError::ParseFail(x.clone(), Type::Int))?;
                            Ok(Value::Float(r))
                        }
                        x => Err(RuntimeError::NoOverloadForTypes("float".into(), vec![x.clone()])),
                    }
                    Op::BoolCast => match &args[0] {
                        Value::Int(x) => Ok(Value::Bool(*x != 0)),
                        Value::Float(x) => Ok(Value::Bool(*x != 0.0)),
                        Value::Bool(x) => Ok(Value::Bool(*x)),
                        Value::String(x) => Ok(Value::Bool(!x.is_empty())),
                        Value::Array(_, vec) => Ok(Value::Bool(!vec.is_empty())),
                        x => Err(RuntimeError::NoOverloadForTypes("bool".into(), vec![x.clone()])),
                    }
                    Op::StringCast => Ok(Value::String(format!("{}", &args[0]))),
                    Op::Print => match &args[0] {
                        Value::String(s) => {
                            println!("{s}");
                            Ok(Value::Nil)
                        }
                        x => {
                            println!("{x}");
                            Ok(Value::Nil)
                        }
                    }
                    _ => unreachable!(),
                }
            }
            ParseTree::Equ(ident, body, scope) => {
                if self.variable_exists(&ident) {
                    Err(RuntimeError::ImmutableError(ident.clone()))
                } else {
                    let value = self.exec(body)?;
                    let g = self.globals.clone();

                    Executor::new(self.exprs, &mut self.globals)
                        .locals(self.locals.clone())
                        .add_local(ident, Arc::new(Mutex::new(Object::value(value, g, self.locals.to_owned()))))
                        .exec(scope)
                }
            },
            ParseTree::LazyEqu(ident, body, scope) => {
                if self.variable_exists(&ident) {
                    Err(RuntimeError::ImmutableError(ident.clone()))
                } else {
                    let g = self.globals.clone();
                    Executor::new(self.exprs, &mut self.globals)
                        .locals(self.locals.clone())
                        .add_local(ident, Arc::new(Mutex::new(Object::variable(*body, g, self.locals.to_owned()))))
                        .exec(scope)
                }
            },
            ParseTree::FunctionDefinition(func, scope) => {
                let g = self.globals.clone();
                Executor::new(self.exprs, &mut self.globals)
                    .locals(self.locals.clone())
                    .add_local(func.name().unwrap().to_string(),
                        Arc::new(Mutex::new(Object::function(
                            func
                                .globals(g)
                                .locals(self.locals.clone()), HashMap::new(), HashMap::new()))))
                    .exec(scope)
            },
            ParseTree::FunctionCall(ident, args) => {
                let obj = self.get_object_mut(&ident)?;
                let v = Self::eval(obj)?;

                match v {
                    Value::Function(mut f) => {
                        let mut args: Vec<_> = args.into_iter()
                            .map(|x| Arc::new(Mutex::new(Object::variable(x, self.globals.clone(), self.locals.clone()))))
                            .collect();

                        for arg in &mut args {
                            Self::eval(arg)?;
                        }

                        f.call(args)
                    },
                    _ => Err(RuntimeError::FunctionUndefined(ident.clone()))
                }
            },
            ParseTree::_FunctionCallLocal(_idx, _args) => todo!(),
            ParseTree::If(cond, body) => if match self.exec(cond)? {
                Value::Float(f) => f != 0.0,
                Value::Int(i) => i != 0,
                Value::Bool(b) => b,
                Value::String(s) => !s.is_empty(),
                Value::Array(_, vec) => !vec.is_empty(),
                Value::Nil => false,
                x => return Err(RuntimeError::NoOverloadForTypes("?".into(), vec![x])),
            } {
                self.exec(body)
            } else {
                Ok(Value::Nil)
            },
            ParseTree::IfElse(cond, istrue, isfalse) => if match self.exec(cond)? {
                Value::Float(f) => f != 0.0,
                Value::Int(i) => i != 0,
                Value::Bool(b) => b,
                Value::String(s) => !s.is_empty(),
                Value::Array(_, vec) => !vec.is_empty(),
                Value::Nil => false,
                x => return Err(RuntimeError::NoOverloadForTypes("??".into(), vec![x])),
            } {
                self.exec(istrue)
            } else {
                self.exec(isfalse)
            },
            ParseTree::Variable(ident) => {
                let obj = self.get_object_mut(&ident)?;

                let v = obj.lock().unwrap().eval()?;

                Ok(v)
            },
            ParseTree::Value(value) => Ok(value),
            ParseTree::LambdaDefinition(func) => Ok(Value::Function(func.globals(self.globals.clone()).locals(self.locals.clone()))),
            ParseTree::Nop => Ok(Value::Nil),
            ParseTree::Export(names) => {
                for name in names {
                    let obj = self.locals.remove(&name).ok_or(RuntimeError::VariableUndefined(name.clone()))?;
                    self.globals.insert(name, obj);
                }

                Ok(Value::Nil)
            }
            ParseTree::NonCall(name) => {
                let obj = self.get_object_mut(&name)?;

                let v = obj.lock().unwrap().eval()?;

                Ok(v)
            }
            ParseTree::_Local(_idx) => todo!(),
            ParseTree::GeneratedFunction(function) => Ok(Value::Function(function.globals(self.globals.clone()).locals(self.locals.clone()))),
        }
    }
}

impl<'a, I: Iterator<Item = Result<ParseTree, ParseError>>> Iterator for Executor<'a, I> {
    type Item = Result<Value, RuntimeError>;

    fn next(&mut self) -> Option<Self::Item> {
        let expr = self.exprs.next();

        match expr {
            Some(Ok(expr)) => Some(self.exec(Box::new(expr))),
            Some(Err(e)) => Some(Err(RuntimeError::ParseError(e))),
            None => None,
        }
    }
}