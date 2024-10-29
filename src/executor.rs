use super::{Value, Type, Object};
use super::parser::ParseTree;
use super::tokenizer::Op;
use super::error::Error;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Executes an input of ParseTrees
pub(crate) struct Executor {
    globals: HashMap<String, Arc<Mutex<Object>>>,
    locals: HashMap<String, Arc<Mutex<Object>>>,
}

impl Executor {
    pub(crate) fn new() -> Self {
        Self {
            globals: HashMap::new(),
            locals: HashMap::new(),
        }
    }

    pub(crate) fn _values<I>(mut self, iter: I) -> impl Iterator<Item = Result<Value, Error>>
    where
        I: Iterator<Item = Result<ParseTree, Error>>
    {
        iter.map(move |x| self.exec(x?))
    }

    pub(crate) fn add_global(mut self, k: String, v: Arc<Mutex<Object>>) -> Self {
        self.globals.insert(k, v);
        self
    }

    pub(crate) fn add_globals<Globals: IntoIterator<Item = (String, Arc<Mutex<Object>>)>>(self, globals: Globals) -> Self {
        globals.into_iter().fold(self, |acc, (k, v)| acc.add_global(k, v))
    }

    pub(crate) fn locals(mut self, locals: HashMap<String, Arc<Mutex<Object>>>) -> Self {
        self.locals = locals;
        self
    }

    pub(crate) fn add_local(mut self, k: String, v: Arc<Mutex<Object>>) -> Self {
        self.locals.insert(k, v);
        self
    }

    pub(crate) fn add_local_mut(&mut self, k: String, v: Arc<Mutex<Object>>) -> &mut Self {
        self.locals.insert(k, v);
        self
    }

    fn _get_object(&self, ident: &String) -> Result<&Arc<Mutex<Object>>, Error> {
        self.locals.get(ident).or(self.globals.get(ident))
            .ok_or(Error::new(format!("undefined identifier {}", ident.clone())))
    }

    fn get_object_mut(&mut self, ident: &String) -> Result<&mut Arc<Mutex<Object>>, Error> {
        self.locals.get_mut(ident).or(self.globals.get_mut(ident))
            .ok_or(Error::new(format!("undefined identifier {}", ident.clone())))
    }

    fn variable_exists(&self, ident: &String) -> bool {
        self.locals.contains_key(ident) || self.globals.contains_key(ident)
    }

    fn eval(obj: &mut Arc<Mutex<Object>>) -> Result<Value, Error> {
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

    pub(crate) fn exec(&mut self, tree: ParseTree) -> Result<Value, Error> {
        match tree {
            ParseTree::Operator(op, args) => {
                let args: Vec<Value> = args.into_iter()
                    .map(|x| self.exec(x)).collect::<Result<_, _>>()?;

                match op {
                    Op::Add => match &args[..] {
                        [Value::Int(x), Value::Int(y)] => Ok(Value::Int(x + y)),
                        [Value::Float(x), Value::Int(y)] => Ok(Value::Float(x + *y as f64)),
                        [Value::Int(x), Value::Float(y)] => Ok(Value::Float(*x as f64 + y)),
                        [Value::Float(x), Value::Float(y)] => Ok(Value::Float(x + y)),
                        [Value::String(x), Value::String(y)] => Ok(Value::String(format!("{x}{y}"))),
                        [Value::Nil, x] => Ok(x.clone()),
                        [x, Value::Nil] => Ok(x.clone()),
                        [x, y] => Err(Error::new(format!("no overload of + exists for types {} and {}", x.get_type(), y.get_type()))),
                        _ => unreachable!(),
                    }
                    Op::Concat => match &args[..] {
                        [Value::Array(xtype, x), Value::Array(ytype, y)] => {
                            if xtype != ytype {
                                return Err(Error::new(format!("expected type {} but found {}", xtype, ytype)));
                            }

                            Ok(Value::Array(xtype.clone(), [x.clone(), y.clone()].concat()))
                        },
                        _ => Err(Error::new("++".into())),
                    }
                    Op::Prepend => match &args[..] {
                        [x, Value::Array(t, y)] => {
                            let xtype = x.get_type();

                            if *t != xtype {
                                return Err(Error::new(format!("expected type {} but found {}", t, xtype)));
                            }

                            Ok(Value::Array(xtype, [vec![x.clone()], y.clone()].concat()))
                        },
                        [x, y] => Err(Error::new(format!("no overload of [+ exists for types {} and {}", x.get_type(), y.get_type()))),
                        _ => unreachable!(),
                    }
                    Op::Append => match &args[..] {
                        [Value::Array(t, y), x] => {
                            let xtype = x.get_type();

                            if *t != xtype {
                                return Err(Error::new(format!("expected type {} but found {}", t, xtype)));
                            }

                            Ok(Value::Array(xtype, [y.clone(), vec![x.clone()]].concat()))
                        },
                        _ => Err(Error::new("+]".into())),
                    }
                    Op::Insert => match &args[..] {
                        [Value::Int(idx), x, Value::Array(t, y)] => {
                            let mut y = y.clone();
                            let xtype = x.get_type();

                            if *t != xtype {
                                return Err(Error::new(format!("expected type {} but found {}", t, xtype)));
                            }
                            
                            if *idx as usize > y.len() {
                                return Err(Error::new("attempt to insert out of array len".into()));
                            }

                            y.insert(*idx as usize, x.clone());

                            Ok(Value::Array(t.clone(), y))
                        },
                        _ => Err(Error::new("[+]".into())),
                    }
                    Op::Sub => match &args[..] {
                        [Value::Int(x), Value::Int(y)] => Ok(Value::Int(x - y)),
                        [Value::Int(x), Value::Float(y)] => Ok(Value::Float(*x as f64 - y)),
                        [Value::Float(x), Value::Int(y)] => Ok(Value::Float(x - *y as f64)),
                        [Value::Float(x), Value::Float(y)] => Ok(Value::Float(x - y)),
                        [Value::Nil, x] => Ok(x.clone()),
                        [x, Value::Nil] => Ok(x.clone()),
                        _ => Err(Error::new("todo: actual error output".into())),
                    }
                    Op::Mul => match &args[..] {
                        [Value::Int(x), Value::Int(y)] => Ok(Value::Int(x * y)),
                        [Value::Int(x), Value::Float(y)] => Ok(Value::Float(*x as f64 * y)),
                        [Value::Float(x), Value::Int(y)] => Ok(Value::Float(x * *y as f64)),
                        [Value::Float(x), Value::Float(y)] => Ok(Value::Float(x * y)),
                        [Value::Nil, x] => Ok(x.clone()),
                        [x, Value::Nil] => Ok(x.clone()),
                        _ => Err(Error::new("todo: actual error output".into())),
                    }
                    Op::Div => match &args[..] {
                        [Value::Int(x), Value::Int(y)] => Ok(Value::Float(*x as f64 / *y as f64)),
                        [Value::Float(x), Value::Int(y)] => Ok(Value::Float(x / *y as f64)),
                        [Value::Int(x), Value::Float(y)] => Ok(Value::Float(*x as f64 / y)),
                        [Value::Float(x), Value::Float(y)] => Ok(Value::Float(x / y)),
                        [Value::Nil, x] => Ok(x.clone()),
                        [x, Value::Nil] => Ok(x.clone()),
                        _ => Err(Error::new("todo: actual error output".into())),
                    }
                    Op::FloorDiv => match &args[..] {
                        [Value::Int(x), Value::Int(y)] => Ok(Value::Int(x / y)),
                        [Value::Float(x), Value::Int(y)] => Ok(Value::Int(*x as i64 / y)),
                        [Value::Int(x), Value::Float(y)] => Ok(Value::Int(x / *y as i64)),
                        [Value::Float(x), Value::Float(y)] => Ok(Value::Int(*x as i64 / *y as i64)),
                        [Value::Nil, x] => Ok(x.clone()),
                        [x, Value::Nil] => Ok(x.clone()),
                        _ => Err(Error::new("todo: actual error output".into())),
                    }
                    Op::Exp => match &args[..] {
                        [Value::Int(x), Value::Int(y)] => Ok(Value::Float((*x as f64).powf(*y as f64))),
                        [Value::Float(x), Value::Int(y)] => Ok(Value::Float(x.powf(*y as f64))),
                        [Value::Int(x), Value::Float(y)] => Ok(Value::Float((*x as f64).powf(*y))),
                        [Value::Float(x), Value::Float(y)] => Ok(Value::Float(x.powf(*y))),
                        [Value::Nil, x] => Ok(x.clone()),
                        [x, Value::Nil] => Ok(x.clone()),
                        _ => Err(Error::new("todo: fsadfdsf".into())),
                    }
                    Op::Mod => match &args[..] {
                        [Value::Int(x), Value::Int(y)] => Ok(Value::Int(x % y)),
                        [Value::Float(x), Value::Int(y)] => Ok(Value::Float(x % *y as f64)),
                        [Value::Int(x), Value::Float(y)] => Ok(Value::Float(*x as f64 % y)),
                        [Value::Float(x), Value::Float(y)] => Ok(Value::Float(x % y)),
                        [Value::Nil, x] => Ok(x.clone()),
                        [x, Value::Nil] => Ok(x.clone()),
                        _ => Err(Error::new("todo: actual error output".into())),
                    }
                    Op::GreaterThan => match &args[..] {
                        [Value::Int(x), Value::Int(y)] => Ok(Value::Bool(x > y)),
                        [Value::Float(x), Value::Int(y)] => Ok(Value::Bool(*x > *y as f64)),
                        [Value::Int(x), Value::Float(y)] => Ok(Value::Bool(*x as f64 > *y)),
                        [Value::Float(x), Value::Float(y)] => Ok(Value::Bool(x > y)),
                        _ => Err(Error::new("todo: actual error output".into())),
                    }
                    Op::GreaterThanOrEqualTo => match &args[..] {
                        [Value::Int(x), Value::Int(y)] => Ok(Value::Bool(x >= y)),
                        [Value::Float(x), Value::Int(y)] => Ok(Value::Bool(*x >= *y as f64)),
                        [Value::Int(x), Value::Float(y)] => Ok(Value::Bool(*x as f64 >= *y)),
                        [Value::Float(x), Value::Float(y)] => Ok(Value::Bool(x >= y)),
                        _ => Err(Error::new("todo: actual error output".into())),
                    }
                    Op::LessThan => match &args[..] {
                        [Value::Int(x), Value::Int(y)] => Ok(Value::Bool(x < y)),
                        [Value::Float(x), Value::Int(y)] => Ok(Value::Bool(*x < *y as f64)),
                        [Value::Int(x), Value::Float(y)] => Ok(Value::Bool((*x as f64) < *y)),
                        [Value::Float(x), Value::Float(y)] => Ok(Value::Bool(x < y)),
                        _ => Err(Error::new("todo: actual error output".into())),
                    }
                    Op::LessThanOrEqualTo => match &args[..] {
                        [Value::Int(x), Value::Int(y)] => Ok(Value::Bool(x <= y)),
                        [Value::Float(x), Value::Int(y)] => Ok(Value::Bool(*x <= *y as f64)),
                        [Value::Int(x), Value::Float(y)] => Ok(Value::Bool(*x as f64 <= *y)),
                        [Value::Float(x), Value::Float(y)] => Ok(Value::Bool(x <= y)),
                        _ => Err(Error::new("todo: actual error output".into())),
                    }
                    Op::EqualTo => match &args[..] {
                        [Value::Int(x), Value::Int(y)] => Ok(Value::Bool(x == y)),
                        [Value::Float(x), Value::Int(y)] => Ok(Value::Bool(*x == *y as f64)),
                        [Value::Int(x), Value::Float(y)] => Ok(Value::Bool(*x as f64 == *y)),
                        [Value::Float(x), Value::Float(y)] => Ok(Value::Bool(x == y)),
                        _ => Err(Error::new("todo: actual error output".into())),
                    }
                    Op::NotEqualTo => match &args[..] {
                        [Value::Int(x), Value::Int(y)] => Ok(Value::Bool(x != y)),
                        [Value::Float(x), Value::Int(y)] => Ok(Value::Bool(*x != *y as f64)),
                        [Value::Int(x), Value::Float(y)] => Ok(Value::Bool(*x as f64 != *y)),
                        [Value::Float(x), Value::Float(y)] => Ok(Value::Bool(x != y)),
                        _ => Err(Error::new("todo: actual error output".into())),
                    }
                    Op::Not => match &args[0] {
                        Value::Bool(b) => Ok(Value::Bool(!b)),
                        _ => Err(Error::new("todo: actual error output".into())),
                    }
                    Op::Or => match &args[..] {
                        [Value::Bool(x), Value::Bool(y)] => Ok(Value::Bool(*x || *y)),
                        [Value::Nil, x] => Ok(x.clone()),
                        [x, Value::Nil] => Ok(x.clone()),
                        _ => Err(Error::new("todo: actual error output".into())),
                    }
                    Op::And => match &args[..] {
                        [Value::Bool(x), Value::Bool(y)] => Ok(Value::Bool(*x && *y)),
                        [Value::Nil, x] => Ok(x.clone()),
                        [x, Value::Nil] => Ok(x.clone()),
                        _ => Err(Error::new("todo: actual error output".into())),
                    }
                    Op::Compose => match &args[..] {
                        [_, v] => Ok(v.clone()),
                        _ => Err(Error::new("todo: actual error output".into())),
                    }
                    Op::Head => match &args[0] {
                        Value::Array(_, x) => Ok(x.first().ok_or(Error::new(format!("passed an empty array to head")))?.clone()),
                        _ => Err(Error::new("head".into())),
                    }
                    Op::Tail => match &args[0] {
                        Value::Array(t, x) => Ok(Value::Array(t.clone(), if x.len() > 0 { x[1..].to_vec() } else { vec![] })),
                        _ => Err(Error::new("tail".into())),
                    }
                    Op::Init => match &args[0] {
                        Value::Array(t, x) => Ok(Value::Array(t.clone(), if x.len() > 0 { x[..x.len() - 1].to_vec() } else { vec![] })),
                        _ => Err(Error::new("init".into())),
                    }
                    Op::Fini => match &args[0] {
                        Value::Array(_, x) => Ok(x.last().ok_or(Error::new(format!("passed an empty array to fini")))?.clone()),
                        _ => Err(Error::new("fini".into())),
                    }
                    Op::Id => match &args[0] {
                        x => Ok(x.clone()),
                    }
                    Op::IntCast => match &args[0] {
                        Value::Int(x) => Ok(Value::Int(*x)),
                        Value::Float(x) => Ok(Value::Int(*x as i64)),
                        Value::Bool(x) => Ok(Value::Int(if *x { 1 } else { 0 })),
                        Value::String(x) => {
                            let r: i64 = x.parse().map_err(|_| Error::new(format!("failed to parse {} into {}", x, Type::Int)))?;
                            Ok(Value::Int(r))
                        }
                            x => Err(Error::new(format!("no possible conversion from {} into {}", x, Type::Int))),
                    }
                    Op::FloatCast => match &args[0] {
                        Value::Int(x) => Ok(Value::Float(*x as f64)),
                        Value::Float(x) => Ok(Value::Float(*x)),
                        Value::Bool(x) => Ok(Value::Float(if *x { 1.0 } else { 0.0 })),
                        Value::String(x) => {
                            let r: f64 = x.parse().map_err(|_| Error::new(format!("failed to parse {} into {}", x, Type::Float)))?;
                            Ok(Value::Float(r))
                        }
                        x => Err(Error::new(format!("no possible conversion from {} into {}", x, Type::Float))),
                    }
                    Op::BoolCast => match &args[0] {
                        Value::Int(x) => Ok(Value::Bool(*x != 0)),
                        Value::Float(x) => Ok(Value::Bool(*x != 0.0)),
                        Value::Bool(x) => Ok(Value::Bool(*x)),
                        Value::String(x) => Ok(Value::Bool(!x.is_empty())),
                        Value::Array(_, vec) => Ok(Value::Bool(!vec.is_empty())),
                        x => Err(Error::new(format!("no possible conversion from {} into {}", x, Type::Bool))),
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
                    Err(Error::new(format!("attempt to override value of variable {ident}")))
                } else {
                    let value = self.exec(*body)?;
                    let g = self.globals.clone();

                    let r = self.add_local_mut(ident.clone(), Arc::new(Mutex::new(Object::value(value, g, self.locals.to_owned()))))
                        .exec(*scope);

                    self.locals.remove(&ident);

                    r
                }
            },
            ParseTree::LazyEqu(ident, body, scope) => {
                if self.variable_exists(&ident) {
                    Err(Error::new(format!("attempt to override value of variable {ident}")))
                } else {
                    let g = self.globals.clone();
                    let r = self.add_local_mut(ident.clone(), Arc::new(Mutex::new(Object::variable(*body, g, self.locals.to_owned()))))
                        .exec(*scope);

                    self.locals.remove(&ident);

                    r
                }
            },
            ParseTree::FunctionDefinition(func, scope) => {
                let name = func.name().unwrap().to_string();
                let g = self.globals.clone();
                let r = self.add_local_mut(name.clone(),
                        Arc::new(Mutex::new(Object::function(
                            func
                                .globals(g)
                                .locals(self.locals.clone()), HashMap::new(), HashMap::new()))))
                    .exec(*scope);

                self.locals.remove(&name);

                r
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
                    _ => Err(Error::new(format!("the function {ident} is not defined")))
                }
            },
            ParseTree::_FunctionCallLocal(_idx, _args) => todo!(),
            ParseTree::If(cond, body) => if match self.exec(*cond)? {
                Value::Float(f) => f != 0.0,
                Value::Int(i) => i != 0,
                Value::Bool(b) => b,
                Value::String(s) => !s.is_empty(),
                Value::Array(_, vec) => !vec.is_empty(),
                Value::Nil => false,
                x => return Err(Error::new(format!("could not convert {x} into a bool for truthiness check"))),
            } {
                self.exec(*body)
            } else {
                Ok(Value::Nil)
            },
            ParseTree::IfElse(cond, istrue, isfalse) => if match self.exec(*cond)? {
                Value::Float(f) => f != 0.0,
                Value::Int(i) => i != 0,
                Value::Bool(b) => b,
                Value::String(s) => !s.is_empty(),
                Value::Array(_, vec) => !vec.is_empty(),
                Value::Nil => false,
                x => return Err(Error::new(format!("could not convert {x} into a bool for truthiness check"))),
            } {
                self.exec(*istrue)
            } else {
                self.exec(*isfalse)
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
                    let obj = self.locals.remove(&name).ok_or(Error::new(format!("attempt to export an object that was not defined")))?;
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
