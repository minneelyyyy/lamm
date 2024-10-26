use crate::parser::ParseTree;
use crate::executor::{Executor, RuntimeError};
use crate::{Type, Object, Value};

use std::collections::HashMap;
use std::fmt::{self, Display};
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug, PartialEq)]
pub struct FunctionType(pub Box<Type>, pub Vec<Type>);

impl Display for FunctionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Function({}, {})", self.0, self.1.iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(", "))
    }
}

#[derive(Clone, Debug)]
pub struct Function {
    name: Option<String>,
    t: FunctionType,
    globals: Option<HashMap<String, Arc<Mutex<Object>>>>,
    locals: Option<HashMap<String, Arc<Mutex<Object>>>>,
    arg_names: Option<Vec<String>>,
    body: Box<ParseTree>,
}

impl Function {
    pub(crate) fn lambda(t: FunctionType, arg_names: Vec<String>, body: Box<ParseTree>) -> Self {
        Self {
            name: None,
            t,
            globals: None,
            locals: None,
            arg_names: Some(arg_names),
            body
        }
    }

    pub(crate) fn named(name: &str, t: FunctionType, arg_names: Vec<String>, body: Box<ParseTree>) -> Self {
        Self {
            name: Some(name.to_string()),
            t,
            globals: None,
            locals: None,
            arg_names: Some(arg_names),
            body
        }
    }

    pub(crate) fn _generated(t: FunctionType, body: Box<ParseTree>) -> Self {
        Self {
            name: None,
            t,
            globals: None,
            locals: None,
            arg_names: None,
            body,
        }
    }

    pub(crate) fn locals(mut self, locals: HashMap<String, Arc<Mutex<Object>>>) -> Self {
        self.locals = Some(locals);
        self
    }

    pub(crate) fn globals(mut self, globals: HashMap<String, Arc<Mutex<Object>>>) -> Self {
        self.globals = Some(globals);
        self
    }

    fn _replace_locals(mut self) -> Self {
        fn replace_locals_(body: Box<ParseTree>, args: &Vec<String>) -> Box<ParseTree> {
            match *body {
                ParseTree::Operator(op, a) =>
                    Box::new(ParseTree::Operator(
                        op,
                        a.into_iter().map(|x| *replace_locals_(Box::new(x), args)).collect())),
                ParseTree::Equ(name, body, scope) =>
                    Box::new(ParseTree::Equ(
                        name,
                        replace_locals_(body, args),
                        replace_locals_(scope, args))),
                ParseTree::LazyEqu(name, body, scope) =>
                    Box::new(ParseTree::LazyEqu(
                        name,
                        replace_locals_(body, args),
                        replace_locals_(scope, args))),
                ParseTree::FunctionDefinition(func, scope) =>
                    Box::new(ParseTree::FunctionDefinition(func, replace_locals_(scope, args))),
                ParseTree::LambdaDefinition(_) => body,
                ParseTree::FunctionCall(ref func, ref a) =>
                    if let Some(idx) = args.into_iter().position(|r| *r == *func) {
                        Box::new(ParseTree::_FunctionCallLocal(
                            idx,
                            a.into_iter().map(|x| *replace_locals_(Box::new(x.clone()), args)).collect()))
                    } else {
                        Box::new(ParseTree::FunctionCall(
                            func.clone(),
                            a.into_iter().map(|x| *replace_locals_(Box::new(x.clone()), args)).collect()))
                    }
                ParseTree::Variable(ref var) => {
                    if let Some(idx) = args.into_iter().position(|r| *r == *var) {
                        Box::new(ParseTree::_Local(idx))
                    } else {
                        body
                    }
                },
                ParseTree::If(cond, branch) =>
                    Box::new(ParseTree::If(replace_locals_(cond, args),
                                           replace_locals_(branch, args))),
                ParseTree::IfElse(cond, t, f) =>
                    Box::new(ParseTree::IfElse(replace_locals_(cond, args),
                                               replace_locals_(t, args),
                                               replace_locals_(f, args))),
                ParseTree::_FunctionCallLocal(_, _) => body,
                ParseTree::_Local(_) => body,
                ParseTree::Value(_) => body,
                ParseTree::Nop => body,
                ParseTree::Export(_) => body,
                ParseTree::NonCall(ref var) => if let Some(idx) =
                    args.into_iter().position(|r| *r == *var) {
                        Box::new(ParseTree::_Local(idx))
                    } else {
                        body
                    }
                ParseTree::GeneratedFunction(_) => todo!(),
            }
        }

        self.body = replace_locals_(self.body, &self.arg_names.take().unwrap());
        self
    }

	pub(crate) fn name(&self) -> Option<&str> {
		self.name.as_ref().map(|x| x.as_str())
	}

	pub(crate) fn get_type(&self) -> FunctionType {
		self.t.clone()
	}

	pub(crate) fn call(&mut self, args: Vec<Arc<Mutex<Object>>>) -> Result<Value, RuntimeError> {
        let mut tree = vec![Ok(*self.body.clone())].into_iter();
        let mut globals = self.globals.clone().unwrap();
        let locals = self.locals.clone().unwrap();

        let mut exec = Executor::new(&mut tree, &mut globals)
            .locals(locals.clone());

        if let Some(names) = self.arg_names.clone() {
            for (obj, name) in std::iter::zip(args.clone().into_iter(), names.into_iter()) {
                exec = exec.add_local(name, obj);
            }
        }

        if let Some(name) = self.clone().name().map(|x| x.to_string()) {
            exec = exec.add_local(name, Arc::new(Mutex::new(Object::function(self.clone(), self.globals.clone().unwrap(), locals))));
        }

        exec.exec(self.body.clone())
	}
}

impl Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.t)
    }
}
