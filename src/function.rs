use std::cell::RefCell;
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

#[derive(Clone, Debug, PartialEq)]
pub struct Function {
    pub(crate) name: Option<String>,
    t: FunctionType,
    arg_names: Vec<String>,
    body: Box<ParseTree>,
}

impl Function {
    pub(crate) fn lambda(t: FunctionType, arg_names: Vec<String>, body: Box<ParseTree>) -> Self {
        Self {
            name: None,
            t,
            arg_names,
            body
        }
    }

    pub(crate) fn named(name: &str, t: FunctionType, arg_names: Vec<String>, body: Box<ParseTree>) -> Self {
        Self {
            name: Some(name.to_string()),
            t,
            arg_names,
            body
        }
    }

	pub(crate) fn name(&self) -> Option<&str> {
		self.name.as_ref().map(|x| x.as_str())
	}

	pub(crate) fn get_type(&self) -> FunctionType {
		self.t.clone()
	}

	pub(crate) fn call(&mut self,
                mut globals: HashMap<String, Arc<Mutex<Object>>>,
                locals: HashMap<String, Arc<Mutex<Object>>>,
                args: Vec<Object>) -> Result<Value, RuntimeError>
    {
        let mut tree = vec![Ok(*self.body.clone())].into_iter();
        let g = globals.clone();

        let mut exec = Executor::new(&mut tree, &mut globals)
            .locals(locals.clone());

        for (obj, name) in std::iter::zip(args.into_iter(), self.arg_names.clone().into_iter()) {
            exec = exec.add_local(name.clone(), Arc::new(Mutex::new(obj)));
        }

        if let Some(name) = self.name().map(|x| x.to_string()) {
            exec = exec.add_local(name, Arc::new(Mutex::new(Object::function(self.clone(), g, locals))));
        }

        exec.next().unwrap()
	}
}

impl Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.t)
    }
}
