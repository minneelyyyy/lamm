use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::{CodeIter, Type};
use crate::error::Error;

use super::Value;
use std::io::BufRead;
use std::ops::Range;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    Exp,
    Equ,
    Mod,
    LazyEqu,
    TypeDeclaration,
    FunctionDefine(usize),
    FunctionDeclare(usize),
    LambdaDefine(usize),
    Arrow,
    Compose,
    Id,
    If,
    IfElse,
    GreaterThan,
    LessThan,
    EqualTo,
    NotEqualTo,
    GreaterThanOrEqualTo,
    LessThanOrEqualTo,
    Not,
    IntCast,
    FloatCast,
    BoolCast,
    StringCast,
    Print,
    OpenArray,
    CloseArray,
    OpenStatement,
    CloseStatement,
    Empty,
    And,
    Or,
    Head,
    Tail,
    Init,
    Fini,
    Export,
    NonCall,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TokenType {
    Identifier(String),
    Operator(Op),
    Constant(Value),
    Type(Type),
}

impl TokenType {
    /// Parse a single token
    fn parse(s: &str) -> Result<Self, Error> {
        Ok(match s {
            // Match keywords first
            "true"  => TokenType::Constant(Value::Bool(true)),
            "false" => TokenType::Constant(Value::Bool(false)),
            "nil" => TokenType::Constant(Value::Nil),
            "int"    => TokenType::Operator(Op::IntCast),
            "float"  => TokenType::Operator(Op::FloatCast),
            "bool"   => TokenType::Operator(Op::BoolCast),
            "string" => TokenType::Operator(Op::StringCast),
            "print" => TokenType::Operator(Op::Print),
            "empty" => TokenType::Operator(Op::Empty),
            "head" => TokenType::Operator(Op::Head),
            "tail" => TokenType::Operator(Op::Tail),
            "init" => TokenType::Operator(Op::Init),
            "fini" => TokenType::Operator(Op::Fini),
            "export" => TokenType::Operator(Op::Export),

            // Types
            "Any" => TokenType::Type(Type::Any),
            "Int" => TokenType::Type(Type::Int),
            "Float" => TokenType::Type(Type::Float),
            "Bool" => TokenType::Type(Type::Bool),
            "String" => TokenType::Type(Type::String),

            // then identifiers and numbers
            _ => {
                if s.starts_with(char::is_alphabetic) {
                    TokenType::Identifier(s.to_string())
                } else if s.starts_with(|c: char| c.is_digit(10)) {
                    if let Ok(int) = s.parse::<i64>() {
                        TokenType::Constant(Value::Int(int))
                    } else if let Ok(float) = s.parse::<f64>() {
                        TokenType::Constant(Value::Float(float))
                    } else {
                        return Err(Error::new(format!("Invalid numeric constant `{s}`")));
                    }
                } else {
                    return Err(Error::new(format!("Couldn't match token `{s}`")));
                }
            }
        })
    }
}

#[derive(Debug, Clone)]
pub struct Token {
    t: TokenType,
    pub lexeme: String,
    pub line: usize,
    pub location: Range<usize>,
}

impl Token {
    pub fn new(t: TokenType, lexeme: String, line: usize, column: usize) -> Self {
        Self {
            t,
            line,
            location: column..column + lexeme.len(),
            lexeme,
        }
    }

    pub fn token(&self) -> TokenType {
        self.t.clone()
    }
}

/// Tokenize an input stream of source code for a Parser
#[derive(Clone)]
pub(crate) struct Tokenizer<R: BufRead> {
    reader: Arc<Mutex<CodeIter<R>>>,
}

impl<R: BufRead> Tokenizer<R> {
    pub fn new(reader: Arc<Mutex<CodeIter<R>>>) -> Self {
        Self {
            reader,
        }
    }

    fn next_char(&mut self) -> Option<char> {
        let mut reader = self.reader.lock().unwrap();
        let c = reader.next();
        c
    }

    fn next_char_if(&mut self, func: impl FnOnce(&char) -> bool) -> Option<char> {
        let mut reader = self.reader.lock().unwrap();
        let c = reader.next_if(func);
        c
    }

    fn getpos(&self) -> (usize, usize) {
        let reader = self.reader.lock().unwrap();
        let r = reader.getpos();
        r
    }

    fn get_dot_count(&mut self) -> usize {
        let mut total = 0;

        while let Some(n) = self.next_char_if(|&c| c == ':' || c == '.').map(|c| match c {
            ':' => 2,
            '.' => 1,
            _ => unreachable!(),
        }) {
            total += n;
        }

        total
    }

    /// Tokenizes more input and adds them to the internal queue
    fn tokenize(&mut self) -> Result<Option<Token>, Error> {
        let operators: HashMap<&'static str, Op> = HashMap::from([
            ("+", Op::Add),
            ("-", Op::Sub),
            ("*", Op::Mul),
            ("/", Op::Div),
            ("//", Op::FloorDiv),
            ("**", Op::Exp),
            ("%", Op::Mod),
            ("=", Op::Equ),
            (".", Op::LazyEqu),
            ("?.", Op::TypeDeclaration),
            (":", Op::FunctionDefine(1)),
            ("?:", Op::FunctionDeclare(1)),
            (";", Op::LambdaDefine(1)),
            ("->", Op::Arrow),
            ("~", Op::Compose),
            (",", Op::Id),
            ("?", Op::If),
            ("??", Op::IfElse),
            (">", Op::GreaterThan),
            ("<", Op::LessThan),
            (">=", Op::GreaterThanOrEqualTo),
            ("<=", Op::LessThanOrEqualTo),
            ("==", Op::EqualTo),
            ("!=", Op::NotEqualTo),
            ("[", Op::OpenArray),
            ("]", Op::CloseArray),
            ("(", Op::OpenStatement),
            (")", Op::CloseStatement),
            ("!", Op::Not),
            ("&&", Op::And),
            ("||", Op::Or),
            ("\\", Op::NonCall),
        ]);

        let c = if let Some(c) = self.next_char() {
            c
        } else {
            return Ok(None);
        };

        if c.is_alphanumeric() {
            let mut token = String::from(c);

            while let Some(c) = self.next_char_if(|&c| c.is_alphanumeric() || c == '.' || c == '\'') {
                token.push(c);
            }

            let (line, column) = self.getpos();

            Ok(Some(Token::new(TokenType::parse(&token).map_err(|e| e.location(line, column - token.len() + 1..column + 1))?, token.clone(), line, column - token.len() + 1)))
        } else if c == '#' {
            while self.next_char_if(|&c| c != '\n').is_some() {}
            self.tokenize()
        } else if c == '\"' {
            let mut token = String::new();
            let (line, col) = self.getpos();

            while let Some(c) = self.next_char() {
                match c {
                    '"' => {
                        let (line, col) = self.getpos();

                        return Ok(Some(Token::new(TokenType::Constant(
                            Value::String(token.clone())), token, line, col)));
                    }
                    '\n' => return Err(
                        Error::new("Unclosed string literal".into())
                            .location(line, col..col+token.len()+1)
                            .note("newlines are not allowed in string literals (try \\n)".into())),
                    '\\' => match self.next_char() {
                        Some('\\') => token.push('\\'),
                        Some('n') => token.push('\n'),
                        Some('t') => token.push('\t'),
                        Some('r') => token.push('\r'),
                        Some('\"') => token.push('"'),
                        Some(c) => token.push(c),
                        None => return Err(
                            Error::new("Unclosed string literal".into())
                                .location(line, col..token.len()+1)
                                .note("end of file found before \"".into())),
                    }
                    _ => token.push(c),
                };
            }

            Err(Error::new("Unclosed string literal".into())
                .location(line, col..self.getpos().1+1)
                .note("end of file found before \"".into()))
        } else if operators.keys().any(|x| x.starts_with(c)) {
            let mut token = String::from(c);

            loop {
                // get a list of all tokens this current token could possibly be
                let possible: HashMap<&'static str, Op> = operators
                    .clone().into_iter()
                    .filter(|(key, _)| key.starts_with(&token))
                    .collect();

                // checks if a character is "expected", aka based on how many chars
                // we have eaten so far, which characters out of the current nominees
                // are expected in the next position
                let is_expected = |c: &char|
                    possible.iter().any(|(op, _)| match op.chars().nth(token.len()) {
                        Some(i) => *c == i,
                        None => false,
                    });

                match possible.len() {
                    1 => {
                        // if the current operator exists in possible, we push it
                        // if not, we need to make sure that the next characters
                        // we grab *actually* match the last operator
                        if let Some(op) = possible.get(token.as_str()) {
                            let t = TokenType::Operator(match op {
                                // special handling for "dynamic" operators
                                Op::FunctionDefine(n) => Op::FunctionDefine(n + self.get_dot_count()),
                                Op::FunctionDeclare(n) => Op::FunctionDeclare(n + self.get_dot_count()),
                                Op::LambdaDefine(n) => Op::LambdaDefine(n + self.get_dot_count()),
                                op => op.clone(),
                            });

                            let (line, col) = self.getpos();

                            let token = Token::new(t, token, line, col);
                            
                            return Ok(Some(token));
                        } else {
                            let next = match self.next_char_if(is_expected) {
                                Some(c) => c,
                                None => {
                                    let (line, col) = self.getpos();

                                    return Err(
                                        Error::new(format!("the operator {token} is undefined"))
                                            .location(line, col - token.len()..col))
                                }
                            };
    
                            token.push(next);
                        }
                    }
                    0 => unreachable!(),
                    _ => {
                        let c = self.next_char_if(is_expected);
                        let next = match c {
                            Some(c) => c,
                            None => {
                                let t = TokenType::Operator(match possible.get(token.as_str()).unwrap() {
                                    // special handling for "dynamic" operators
                                    Op::FunctionDefine(n) => Op::FunctionDefine(n + self.get_dot_count()),
                                    Op::FunctionDeclare(n) => Op::FunctionDeclare(n + self.get_dot_count()),
                                    Op::LambdaDefine(n) => Op::LambdaDefine(n + self.get_dot_count()),
                                    op => op.clone(),
                                });
    
                                let (line, col) = self.getpos();
    
                                let token = Token::new(t, token, line, col);

                                return Ok(Some(token))
                            }
                        };

                        token.push(next);
                    }
                }
            }
        } else if c.is_whitespace() {
            self.tokenize()
        } else {
            let (line, col) = self.getpos();

            Err(Error::new(format!("an unidentified character {c} was found"))
                    .location(line, col..col+1))
        }
    }
}

impl<R: BufRead> Iterator for Tokenizer<R> {
    type Item = Result<Token, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        self.tokenize().transpose()
    }
}