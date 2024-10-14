use std::{error, io};
use std::collections::VecDeque;

use super::Value;
use std::fmt::{Display, Formatter};
use std::io::{BufRead, Cursor};

#[derive(Debug)]
pub enum TokenizeError {
    InvalidDynamicOperator(String),
    InvalidNumericConstant(String),
    InvalidIdentifier(String),
    UnableToMatchToken(String),
    IO(io::Error),
}

impl Display for TokenizeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenizeError::InvalidDynamicOperator(op)
                => write!(f, "invalid dynamic operator `{op}`"),
            TokenizeError::InvalidNumericConstant(t)
                => write!(f, "invalid numeric constant `{t}`"),
            TokenizeError::InvalidIdentifier(ident)
                => write!(f, "invalid identifier `{ident}`"),
            TokenizeError::UnableToMatchToken(token)
                => write!(f, "the token `{token}` was unable to be parsed"),
            TokenizeError::IO(io) => write!(f, "{io}")
        }
    }
}

impl error::Error for TokenizeError {}

#[derive(Debug, Clone)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Exp,
    Equ,
    Mod,
    LazyEqu,
    GlobalEqu,
    LazyGlobalEqu,
    FunctionDeclare(usize),
    Compose,
    Id,
    If,
    IfElse,
    GreaterThan,
    LessThan,
    EqualTo,
    GreaterThanOrEqualTo,
    LessThanOrEqualTo,
    Not,
    IntCast,
    FloatCast,
    BoolCast,
    StringCast,
}

#[derive(Debug, Clone)]
pub enum Token {
    Identifier(String),
    Operator(Op),
    Constant(Value),
}

fn get_dot_count(s: &str) -> Option<usize> {
    s.chars().fold(Some(0), |acc, c|
        match c {
            ':' => acc.map(|acc| acc + 2),
            '.' => acc.map(|acc| acc + 1),
            _ => None,
        }
    )
}

fn valid_identifier(c: char) -> bool {
    c.is_alphanumeric() || c == '\'' || c == '_'
}

impl Token {
    fn parse(s: &str) -> Result<Self, TokenizeError> {
        let string = regex::Regex::new(r#"".+""#).expect("LOL!");

        if string.is_match(s) {
            return Ok(Token::Constant(Value::String(s[1..s.len() - 1].to_string())));
        }

        match s {
            // First check if s is an operator
            "+"  => Ok(Token::Operator(Op::Add)),
            "-"  => Ok(Token::Operator(Op::Sub)),
            "*"  => Ok(Token::Operator(Op::Mul)),
            "/"  => Ok(Token::Operator(Op::Div)),
            "**" => Ok(Token::Operator(Op::Exp)),
            "%" => Ok(Token::Operator(Op::Mod)),
            "="  => Ok(Token::Operator(Op::Equ)),
            "."  => Ok(Token::Operator(Op::LazyEqu)),
            "=>" => Ok(Token::Operator(Op::GlobalEqu)),
            ".>" => Ok(Token::Operator(Op::LazyGlobalEqu)),
            "~"  => Ok(Token::Operator(Op::Compose)),
            "," => Ok(Token::Operator(Op::Id)),
            "?" => Ok(Token::Operator(Op::If)),
            "??" => Ok(Token::Operator(Op::IfElse)),
            ">" => Ok(Token::Operator(Op::GreaterThan)),
            "<" => Ok(Token::Operator(Op::LessThan)),
            ">=" => Ok(Token::Operator(Op::GreaterThanOrEqualTo)),
            "<=" => Ok(Token::Operator(Op::LessThanOrEqualTo)),
            "==" => Ok(Token::Operator(Op::EqualTo)),

            // then some keywords
            "true" => Ok(Token::Constant(Value::Bool(true))),
            "false" => Ok(Token::Constant(Value::Bool(false))),
            "not" => Ok(Token::Operator(Op::Not)),

            // Type casting
            "int" => Ok(Token::Operator(Op::IntCast)),
            "float" => Ok(Token::Operator(Op::FloatCast)),
            "bool" => Ok(Token::Operator(Op::BoolCast)),
            "string" => Ok(Token::Operator(Op::StringCast)),

            // then variable length keywords, constants, and identifiers
            _ => {
                if s.starts_with(':') {
                    Ok(Token::Operator(Op::FunctionDeclare(
                        get_dot_count(s).map(|x| x - 1).ok_or(TokenizeError::InvalidDynamicOperator(s.to_string()))?
                    )))
                } else if s.starts_with(|c| char::is_digit(c, 10) || c == '-') {
                    if let Ok(int) = s.parse::<i64>() {
                        Ok(Token::Constant(Value::Int(int)))
                    } else if let Ok(float) = s.parse::<f64>() {
                        Ok(Token::Constant(Value::Float(float)))
                    } else {
                        Err(TokenizeError::InvalidNumericConstant(s.to_string()))
                    }
                } else if s.starts_with(valid_identifier) {
                    let valid = s.chars().skip(1).all(valid_identifier);
                    valid.then(|| Token::Identifier(s.to_string())).ok_or(TokenizeError::InvalidIdentifier(s.to_string()))
                } else {
                    Err(TokenizeError::UnableToMatchToken(s.to_string()))
                }
            }
        }
    }
}

/// Tokenize an input stream of source code for a Parser
pub struct Tokenizer<R: BufRead> {
    reader: R,
    tokens: VecDeque<Token>,
}

impl<R: BufRead> Tokenizer<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            tokens: VecDeque::new(),
        }
    }
}

impl std::str::FromStr for Tokenizer<Cursor<String>> {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let cursor = Cursor::new(s.to_string());
        Ok(Tokenizer::new(cursor))
    }
}

impl<R: BufRead> std::iter::Iterator for Tokenizer<R> {
    type Item = Result<Token, TokenizeError>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(token) = self.tokens.pop_front() {
            return Some(Ok(token));
        }

        let mut input = String::new();

        match self.reader.read_to_string(&mut input) {
            Ok(0) => None,
            Err(e) => Some(Err(TokenizeError::IO(e))),
            _ => {
                let re = regex::Regex::new(r#"[a-zA-Z0-9\.'_]+|[`~!@#\$%\^&\*\(\)\+-=\[\]\{\}\\|;:,<\.>/\?]+|("[^"]+")"#).expect("This wont fail promise :3");

                for token in re.find_iter(input.as_str()).map(|mat| mat.as_str()).map(Token::parse) {
                    match token {
                        Ok(token) => self.tokens.push_back(token),
                        Err(e) => return Some(Err(e)),
                    }
                }

                self.tokens.pop_front().map(|x| Ok(x))
            }
        }
    }
}