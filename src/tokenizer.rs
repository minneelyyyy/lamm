use std::iter::Peekable;
use std::{error, io};
use std::collections::{VecDeque, HashMap};

use crate::Type;

use super::Value;
use std::fmt::{Display, Formatter};
use std::io::{BufRead, Cursor};

#[derive(Debug)]
pub enum TokenizeError {
    InvalidDynamicOperator(String),
    InvalidNumericConstant(String),
    InvalidIdentifier(String),
    UnableToMatchToken(String),
    InvalidCharacter(char),
    UnclosedString,
    IO(io::Error),
    Regex(regex::Error),
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
            TokenizeError::InvalidCharacter(c) => write!(f, "`{c}` is not understood"),
            TokenizeError::UnclosedString => write!(f, "newline was found before string was closed"),
            TokenizeError::IO(io) => write!(f, "{io}"),
            TokenizeError::Regex(re) => write!(f, "{re}"),
        }
    }
}

impl error::Error for TokenizeError {}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Op {
    Add,
    Sub,
    Mul,
    Div,
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
    Empty,
    And,
    Or,
    NonCall,
}

#[derive(Debug, Clone)]
pub(crate) enum Token {
    Identifier(String),
    Operator(Op),
    Constant(Value),
    Type(Type),
}

fn get_dot_count<I: Iterator<Item = char>>(s: &mut Peekable<I>) -> Option<usize> {
    let mut total = 0;

    while let Some(n) = s.next_if(|&c| c == ':' || c == '.').map(|c| match c {
        ':' => 2,
        '.' => 1,
        _ => 0,
    }) {
        total += n;
    }

    Some(total)
}

impl Token {
    /// Parse a single token
    fn parse(s: &str) -> Result<Self, TokenizeError> {
        let identifier = regex::Regex::new(r#"[A-Za-z_][A-Za-z0-9_']*"#).map_err(|e| TokenizeError::Regex(e))?;
        let number = regex::Regex::new(r#"([0-9]+\.?[0-9]*)|(\.[0-9])"#).map_err(|e| TokenizeError::Regex(e))?;

        match s {
            // Match keywords first
            "true"  => Ok(Token::Constant(Value::Bool(true))),
            "false" => Ok(Token::Constant(Value::Bool(false))),
            "int"    => Ok(Token::Operator(Op::IntCast)),
            "float"  => Ok(Token::Operator(Op::FloatCast)),
            "bool"   => Ok(Token::Operator(Op::BoolCast)),
            "string" => Ok(Token::Operator(Op::StringCast)),
            "print" => Ok(Token::Operator(Op::Print)),
            "empty" => Ok(Token::Operator(Op::Empty)),

            // Types
            "Any" => Ok(Token::Type(Type::Any)),
            "Int" => Ok(Token::Type(Type::Int)),
            "Float" => Ok(Token::Type(Type::Float)),
            "Bool" => Ok(Token::Type(Type::Bool)),
            "String" => Ok(Token::Type(Type::String)),

            // then identifiers and numbers
            _ => {
                if identifier.is_match(s) {
                    Ok(Token::Identifier(s.to_string()))
                } else if number.is_match(s) {
                    if let Ok(int) = s.parse::<i64>() {
                        Ok(Token::Constant(Value::Int(int)))
                    } else if let Ok(float) = s.parse::<f64>() {
                        Ok(Token::Constant(Value::Float(float)))
                    } else {
                        Err(TokenizeError::InvalidNumericConstant(s.to_string()))
                    }
                } else {
                    Err(TokenizeError::UnableToMatchToken(s.to_string()))
                }
            }
        }
    }
}

/// Tokenize an input stream of source code for a Parser
pub(crate) struct Tokenizer<R: BufRead> {
    reader: R,
    tokens: VecDeque<Result<Token, TokenizeError>>,
}

impl<R: BufRead> Tokenizer<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            tokens: VecDeque::new(),
        }
    }

    /// Tokenizes more input and adds them to the internal queue
    fn tokenize<I: Iterator<Item = char>>(&mut self, mut iter: Peekable<I>) {
        let operators: HashMap<&'static str, Op> = HashMap::from([
            ("+", Op::Add),
            ("-", Op::Sub),
            ("*", Op::Mul),
            ("/", Op::Div),
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
            ("!", Op::Not),
            ("&&", Op::And),
            ("||", Op::Or),
            ("'", Op::NonCall),
        ]);

        let c = if let Some(c) = iter.next() {
            c
        } else {
            return;
        };

        if c.is_alphanumeric() {
            let mut token = String::from(c);

            while let Some(c) = iter.next_if(|&c| c.is_alphanumeric() || c == '.' || c == '\'') {
                token.push(c);
            }

            self.tokens.push_back(Token::parse(&token));
            self.tokenize(iter)
        } else if c == '#' {
            let _: String = iter.by_ref().take_while(|&c| c != '\n').collect();
        } else if c == '\"' {
            let mut token = String::new();

            while let Some(c) = iter.next() {
                match c {
                    '"' => break,
                    '\n' => {
                        self.tokens.push_back(Err(TokenizeError::UnclosedString));
                        return;
                    }
                    '\\' => match iter.next() {
                        Some('\\') => token.push('\\'),
                        Some('n') => token.push('\n'),
                        Some('t') => token.push('\t'),
                        Some('r') => token.push('\r'),
                        Some('\"') => token.push('"'),
                        Some(c) => token.push(c),
                        None => {
                            self.tokens.push_back(Err(TokenizeError::UnclosedString));
                            return;
                        },
                    }
                    _ => token.push(c),
                }
            }

            self.tokens.push_back(Ok(Token::Constant(Value::String(token))));
            self.tokenize(iter)
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
                            self.tokens.push_back(Ok(Token::Operator(match op {
                                // special handling for "dynamic" operators
                                Op::FunctionDefine(n) => {
                                    let count = match get_dot_count(&mut iter) {
                                        Some(count) => count,
                                        None => {
                                            self.tokens.push_back(Err(TokenizeError::InvalidDynamicOperator(token)));
                                            return;
                                        }
                                    };
                                    Op::FunctionDefine(n + count)
                                }
                                Op::FunctionDeclare(n) => {
                                    let count = match get_dot_count(&mut iter) {
                                        Some(count) => count,
                                        None => {
                                            self.tokens.push_back(Err(TokenizeError::InvalidDynamicOperator(token)));
                                            return;
                                        }
                                    };
                                    Op::FunctionDeclare(n + count)
                                }
                                Op::LambdaDefine(n) => {
                                    let count = match get_dot_count(&mut iter) {
                                        Some(count) => count,
                                        None => {
                                            self.tokens.push_back(Err(TokenizeError::InvalidDynamicOperator(token)));
                                            return;
                                        }
                                    };
                                    Op::LambdaDefine(n + count)
                                }
                                op => op.clone(),
                            })));

                            break;
                        } else {
                            let next = match iter.next_if(is_expected) {
                                Some(c) => c,
                                None => {
                                    self.tokens.push_back(Err(TokenizeError::UnableToMatchToken(format!("{token}"))));
                                    return;
                                }
                            };
    
                            token.push(next);
                        }
                    }
                    0 => unreachable!(),
                    _ => {
                        let next = match iter.next_if(is_expected) {
                            Some(c) => c,
                            None => {
                                // at this point, token must be in the hashmap possible, otherwise it wouldnt have any matches
                                self.tokens.push_back(Ok(Token::Operator(match possible.get(token.as_str()).unwrap() {
                                    // special handling for "dynamic" operators
                                    Op::FunctionDefine(n) => {
                                        let count = match get_dot_count(&mut iter) {
                                            Some(count) => count,
                                            None => {
                                                self.tokens.push_back(Err(TokenizeError::InvalidDynamicOperator(token)));
                                                return;
                                            }
                                        };
                                        println!("{n} + {count}");
    
                                        Op::FunctionDefine(n + count)
                                    }
                                    Op::FunctionDeclare(n) => {
                                        let count = match get_dot_count(&mut iter) {
                                            Some(count) => count,
                                            None => {
                                                self.tokens.push_back(Err(TokenizeError::InvalidDynamicOperator(token)));
                                                return;
                                            }
                                        };
                                        Op::FunctionDeclare(n + count)
                                    }
                                    Op::LambdaDefine(n) => {
                                        let count = match get_dot_count(&mut iter) {
                                            Some(count) => count,
                                            None => {
                                                self.tokens.push_back(Err(TokenizeError::InvalidDynamicOperator(token)));
                                                return;
                                            }
                                        };
                                        Op::LambdaDefine(n + count)
                                    }
                                    op => op.clone(),
                                })));
                                break;
                            }
                        };

                        token.push(next);
                    }
                }
            }

            self.tokenize(iter)
        } else if c.is_whitespace() {
            self.tokenize(iter)
        } else {
            self.tokens.push_back(Err(TokenizeError::InvalidCharacter(c)));
            return;
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
            return Some(token);
        }

        let mut input = String::new();

        match self.reader.read_line(&mut input) {
            Ok(0) => None,
            Ok(_n) => {
                self.tokenize(input.chars().peekable());
                self.next()
            },
            Err(e) => Some(Err(TokenizeError::IO(e))),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use crate::parser::{Parser, ParseTree, ParseError};

    use super::*;

    #[test]
    fn uwu() {
        let program = ":. add x y + x y";

        let tokens: Vec<Token> = Tokenizer::from_str(program).unwrap().collect::<Result<_, TokenizeError>>().unwrap();

        println!("{tokens:?}");

        let trees: Result<Vec<ParseTree>, ParseError> = Parser::new(tokens.into_iter().map(|x| Ok(x))).collect();

        println!("{trees:?}");
    }
}