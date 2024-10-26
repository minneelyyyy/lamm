use std::iter::Peekable;
use std::{error, io};
use std::collections::{VecDeque, HashMap};

use crate::Type;

use super::Value;
use std::fmt::{Display, Formatter};
use std::io::BufRead;
use std::sync::Arc;
use std::ops::Range;

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
    fn parse(s: &str) -> Result<Self, TokenizeError> {
        let identifier = regex::Regex::new(r#"[A-Za-z_][A-Za-z0-9_']*"#).map_err(|e| TokenizeError::Regex(e))?;
        let number = regex::Regex::new(r#"([0-9]+\.?[0-9]*)|(\.[0-9])"#).map_err(|e| TokenizeError::Regex(e))?;

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
                if identifier.is_match(s) {
                    TokenType::Identifier(s.to_string())
                } else if number.is_match(s) {
                    if let Ok(int) = s.parse::<i64>() {
                        TokenType::Constant(Value::Int(int))
                    } else if let Ok(float) = s.parse::<f64>() {
                        TokenType::Constant(Value::Float(float))
                    } else {
                        return Err(TokenizeError::InvalidNumericConstant(s.to_string()));
                    }
                } else {
                    return Err(TokenizeError::UnableToMatchToken(s.to_string()));
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
    pub file: Arc<String>,
    pub location: Range<usize>,
}

impl Token {
    pub fn new(t: TokenType, lexeme: String, file: Arc<String>, line: usize, column: usize) -> Self {
        Self {
            t,
            line,
            file,
            location: column..column+lexeme.len(),
            lexeme,
        }
    }

    pub fn token(&self) -> TokenType {
        self.t.clone()
    }
}

/// Tokenize an input stream of source code for a Parser
pub(crate) struct Tokenizer<R: BufRead> {
    reader: R,
    line: usize,
    column: usize,
    code: String,
    filename: Arc<String>,
    tokens: VecDeque<Token>,
}

impl<R: BufRead> Tokenizer<R> {
    pub fn new(reader: R, filename: &str) -> Self {
        Self {
            reader,
            line: 0,
            column: 0,
            filename: Arc::new(filename.to_string()),
            code: String::new(),
            tokens: VecDeque::new(),
        }
    }

    fn get_dot_count<I: Iterator<Item = char>>(&mut self, s: &mut Peekable<I>) -> Option<usize> {
        let mut total = 0;

        while let Some(n) = self.next_char_if(s, |&c| c == ':' || c == '.').map(|c| match c {
            ':' => 2,
            '.' => 1,
            _ => 0,
        }) {
            total += n;
        }

        Some(total)
    }

    fn next_char<I: Iterator<Item = char>>(&mut self, iter: &mut Peekable<I>) -> Option<char> {
        if let Some(c) = iter.next() {
            self.column += 1;
            Some(c)
        } else {
            None
        }
    }

    fn next_char_if<I: Iterator<Item = char>>(
        &mut self,
        iter: &mut Peekable<I>,
        pred: impl FnOnce(&char) -> bool) -> Option<char>
    {
        if let Some(c) = iter.next_if(pred) {
            self.column += 1;
            Some(c)
        } else {
            None
        }
    }

    fn next_char_while<I: Iterator<Item = char>>(
        &mut self,
        iter: &mut Peekable<I>,
        mut pred: impl FnMut(&char) -> bool) -> Option<char>
    {
        if let Some(c) = self.next_char(iter) {
            if (pred)(&c) {
                Some(c)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Tokenizes more input and adds them to the internal queue
    fn tokenize<I: Iterator<Item = char>>(&mut self, mut iter: Peekable<I>) -> Result<(), TokenizeError> {
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

        let c = if let Some(c) = self.next_char(&mut iter) {
            c
        } else {
            return Ok(());
        };

        if c.is_alphanumeric() {
            let mut token = String::from(c);

            while let Some(c) = self.next_char_if(&mut iter, |&c| c.is_alphanumeric() || c == '.' || c == '\'') {
                token.push(c);
            }

            self.tokens.push_back(Token::new(TokenType::parse(&token)?, token, self.filename.clone(), self.line, self.column));
            self.tokenize(iter)
        } else if c == '#' {
            let _: String = iter.by_ref().take_while(|&c| c != '\n').collect();
            self.tokenize(iter)
        } else if c == '\"' {
            let mut token = String::new();

            while let Some(c) = self.next_char(&mut iter) {
                match c {
                    '"' => break,
                    '\n' => return Err(TokenizeError::UnclosedString),
                    '\\' => match iter.next() {
                        Some('\\') => token.push('\\'),
                        Some('n') => token.push('\n'),
                        Some('t') => token.push('\t'),
                        Some('r') => token.push('\r'),
                        Some('\"') => token.push('"'),
                        Some(c) => token.push(c),
                        None => return Err(TokenizeError::UnclosedString),
                    }
                    _ => token.push(c),
                }
            }

            self.tokens.push_back(
                Token::new(TokenType::Constant(
                    Value::String(token.clone())), token, self.filename.clone(), self.line, self.column));

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
                            let token = Token::new(TokenType::Operator(match op {
                                // special handling for "dynamic" operators
                                Op::FunctionDefine(n) => {
                                    let count = match self.get_dot_count(&mut iter) {
                                        Some(count) => count,
                                        None => return Err(TokenizeError::InvalidDynamicOperator(token)),
                                    };
                                    Op::FunctionDefine(n + count)
                                }
                                Op::FunctionDeclare(n) => {
                                    let count = match self.get_dot_count(&mut iter) {
                                        Some(count) => count,
                                        None => return Err(TokenizeError::InvalidDynamicOperator(token)),
                                    };
                                    Op::FunctionDeclare(n + count)
                                }
                                Op::LambdaDefine(n) => {
                                    let count = match self.get_dot_count(&mut iter) {
                                        Some(count) => count,
                                        None => return Err(TokenizeError::InvalidDynamicOperator(token)),
                                    };
                                    Op::LambdaDefine(n + count)
                                }
                                op => op.clone(),
                            }), token, self.filename.clone(), self.line, self.column);
                            
                            self.tokens.push_back(token);

                            break;
                        } else {
                            let next = match self.next_char_if(&mut iter, is_expected) {
                                Some(c) => c,
                                None => return Err(TokenizeError::UnableToMatchToken(format!("{token}"))),
                            };
    
                            token.push(next);
                        }
                    }
                    0 => unreachable!(),
                    _ => {
                        let next = match self.next_char_if(&mut iter, is_expected) {
                            Some(c) => c,
                            None => {
                                let token = Token::new(TokenType::Operator(match possible.get(token.as_str()).unwrap() {
                                    // special handling for "dynamic" operators
                                    Op::FunctionDefine(n) => {
                                        let count = match self.get_dot_count(&mut iter) {
                                            Some(count) => count,
                                            None => return Err(TokenizeError::InvalidDynamicOperator(token)),
                                        };
    
                                        Op::FunctionDefine(n + count)
                                    }
                                    Op::FunctionDeclare(n) => {
                                        let count = match self.get_dot_count(&mut iter) {
                                            Some(count) => count,
                                            None => return Err(TokenizeError::InvalidDynamicOperator(token)),
                                        };
                                        Op::FunctionDeclare(n + count)
                                    }
                                    Op::LambdaDefine(n) => {
                                        let count = match self.get_dot_count(&mut iter) {
                                            Some(count) => count,
                                            None => return Err(TokenizeError::InvalidDynamicOperator(token)),
                                        };
                                        Op::LambdaDefine(n + count)
                                    }
                                    op => op.clone(),
                                }), token, self.filename.clone(), self.line, self.column);

                                // at this point, token must be in the hashmap possible, otherwise it wouldn't have any matches
                                self.tokens.push_back(token);
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
            return Err(TokenizeError::InvalidCharacter(c));
        }
    }
}

impl<R: BufRead> Iterator for Tokenizer<R> {
    type Item = Result<Token, TokenizeError>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(token) = self.tokens.pop_front() {
            return Some(Ok(token));
        }

        let mut input = String::new();

        match self.reader.read_line(&mut input) {
            Ok(0) => None,
            Ok(_n) => {
                self.code.push_str(&input);
                self.line += 1;
                self.column = 0;

                match self.tokenize(input.chars().peekable()) {
                    Ok(()) => (),
                    Err(e) => return Some(Err(e)),
                }

                self.next()
            },
            Err(e) => Some(Err(TokenizeError::IO(e))),
        }
    }
}

#[cfg(test)]
mod tests {
    use io::Cursor;

    use crate::parser::Parser;
    use super::*;

    #[test]
    fn tokenizer() {
        let program = ": length ?. x [] -> Int ?? x + 1 length tail x 0 length [ 1 2 3 ]";

        let tokens: Vec<Token> = Tokenizer::new(Cursor::new(program), "<tokenizer>").collect::<Result<_, _>>().unwrap();

        println!("{tokens:#?}");
    }

    #[test]
    fn a() {
        let program = ": length ?. x [] -> Int ?? x + 1 length tail x 0 length [ 1 2 3 ]";

        let mut tokenizer = Tokenizer::new(Cursor::new(program), "<a>").peekable();

        let mut globals = HashMap::new();
        let mut parser = Parser::new(&mut tokenizer, &mut globals);

        let tree = parser.next();
        println!("{tree:#?}");
    }
}