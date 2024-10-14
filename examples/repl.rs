
use lamm::{Tokenizer, Parser, Executor};
use std::io::{self, BufReader};

fn main() {
	let tokenizer = Tokenizer::new(BufReader::new(io::stdin()));
	let parser = Parser::new(tokenizer);
	let values = Executor::new(parser);

	for value in values {
		match value {
			Ok(v) => println!("{v}"),
			Err(e) => eprintln!("{e}"),
		}
	}
}