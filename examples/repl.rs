use std::io::{self, BufReader};

fn main() {
	for value in lamm::evaluate(BufReader::new(io::stdin())) {
		match value {
			Ok(v) => println!("{v}"),
			Err(e) => eprintln!("{e}"),
		}
	}
}
