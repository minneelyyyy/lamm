use std::io::{self, BufReader};

fn main() {
	let runtime = lamm::Runtime::new(BufReader::new(io::stdin()));

	for value in runtime.values() {
		match value {
			Ok(v) => println!("{v}"),
			Err(e) => eprintln!("{e}"),
		}
	}
}
