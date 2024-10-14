use std::io::{self, BufReader, Read};
use std::fs::File;
use std::env;

fn main() -> io::Result<()> {
	let file = env::args()
		.skip(1)
		.next()
		.map(|name| Box::new(File::open(name).expect("failed to open file")));

	let file: Box<dyn Read> = if file.is_none() {
		Box::new(io::stdin())
	} else {
		file.unwrap()
	};

	for value in lamm::evaluate(BufReader::new(file)) {
		match value {
			Ok(v) => println!("{v}"),
			Err(e) => eprintln!("{e}"),
		}
	}

	Ok(())
}
