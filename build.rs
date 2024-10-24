use std::process::Command;

fn main() {
    // get
    let output = Command::new("git").args(&["rev-parse", "--short", "HEAD"]).output()
        .expect("git rev-parse HEAD failed");
    let hash = String::from_utf8(output.stdout).expect("not UTF-8 output");

    let output = Command::new("git").args(&["rev-parse", "--abbrev-ref", "HEAD"]).output()
        .expect("git rev-parse HEAD failed");
    let branch = String::from_utf8(output.stdout).expect("not UTF-8 output");

    println!("cargo:rustc-env=GIT_HASH={}", hash);
    println!("cargo:rustc-env=GIT_BRANCH={}", branch);
}
