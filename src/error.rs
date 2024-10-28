
use std::error;
use std::fmt;
use std::ops::Range;

#[derive(Debug, Clone)]
pub struct Error {
    message: String,
    note: Option<String>,

    file: Option<String>,
    code: Option<String>,
    location: Option<(usize, Range<usize>)>,
}

impl Error {
    pub(crate) fn new(message: String) -> Self {
        Self {
            message,
            note: None,
            file: None,
            code: None,
            location: None
        }
    }

    pub(crate) fn note(mut self, note: String) -> Self {
        self.note = Some(note);
        self
    }

    pub(crate) fn file(mut self, file: String) -> Self {
        self.file = Some(file);
        self
    }

    pub(crate) fn location(mut self, line: usize, r: Range<usize>) -> Self {
        self.location = Some((line, r));
        self
    }

    pub(crate) fn code(mut self, code: String) -> Self {
        self.code = Some(code);
        self
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)?;

        if let Some((line, loc)) = &self.location {
            let filename = self.file.clone().unwrap_or("<unknown>".into());

            if let Some(code) = &self.code {
                let mut lines = code.lines();
                let linect = match lines.nth(*line - 1) {
                    Some(l) => l,
                    None => return Ok(()), // there should probably be an error if the line number is somehow out of range
                };

                let numspaces = " ".repeat((*line as f64).log10() as usize + 1);

                write!(f, "\n --> {filename}:{line}:{}\n", loc.start)?;
                write!(f, "{numspaces} |\n")?;
                write!(f, "{line} | {linect}\n")?;

                let spaces = " ".repeat(loc.start);
                let pointers: String = loc.clone().map(|_| '^').collect();

                write!(f, "{numspaces} |{spaces}{pointers}")?;

                if let Some(note) = &self.note {
                    write!(f, " {note}")?;
                }
            } else {
                write!(f, " @ {filename}:{line}:{}", loc.start)?;
            }
        }

        Ok(())
    }
}

impl error::Error for Error {}