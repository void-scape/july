#[derive(Debug)]
pub struct Source {
    source: String,
    origin: String,
}

impl Source {
    pub fn new(source: String, origin: String) -> Self {
        Self { source, origin }
    }

    pub fn as_str(&self) -> &str {
        &self.source
    }

    pub fn origin(&self) -> &str {
        &self.origin
    }
}
