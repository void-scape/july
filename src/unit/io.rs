use std::io::{self, BufReader, Read};
use std::path::Path;

/// Atomically writes `contents` to `path`.
pub fn write<P: AsRef<Path>>(path: P, contents: &[u8]) -> io::Result<()> {
    let path = path.as_ref();
    let tmp_path = path.with_extension("tmp");

    std::fs::write(&tmp_path, contents)?;
    std::fs::rename(tmp_path, path)?;

    Ok(())
}

pub fn read<P: AsRef<Path>>(path: P) -> io::Result<Vec<u8>> {
    let path = path.as_ref();
    let file = std::fs::File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut contents = Vec::new();
    reader.read(&mut contents)?;
    Ok(contents)
}

pub fn read_string<P: AsRef<Path>>(path: P) -> io::Result<String> {
    let path = path.as_ref();
    let file = std::fs::File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut contents = String::new();
    reader.read_to_string(&mut contents)?;
    Ok(contents)
}
