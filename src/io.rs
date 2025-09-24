use std::fs::{create_dir_all, File};
use std::io::Write;

pub fn write_csv(path: &str, data: &[Vec<f64>]) -> std::io::Result<()> {
    if let Some(dir) = std::path::Path::new(path).parent() {
        if !dir.as_os_str().is_empty() { let _ = create_dir_all(dir); }
    }
    let mut f = File::create(path)?;
    if data.is_empty() { return Ok(()); }
    let n_el = data[0].len();
    // header
    for j in 0..n_el {
        if j > 0 { write!(f, ",")?; }
        write!(f, "e{}", j)?;
    }
    writeln!(f)?;
    for row in data {
        for j in 0..n_el {
            if j > 0 { write!(f, ",")?; }
            write!(f, "{}", row[j])?;
        }
        writeln!(f)?;
    }
    Ok(())
}
