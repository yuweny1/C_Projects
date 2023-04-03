
use failure::Error;

/// `to_failure` is a natural transformation that convers `Result<T, E>` to
/// `Result<T, Error>`.
///
/// # Arguments
///
/// * `x` - `Result<T, E>`
pub fn to_failure<T, E: ToString>(x: Result<T, E>) -> Result<T, Error> {
    match x {
        Err(e) => Err(failure::format_err!("{}", e.to_string())),
        Ok(s) => Ok(s),
    }
}

/// `failure_to_io` is a natural transformation that convers `Result<T, E>` to
/// `std::io::Result<T>`.
///
/// # Arguments
///
/// * `x` - `Result<T, Error>`
pub fn to_io<T, E: ToString>(x: Result<T, E>, kind: std::io::ErrorKind) -> std::io::Result<T> {
    match x {
        Err(e) => Err(std::io::Error::new(kind, e.to_string())),
        Ok(s) => Ok(s),
    }
}

/// `opt_to_failure` is a natural transformation that convers `Option<...>` to
/// `Result<..., Error>`.
///
/// # Arguments
///
/// * `x` - `Option<T>`
/// * s - `&str`
pub fn opt_to_failure<T>(x: Option<T>, s: &str) -> Result<T, Error> {
    match x {
        Some(s) => Ok(s),
        None => Err(failure::format_err!("{}", s)),
    }
}