use std::error::Error as StdError;
use std::fmt;
use std::result::Result as StdResult;

/// Custom error type for the application
#[derive(Debug)]
pub struct Error(Box<dyn StdError + Send + Sync>);

/// Custom result type for the application
pub type Result<T> = StdResult<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.0.source()
    }
}

impl From<String> for Error {
    fn from(err: String) -> Self {
        Error(Box::new(SimpleError(err)))
    }
}

impl From<&str> for Error {
    fn from(err: &str) -> Self {
        Error(Box::new(SimpleError(err.to_string())))
    }
}

impl<E> From<E> for Error
where
    E: StdError + Send + Sync + 'static,
{
    fn from(err: E) -> Self {
        Error(Box::new(err))
    }
}

/// Simple error implementation for string errors
#[derive(Debug)]
struct SimpleError(String);

impl fmt::Display for SimpleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl StdError for SimpleError {}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_from_string() {
        let err: Error = "test error".into();
        assert_eq!(err.to_string(), "test error");
    }
    
    #[test]
    fn test_result_ok() {
        let result: Result<i32> = Ok(42);
        assert_eq!(result.unwrap(), 42);
    }
    
    #[test]
    fn test_result_err() {
        let result: Result<i32> = Err("test error".into());
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().to_string(), "test error");
    }
}
