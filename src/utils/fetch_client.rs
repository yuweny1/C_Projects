
extern crate crypto;
extern crate libflate;
extern crate ndarray;
extern crate reqwest;
extern crate tokio;

use super::super::utils::natural_transform::opt_to_failure;
use bytes::Bytes;
use crypto::digest::Digest;
use crypto::sha2::Sha256;
use failure::Error;
use futures_util::stream::{self, StreamExt};
use std::collections::HashMap;
use std::env;
use std::fmt;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{self, PathBuf};

fn remove_ext(fname: &str) -> Result<PathBuf, Error> {
    Ok(PathBuf::from(opt_to_failure(
        opt_to_failure(
            PathBuf::from(fname).file_stem(),
            "Cannot dedude the name of unarchived file from archived file",
        )?
        .to_str(),
        "Cannot convert native string",
    )?))
}

/// Information about the file to get. Used when building `FConf`
#[derive(Debug)]
pub struct RemoteFile<'a> {
    /// Host name and its path
    pub host_and_path: &'a str,
    /// File name
    pub fname: &'a str,
    /// sha256 value of the file
    pub sha256: &'a str,
    /// Query used to get the file
    pub query: &'a str,
}

impl<'a> fmt::Display for RemoteFile<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "host_and_path: {}\n file name: {}\n sha256 hash: {}\n reqest query: {}",
            self.host_and_path, self.fname, self.sha256, self.query
        )
    }
}

impl<'a> RemoteFile<'a> {