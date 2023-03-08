
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
    /// `RemoteFile` constructor
    pub fn new(host_and_path: &'a str, fname: &'a str, sha256: &'a str, query: &'a str) -> Self {
        Self {
            host_and_path: host_and_path,
            fname: fname,
            sha256: sha256,
            query: query,
        }
    }
}

/// The structure to perform initialization setting of `FetchClient`
pub struct FConf<'a, T>
where
    T: Iterator<Item = &'a RemoteFile<'a>> + ExactSizeIterator,
{
    /// Directory name
    pub save_dir_name: &'a str,
    /// Information of the file to be fetched
    pub remote_file: T,
}

impl<'a, T> FConf<'a, T>
where
    T: Iterator<Item = &'a RemoteFile<'a>> + ExactSizeIterator,
{
    /// `FConf` constructor
    pub fn new(save_dir_name: &'a str, rf: T) -> Self {
        Self {
            save_dir_name: save_dir_name,
            remote_file: rf,
        }
    }
}

/// The `FileInfo`mation
#[derive(Clone)]
pub struct FileInfo<'a> {
    /// Host name and its path
    pub host_and_path: &'a str,
    /// sha256 value of the file
    pub sha256: &'a str,
    /// query used to get the file
    pub query: &'a str,
}

/// A type that executes the specified file existence check and creation,
/// path resolution, etc.
/// based on the save destination directory.
#[derive(Clone)]
pub struct DirClient<'a> {
    save_dir: PathBuf,
    /// Hash map with file name as key and related information as value
    pub file: HashMap<String, FileInfo<'a>>,
}