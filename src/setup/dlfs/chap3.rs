use super::super::super::utils::fetch_client::{FConf, FetchClient, RemoteFile};
use super::super::super::utils::natural_transform::to_io;
use ndarray::Array2;
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::{ObjectProtocol, PyResult, Python};
use pyo3::types::IntoPyDict;
use std::fmt;
use std::io;
use std::vec::Vec;

const URL_BASE: &'static str = "https://github.com/oreilly-japan/deep-learning-from-scratch/blob/0dda3d1715e2431b76eb4089b60881948853ba2a/ch03/";
const WEIGHT_SAVE_DIR: &'static str =