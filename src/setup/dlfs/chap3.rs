use super::super::super::utils::fetch_client::{FConf, FetchClient, RemoteFile};
use super::super::super::utils::natural_transform::to_io;
use ndarray::Array2;
use numpy::{PyArray1, PyArray2};
use pyo3::prelu