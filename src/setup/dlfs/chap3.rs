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
const WEIGHT_SAVE_DIR: &'static str = ".weight_data";
const FILE_NAME: &'static str = "sample_weight.pkl";
const WEIGHT_NAMES: [&'static str; 3] = ["W1", "W2", "W3"];
const BIAS_NAMES: [&'static str; 3] = ["b1", "b2", "b3"];

/// `Chap3Param` is data structure that has weight matrixes and bias matrixes.
#[derive(Debug)]
pub struct Chap3Param {
    /// weight matrixes
    pub weight: Vec<Array2<f32>>,
    /// bias matrixes
    pub bias: Vec<Array2<f32>>,
}

impl fmt::Display for Chap3Param {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "weight: {:?}\n\nbias: {:?}", self.weight, self.bias)
    }
}

fn deserialize<'py>(
    py: Python<'py>,
    client: &FetchClient,
    fname: &str,
) -> PyResult<io::Result<Chap3Param>> {
    if client.dir_client.exists() && client.dir_client.file_exists(fname) {
        if let Some(path) = client.dir_client.file_path(fname).to_str() {
            let locals = [
                ("io", py.import("io")?),
                ("numpy", py.import("numpy")?),
                ("pickle", py.import("pickle")?),
            ]
            .into_py_dict(py);

            let mut weight: Vec<Array2<f32>> = vec![];
            let mut bias: Vec<Array2<f32>> = vec![];

            for w in WEIGHT_NAMES.iter() {
                let code = "pickle.load(io.open('".to_owned() + path + "','rb'))['" + w + "']";
                let pyarray: &PyArray2<f32> = py.eval(&code, None, Some(&locals))?.extract()?;
                weight.push(pyarray.as_array().to_owned());
            }

            for b in BIAS_NAMES.iter() {
                let code = "pickle.load(io.open('".to_owned() + path + "','rb'))['" + b + "']";
                let pyarray: &PyArray1<f32> = py.eval(&code, None, Some(&locals))?.extract()?;
                let ar = pyarray.as_array().to_owned();
                let len = ar.dim();
                bias.push(to_io(ar.into_shape((1, len)), io::ErrorKind::Other)?);
            }

            return Ok(Ok(Chap3Param { weight, bias }));
        }
    }

    Ok(Err(io::Error::new(
        io::ErrorKind::NotFound,
        "no such file or directory",
    )))
}

/// `load_trained_params` loads trained parameters from pickle
/// ([oreilly-japan/deep-learning-from-scratch/ch03/sample_weight.pkl](https://github.com/oreilly-japan/deep-learning-from-scratch/blob/0dda3d1715e2431b76eb4089b60881948853ba2a/ch03/sample_weight.pkl)).
/// It requires some python packages. E.g. python3-dev, python-dev (On Ubuntu 18.04)
/// and `numpy`.
pub fn load_trained_params() -> io::Result<Chap3Param> {
    let file = [RemoteFile::new(
        URL_BASE,
        FILE_NAME,
        "b7f55a27988ba34c3777b0f1bbd464817c8a4db855723e6ff26f703501917a13",
        "raw=true",
    )];

    let client = FetchClient::new(FConf::new(WEIGHT_SAVE_DIR, file.iter()))?;
    client.get()?;

    let gil = Python::acquire_gil();
    deserialize(gil.python(), &client, FILE_NAME).map_err(|e| {
        io::Error::new(
            io::ErrorKind::Other,
            format!("Python interpreter error: {:?}", e),
        )
    })?
}
