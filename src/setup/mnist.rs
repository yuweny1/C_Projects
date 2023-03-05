
extern crate crypto;
extern crate libflate;
extern crate ndarray;
extern crate reqwest;
extern crate tokio;

use super::super::utils::fetch_client::{FConf, FetchClient, RemoteFile};
use super::super::utils::natural_transform::to_io;
use byteorder::{BigEndian, ReadBytesExt};
use failure::Error;
use libflate::gzip::Decoder;
use ndarray::{stack, Array2, Axis};
use std::fmt;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Cursor, Read};
use std::vec;

#[repr(usize)]
#[derive(Clone)]
enum KeyFile {
    TrainImg,
    TrainLabel,
    TestImg,
    TestLabel,
}

#[derive(Clone)]
pub struct DatasetKey {
    img: KeyFile,
    label: KeyFile,
}

pub fn train_dataset() -> DatasetKey {
    DatasetKey {
        img: KeyFile::TrainImg,
        label: KeyFile::TrainLabel,
    }
}

pub fn test_dataset() -> DatasetKey {
    DatasetKey {
        img: KeyFile::TestImg,
        label: KeyFile::TestLabel,
    }
}

struct MnistData {
    sizes: vec::Vec<i32>,