
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
    data: vec::Vec<u8>,
}

fn unarchive_gz(client: &FetchClient, fname: &str) -> Result<vec::Vec<u8>, Error> {
    if client.dir_client.exists() && client.dir_client.file_exists(fname) {
        let src_path = client.dir_client.file_path(fname);
        let sf = BufReader::new(File::open(src_path).unwrap());
        let mut buf = vec![];
        {
            let mut df = BufWriter::new(&mut buf);
            let mut decoder = Decoder::new(sf).expect("failed to construct decoder");
            io::copy(&mut decoder, &mut df).expect("failed to copy file");
        }
        Ok(buf)
    } else {
        Err(failure::format_err!("no such file or directory"))
    }
}

fn unarchive_mnist(client: &FetchClient, fname: &str) -> io::Result<MnistData> {
    let s = to_io(unarchive_gz(&client, fname), io::ErrorKind::NotFound)?;
    let mut r = Cursor::new(&s);

    // Read the magic number indicating the data type.
    // If it is 2049, the data type is TRAINING SET LABEL FILE or TEST SET LABEL FILE.
    // It it is 2051, the data type is TRAINING SET IMAGE FILE or TEST SET IMAGE FILE.
    // See also: FILE FORMATS FOR THE MNIST DATABASE of http://yann.lecun.com/exdb/mnist/
    let magic_number = r.read_i32::<BigEndian>()?;

    let mut sizes: vec::Vec<i32> = vec::Vec::new();
    let mut data: vec::Vec<u8> = vec::Vec::new();

    match magic_number {
        2049 => sizes.push(r.read_i32::<BigEndian>()?), // number of items
        2051 => {
            sizes.push(r.read_i32::<BigEndian>()?); // number of images
            sizes.push(r.read_i32::<BigEndian>()?); // number of rows