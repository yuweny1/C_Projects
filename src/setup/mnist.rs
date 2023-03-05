
extern crate crypto;
extern crate libflate;
extern crate ndarray;
extern crate reqwest;
extern crate tokio;

use super::super::utils::fetch_client::{FConf, FetchClient, RemoteFile};
use super::super::utils::natural_transform::to_io;
use byteorder::{BigEndian, ReadBytesExt};