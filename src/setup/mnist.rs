
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
            sizes.push(r.read_i32::<BigEndian>()?); // number of columns
        }
        _ => return Err(io::Error::new(io::ErrorKind::Other, "unexpected value")),
    }
    r.read_to_end(&mut data)?;
    Ok(MnistData {
        sizes: sizes,
        data: data,
    })
}

#[derive(Debug, Clone)]
pub struct MnistImage {
    pub image: Array2<f64>,
    pub label: u8,
}

impl fmt::Display for MnistImage {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "image matrix: {}\nclassification: {}",
            self.image, self.label
        )
    }
}

/// Loading MNIST Data (<http://yann.lecun.com/exdb/mnist/>).
/// If the following files are not found in the .mnist directory of the execution path,
/// download them from the database and decode the data.
///
/// * train-images-idx3-ubyte.gz:  training set images (9912422 bytes)
/// * train-labels-idx1-ubyte.gz:  training set labels (28881 bytes)
/// * t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes)
/// * t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)
///
/// If they existed, read that file and decode the data.
///
/// # Arguments
///
/// * `dataset_key` - `train_dataset()` or `test_dataset()`.
/// * `normalize` - Flag that determines whether the image is normalized between 0.0 and 1.0.
pub fn load_data(dataset_key: DatasetKey, normalize: bool) -> io::Result<vec::Vec<MnistImage>> {
    const URL_BASE: &'static str = "http://yann.lecun.com/exdb/mnist/";
    const MNIST_SAVE_DIR: &'static str = ".mnist";

    static FILES: [RemoteFile; 4] = [
        RemoteFile {
            host_and_path: URL_BASE,
            fname: "train-images-idx3-ubyte.gz",
            sha256: "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
            query: "",
        },
        RemoteFile {
            host_and_path: URL_BASE,
            fname: "train-labels-idx1-ubyte.gz",
            sha256: "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
            query: "",
        },
        RemoteFile {
            host_and_path: URL_BASE,
            fname: "t10k-images-idx3-ubyte.gz",
            sha256: "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
            query: "",
        },
        RemoteFile {
            host_and_path: URL_BASE,
            fname: "t10k-labels-idx1-ubyte.gz",
            sha256: "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
            query: "",
        },
    ];

    let mnist = FetchClient::new(FConf::new(MNIST_SAVE_DIR, FILES.iter()))?;
    mnist.get()?;

    let task = |key_idx: KeyFile, mnistl: FetchClient| -> io::Result<MnistData> {
        println!(
            "start to decode {}...",
            FILES[key_idx.clone() as usize].fname
        );
        let ret = unarchive_mnist(&mnistl, FILES[key_idx.clone() as usize].fname);
        println!("Complete to decode {}", FILES[key_idx as usize].fname);
        ret
    };

    let mut rt = tokio::runtime::Runtime::new()?;
    let res: io::Result<(io::Result<MnistData>, io::Result<MnistData>)> = rt.block_on(async move {
        let label = dataset_key.label;
        let img = dataset_key.img;
        let mnistl = mnist.clone();
        let res_label = tokio::spawn(async move { task(label, mnist) }).await?;
        let res_data = tokio::spawn(async move { task(img, mnistl) }).await?;
        Ok((res_label, res_data))
    });
    let (label, images) = res?;
    let label_data = label?;
    let images_data = images?;

    let mut images: vec::Vec<Array2<f64>> = vec::Vec::new();
    let image_shape = (images_data.sizes[1] * images_data.sizes[2]) as usize;
    let compute_normalize: Box<dyn Fn(vec::Vec<u8>) -> vec::Vec<f64>> = if normalize {
        Box::new(|img_data: vec::Vec<u8>| -> vec::Vec<f64> {
            img_data.into_iter().map(|x| x as f64 / 255.).collect()
        })
    } else {
        Box::new(|img_data: vec::Vec<u8>| -> vec::Vec<f64> {
            img_data.into_iter().map(|x| x as f64).collect()
        })
    };

    let shape = (1, image_shape);
    for i in 0..images_data.sizes[0] as usize {
        let start = i * image_shape;
        let end = start + image_shape;
        let image_data = images_data.data[start..end].to_vec(); // 0~784, 784~(784 * 2), (784 * 2)~(784 * 3) ...
        images.push(Array2::from_shape_vec(shape, compute_normalize(image_data)).unwrap());
    }

    let mut ret: Vec<MnistImage> = vec::Vec::new();
    for (img, lbl) in images.into_iter().zip(label_data.data.into_iter()) {
        ret.push(MnistImage {
            image: img,
            label: lbl,
        })
    }
    Ok(ret)
}

#[derive(Debug)]
pub struct Batched {
    pub images: Array2<f64>,
    pub labels: vec::Vec<u8>,
}

impl Batched {
    pub fn new(images: Array2<f64>, labels: vec::Vec<u8>) -> Self {
        Batched {
            images: images,
            labels: labels,
        }
    }

    pub fn batch_size(&self) -> usize {
        self.labels.len()
    }
}

/// Make batche from `MnistImage`.
///
/// * `mnist_images` - mnist images
/// * `bsize` - Specify the batch size. Batch size must be non-zero and divisible by data size (10000)
pub fn batched(mnist_images: vec::Vec<MnistImage>, bsize: usize) -> io::Result<vec::Vec<Batched>> {
    if bsize == 0 {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "The batch size must be non-zero",
        ));
    } else if mnist_images.len() % bsize != 0 {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!(
                "The batch size ({}) must be divisible by data size ({})",
                bsize,
                mnist_images.len()
            ),
        ));
    }

    let mut res: vec::Vec<Batched> = vec![];
    let mut i = 0;

    while i < mnist_images.len() {
        let mut li = vec![];
        let labels = mnist_images[i..i + bsize]
            .iter()
            .map(|x: &MnistImage| -> u8 {
                li.push(x.image.view());
                x.label
            })
            .collect::<Vec<u8>>();
        res.push(Batched::new(
            to_io(stack(Axis(0), &li), io::ErrorKind::Other)?,
            labels,
        ));
        i += bsize;
    }
    Ok(res)
}