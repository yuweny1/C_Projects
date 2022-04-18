
extern crate deep_learning_playground;
extern crate image;
extern crate ndarray;

use deep_learning_playground::setup::mnist::{load_data, train_dataset};
use image::ImageBuffer;
use ndarray::Array2;
use num::integer::Roots;

const OUT_FILE_NAME: &'static str = "out.png";

fn to_image(m: &Array2<f64>, out_fname: &str) {
    let (h, w) = m.dim();
    let mut imgbuf = ImageBuffer::new(h as u32, w as u32);
    for (e, (_, _, pixel)) in m.iter().zip(imgbuf.enumerate_pixels_mut()) {
        *pixel = image::Luma([*e as u8]);
    }
    imgbuf.save(out_fname).unwrap();
}

fn main() {
    match load_data(train_dataset(), false) {
        Err(e) => eprintln!("{}", e),
        Ok(s) => {
            let (h, _) = s[0].image.dim();
            let size = h.sqrt() as usize;
            match s[0].image.clone().into_shape((size, size)) {
                Ok(ss) => {
                    println!("saving png file to {}...", OUT_FILE_NAME);
                    to_image(&ss, OUT_FILE_NAME)
                }
                Err(e) => eprintln!("{}", e),
            }
        }
    }
}