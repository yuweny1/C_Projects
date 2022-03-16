use deep_learning_playground::neural_network::{self, activate_functions};
use deep_learning_playground::setup::dlfs::chap3;
use deep_learning_playground::setup::mnist::{batched, load_data, test_dataset, Batched};
use deep_learning_playground::utils::natural_transform::to_io;
use std::env;
use std::io;
use std::time::{Duration, Instant};
use std::vec::Vec;

fn compute(td: Batched, trained_data: &chap3::Chap3Param) -> io::Result<u32> {
    let mut nn = to_io(
        neural_network::NeuralNetwork::<f64>::new(td.images),
        io::ErrorKind::Other,
    )?;
    let mut afunc = vec![];
    for _ in 1..trained_data.bias.len() {
        afunc.push(activate_functions::sigmoid());
    }
    afunc.push(activate_functions::softmax());

    for ((w, b), af) in trained_data
        .weight
        .iter()
        .zip(trained_data.bias.iter())
        .zip(afunc.iter())
    {
        nn.next(&w.map(|x| *x as f64), &b.map(|x| *x as f64), &af);
    }

    Ok(td
        .labels
        .iter()
        .zip(nn.argmax().iter())
        .map(|(l, r)| (*l as u32 == *r as u32) as u32)
        .sum())
}

fn execute(bsize: usize) -> io::Result<(f64, Duration)> {
    let data = load_data(test_dataset(), true)?;
    let len = data.len();
    let data = batched(data, bsize)?;
    let trained_data = chap3::load_trained_params()?;
        
    println!("Loading success:\n\t* MNIST dataset (size: {})\n\t* trained params", len);

    let mut accuracy_cnt: u32 = 0;

    println!("Start computing...");
    let start_time = Instant::now();

    for td in data.into_i