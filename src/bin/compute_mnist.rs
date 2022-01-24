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
      