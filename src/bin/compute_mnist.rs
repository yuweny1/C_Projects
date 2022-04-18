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

    for td in data.into_iter() {
        accuracy_cnt += compute(td, &trained_data)?;
    }
    Ok((accuracy_cnt as f64 / len as f64, start_time.elapsed()))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut batch_size = 100;

    if args.len() != 1 {
        match args[1].parse::<usize>() {
            Err(e) => eprintln!("{}", e.to_string()),
            Ok(i) => batch_size = i,
        }
    }

    match execute(batch_size) {
        Err(e) => eprintln!("{}", e),
        Ok((s, pt)) => println!(
            "Accuracy: {}%, Process time: {}.{:03} seconds",
            s * 100.,
            pt.as_secs(),
            pt.subsec_nanos() / 1_000_000
        ),
    }
}
