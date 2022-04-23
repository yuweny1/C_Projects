extern crate deep_learning_playground;
extern crate ndarray;

use deep_learning_playground::neural_network;
use deep_learning_playground::neural_network::activate_functions;

fn main() {
    if let Ok(mut n) = neural_network::NeuralNetwork::<f64>::new(ndarray::array![[1.0, 0.5]]