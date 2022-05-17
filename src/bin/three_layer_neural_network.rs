extern crate deep_learning_playground;
extern crate ndarray;

use deep_learning_playground::neural_network;
use deep_learning_playground::neural_network::activate_functions;

fn main() {
    if let Ok(mut n) = neural_network::NeuralNetwork::<f64>::new(ndarray::array![[1.0, 0.5]]) {
        let weights = [
            &ndarray::array![[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]],
            &ndarray::array![[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]],
            &ndarray::array![[0.1, 0.3], [0.2, 0.4]],
        ];
        let biases = [
            &ndarray::array![[0.1, 0.2, 0.3]],
            &ndarray::array![[0.1, 0.2]],
            &ndarray::array![[0.1, 0.2]],
        ];
       