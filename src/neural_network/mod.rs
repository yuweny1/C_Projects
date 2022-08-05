use failure::Error;
use ndarray::Array2;
use ndarray_stats::QuantileExt;
use num::Float;
use std::fmt;

pub mod activate_functions;

#[derive(Default)]
pub struct NeuralNetwork<T> {
    neurons: Array2<T>,
}

impl<T: Float + fmt::Display> fmt::Display for NeuralNetwork<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.neurons)
    }
}

impl<T: Float + 'static> NeuralNetwork<T> {
    /// `new` is the constructor of `NeuralNetwork`.
    /// If the height of a