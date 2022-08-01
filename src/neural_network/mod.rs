use failure::Error;
use ndarray::Array2;
use ndarray_stats::QuantileExt;
use num::Float;
use std::fmt;

pub mod activate_functions;

#[derive(Default)]
pub struct NeuralNetwork<T> 