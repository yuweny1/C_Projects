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
    /// If the height of a given matrix is not 1, it means batch processing.
    ///
    /// # Arguments
    ///
    /// * `init_neurons` - The initial matrix \\(\mathbb{R}^{n\times m}\\).
    pub fn new(init_neurons: Array2<T>) -> Result<Self, Error> {
        if init_neurons.is_empty() {
            return Err(failure::format_err!("the matrix is empty"));
        }

        Ok(NeuralNetwork::<T> {
            neurons: init_neurons,
        })
    }

    /// Let a current matrix \\(X^{1\times m_X}\\),
    /// given arguments \\(W^{n_W\times m_W}\\) (weight) and \\(B^{1\times m_B}\\) (bias)
    /// where \\(m_X=n_W\\), \\(m_W=m_B\\).
    /// Thus, `next` computes next neurons \\(X W+B\\).
    /// If \\(m_X \not = n_W\\) or \\(m_W \not = m_B\\), it returns `Err`.
    ///
    /// # Arguments
    ///
    