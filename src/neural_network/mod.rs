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
    /// * `weight` - Weight matrix \\(W^{n_W\times m_W}\\) for computing next neuron.
    /// * `bias` - Bias matrix \\(B^{1\times m_B}\\) for computing next neuron.
    /// * `activate_function` - The activate function.
    #[inline]
    pub fn safe_next(
        &mut self,
        weight: &Array2<T>,
        bias: &Array2<T>,
        activate_function: &Box<dyn Fn(Array2<T>) -> Array2<T>>,
    ) -> Result<(), Error> {
        match (self.neurons.dim(), weight.dim(), bias.dim()) {
            ((_, width1), (height, width2), (_, width3))
                if width1 == height && width2 == width3 =>
            {
                Ok(self.next(weight, bias, activate_function))
            }
            _ => Err(failure::format_err!("Invalid argument")),
        }
    }

    /// Compute \\(h(X\cdot W+B)\\) where \\(X^{n_X\times m_X}\\) is a neurons matrix,
    /// \\(W^{n_W\times m_W\\) is a weights matrix,
    /// \\(B^{1\tims m_B}\\) is a bias matrix.
    /// These arguments must follow \\(m_X=n_W\\), \\(m_W=m_B\\).
    ///
    /// # Arguments
    ///
    /// * `weight` - Weight matrix \\(W^{n_W\times m_W\\) for computing next neuron.
    /// * `bias` - Bias matrix \\(B^{n_B\times m_B}\\) for computing next neuron.
    /// * `activate_function` - The activate_function.
    #[inline]
    pub fn next(
        &mut self,
        weight: &Array2<T>,
        bias: &Array2<T>,
        activate_function: &Box<dyn Fn(Array2<T>) -> Array2<T>>,
    ) {
        self.neurons = activate_function(self.neurons.dot(weight) + bias)
    }

    /// `dim` returns the shape of the array.
    #[inline]
    pub fn dim(&self) -> (ndarray::Ix, ndarray::Ix) {
        self.neurons.dim()
    }

    /// `argmax` returns the index of maximum value.
    /// 行毎の最大値
    #[inline]
    pub fn argmax(&self) -> Vec<usize> {
        self.neurons
            .outer_iter()
            .map(|x| x.argmax().unwrap())
            .collect::<Vec<usize>>()
    }
}
