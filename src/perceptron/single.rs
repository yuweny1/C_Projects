
extern crate num;

use num::Float;

/// `step_function` is the kind of activation functions. That is
/// \\[
/// \text{step}(x) =
/// \begin{cases}
/// 1 & (x\gt 0) \\\\
/// 0 & (\text{otherwise})
/// \end{cases}
/// \\]
/// where \\(x\in\mathbb{R}\\).
///
/// # e.g.
///
/// ```
/// assert_eq!(true, deep_learning_playground::perceptron::single::step_function(0.1));
/// assert_eq!(false, deep_learning_playground::perceptron::single::step_function(-0.1));
/// ```
pub fn step_function<T: Float>(val: T) -> bool {
    if val <= T::zero() {
        false
    } else {
        true
    }
}

/// `logical_perceptron` is the logical perceptron.
/// It can generates some logical gates (`and_perceptron`, `nand_perceptron`, `or_perceptron`)
/// and generate a function that passes the sum of weighted signals
/// (\\(b+X\cdot W\\) where \\(b\in\mathbb{R}\\) is a bias
/// value, \\(W^{2\times 1}=\left(w_1,w_2\right)^T, X^{1\times 2}=\left(x_1,x_2\right)\\) and
/// \\(w_1\\) and \\(w_2\\) are weights of \\(x_1\\) and \\(x_2\\)) to the activation function.
///
/// # Arguments
///