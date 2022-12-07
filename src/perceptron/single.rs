
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
/// * `w1` - Weights for signal `x1`
/// * `w2` - Weights for signal `x2`
/// * `bias` - Bias that determines the ease of firing of neurons
///
/// # e.g.
///
/// ```
/// assert_eq!(true, deep_learning_playground::perceptron::single::logical_perceptron(&0.5, &0.5, &-0.7)(true, true));
/// assert_eq!(false, deep_learning_playground::perceptron::single::logical_perceptron(&-0.5, &-0.5, &0.7)(true, true));
/// assert_eq!(true, deep_learning_playground::perceptron::single::logical_perceptron(&0.5, &0.5, &-0.2)(true, false));
/// ```
pub fn logical_perceptron<T: Float>(
    w1: &'static T,
    w2: &'static T,
    bias: &'static T,
) -> Box<dyn Fn(bool, bool) -> bool> {
    Box::new(move |x1: bool, x2: bool| -> bool {
        let f = |x| if x { T::one() } else { T::zero() };
        step_function(*bias + *w1 * f(x1) + *w2 * f(x2))
    })
}

/// `and_perceptron` generates the logical AND function.
/// Let \\(p\\) and \\(q\\) are logical variables, `and_perceptron` generates the function which
/// satisfies the following truth table.
///
/// | \\(p\\) | \\(q\\) | `and_perceptron()(`\\(p\\)`,`\\(q\\)`)` |
/// | -- | -- | -- |
/// | \\(1\\) | \\(1\\) | \\(1\\) |
/// | \\(1\\) | \\(0\\) | \\(0\\) |
/// | \\(0\\) | \\(1\\) | \\(0\\) |
/// | \\(0\\) | \\(0\\) | \\(0\\) |
///
/// # e.g.
///
/// ```
/// assert_eq!(true, deep_learning_playground::perceptron::single::and_perceptron()(true, true));
/// assert_eq!(false, deep_learning_playground::perceptron::single::and_perceptron()(false, true));