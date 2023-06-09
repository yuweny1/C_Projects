
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
/// assert_eq!(false, deep_learning_playground::perceptron::single::and_perceptron()(false, false));
/// ```
pub fn and_perceptron() -> Box<dyn Fn(bool, bool) -> bool> {
    logical_perceptron(&0.5, &0.5, &-0.7)
}

/// `nand_perceptron` generates the logical NAND function.
/// Let \\(p\\) and \\(q\\) are logical variables, `nand_perceptron` generates the function which
/// satisfies the following truth table.
///
/// | \\(p\\) | \\(q\\) | `nand_perceptron()(`\\(p\\)`,`\\(q\\)`)` |
/// | -- | -- | -- |
/// | \\(1\\) | \\(1\\) | \\(0\\) |
/// | \\(1\\) | \\(0\\) | \\(1\\) |
/// | \\(0\\) | \\(1\\) | \\(1\\) |
/// | \\(0\\) | \\(0\\) | \\(1\\) |
///
/// # e.g.
///
/// ```
/// assert_eq!(true, deep_learning_playground::perceptron::single::nand_perceptron()(true, false));
/// assert_eq!(false, deep_learning_playground::perceptron::single::nand_perceptron()(true, true));
/// ```
pub fn nand_perceptron() -> Box<dyn Fn(bool, bool) -> bool> {
    logical_perceptron(&-0.5, &-0.5, &0.7)
}

/// `or_perceptron` generates the logical OR function.
/// Let \\(p\\) and \\(q\\) are logical variables, `or_perceptron()(`\\(p\\)`,`\\(q\\)`)` generates the function which
/// satisfies the following truth table.
///
/// | \\(p\\) | \\(q\\) | `or_perceptron()(`\\(p\\)`,`\\(q\\)`)` |
/// | -- | -- | -- |
/// | \\(1\\) | \\(1\\) | \\(1\\) |
/// | \\(1\\) | \\(0\\) | \\(1\\) |
/// | \\(0\\) | \\(1\\) | \\(1\\) |
/// | \\(0\\) | \\(0\\) | \\(0\\) |
///
/// # e.g.
///
/// ```
/// assert_eq!(true, deep_learning_playground::perceptron::single::or_perceptron()(true, false));
/// assert_eq!(false, deep_learning_playground::perceptron::single::or_perceptron()(false, false));
/// ```
pub fn or_perceptron() -> Box<dyn Fn(bool, bool) -> bool> {
    logical_perceptron(&0.5, &0.5, &-0.2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_logical_gates_impl(
        logical_gate: &Box<dyn Fn(bool, bool) -> bool>,
        expected_answer: &[bool; 4],
    ) {
        let mut i: usize = 0;
        for b1 in [true, false].iter() {
            for b2 in [true, false].iter() {
                assert_eq!(logical_gate(*b1, *b2), expected_answer[i]);
                i += 1;
            }
        }
    }

    #[test]
    fn test_logical_gates() {
        test_logical_gates_impl(&and_perceptron(), &[true, false, false, false]);
        test_logical_gates_impl(&nand_perceptron(), &[false, true, true, true]);
        test_logical_gates_impl(&or_perceptron(), &[true, true, true, false]);
    }
}