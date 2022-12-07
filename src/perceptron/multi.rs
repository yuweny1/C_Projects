extern crate num;

use super::single::{and_perceptron, nand_perceptron, or_perceptron};

/// `xor_perceptron` generates the XOR function.
/// Let \\(p\\) and \\(q\\) are logical variables, `xor_perceptron` generates the function which
/// satisfies the following truth table.
///
/// | \\(p\\) | \\(q\\) | `xor_perceptron()(`\\(p\\)`,`\\(q\\)`)` |
/// | -- | -- | -- |
/// | \\(1\\) | \\(1\\) | \\(0\\) |
/// | \\(1\\) | \\(0\\) | \\(1\\) |
/// | \\(0\\) | \\(1\\) | \\(1\\) |
/// | \\(0\\) | \\(0\\) | \\(0\\) |
///
/// # e.g.
///
/// ```
/// assert_eq!(true, deep_learning_playground::perceptron::multi::xor_perceptron()(true, false));
/// assert_eq!(false, deep_learning_playground::perceptron::multi::xor_perceptron()(true, true));
/// ```
pub fn xor_perceptron() -> Box<dyn Fn(bool, bool) -> bool> {
    Box::new(|x1: bool, x2: bool| -> bool {
        and_perceptron()(nand_perceptron()(x1, x2), or_perceptron()(x1, x2))
    })
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
        test_logical_gates_impl(&xor_perceptron(), &[false, true, true, false]);
    }
}
