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
/// ``