extern crate num;

use super::single::{and_perceptron, nand_perceptron, or_perceptron};

/// `xor_perceptron` generates the XOR function.
/// Let \\(p\\) and \\(q\\) are logical variables, `xor_perceptron` generates the function which
/// satisfies th