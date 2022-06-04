use ndarray::Array2;
use ndarray_stats::QuantileExt;
use num::Float;

/// `identity` generates the identity function
/// \\(
/// \text{id}(x)=x
/// \\) for `Array2`
pub fn identity<T>() -> Box<dyn Fn(Array2<T>) -> Array2<T>> {
    Box::new(|x| -> Array2<T> { x })
}

/// `sigmoid` generates the sigmoid function
/// \\[
/// S(x)=\dfrac{1}{1+\exp(-x)}
/// \\] for `Array2` w