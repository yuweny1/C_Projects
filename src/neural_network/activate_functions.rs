use ndarray::Array2;
use ndarray_stats::QuantileExt;
use num::Float;

/// `identity` generates the identity function
/// \\(
/// \text{id}(x)=x
/// \\) for `Array2`
pub fn identity<T>() -> Box<dyn Fn(Array2<T>) -> Array2<T>> {
    Box::new(|x| -> Ar