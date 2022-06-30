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
/// \\] for `Array2` where \\(x\in\mathbb{R}\\).
pub fn sigmoid<T: Float>() -> Box<dyn Fn(Array2<T>) -> Array2<T>> {
    Box::new(|x| -> Array2<T> { x.map(|val| T::one() / (T::one() + (-*val).exp())) })
}

/// `rectified_linear_unit` generates the Rectufied Linear Unit (ReLU) function
/// \\[
/// \text{ReLU}(x)=\begin{cases}
/// x & (x\gt 0) \\\\
/// 0 & (\text{otherwise})
/// \end{cases}
/// \\] for `Array2` where \\(x\in\mathbb{R}\\).
pub fn rectified_linear_unit<T: Float>() -> Box<dyn Fn(Array2<T>) -> Array2<T>> {
    Box::new(|x| -> Array2<T> { x.map(|val| if *val > T::zero() { *val } else { T::zero() }) })
}

/// `softmax