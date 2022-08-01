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

/// `softmax` generates the softmax function:
/// \\\[
/// \text{SoftMax}(\boldsymbol{x})=\left( \dfrac{\exp(x_1)}{\displaystyle\sum^n_{j=1}\exp(x_j)},
/// \dfrac{\exp(x_2)}{\displaystyle\sum^n_{j=1}\exp(x_j)}, \cdots,
/// \dfrac{\exp(x_n)}{\displaystyle\sum^n_{j=1}\exp(x_j)} \right)
/// \\] for `Array2` where \\(\boldsymbol{x}=\left(x_1,\cdots,x_n\right)^T\subseteq\mathbb{R}^{n\times 1}\\).
/// To prevent overflow, actually calculate according to the following equation:
/// \\[
/// \begin{array}{lll}
/// \dfrac{\exp(x_i)}{\displaystyle\sum^n_{j=1}\exp(x_j)}&=&\dfrac{C\exp(x_i)}{C\displaystyle\sum^n_{j=1}\exp(x_j)}\\\\
/// &=&\dfrac{\exp(x_i+\log C)}{\displaystyle\sum^n_{j=1}\exp(x_j+\log C)}\\\\
/// &=&\dfrac{\exp(x_i+C')}{\displaystyle\sum^n_{j=1}\exp(x_j+C')}
/// \end{array}
/// \\]
/// Therefore, \\(C'\\) can be for all value.
/// Thus \\(C'=x_{\text{max}}\\) where \\(^\forall x_i, ^\exists x_{\text{max}}\in\boldsymbol{x}\\) s.t. \\(x_{\text{max}}\geq x_i\\).
pub fn softmax<T: Float>() -> Box<dyn Fn(Array2<T>) -> Array2<T>> {
    Box::new(|x| -> Array2<T> {
        let max: T = *x.max().unwrap();
        let sum: T = x.fold(T::zero(), |acc, val| acc + (*val - max).exp()); // Subtract the maximum value to prevent overflow (this is the equivalent calculation as explained above)
        x.map(|val| (*val - max).exp() / sum)
    })
}
