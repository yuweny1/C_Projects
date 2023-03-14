pub mod fetch_client;
pub mod natural_transform;

#[inline]
pub fn fst<T, U>(x: (T, U)) -> T {
    let (y, _) 