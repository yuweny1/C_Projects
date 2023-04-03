pub mod fetch_client;
pub mod natural_transform;

#[inline]
pub fn fst<T, U>(x: (T, U)) -> T {
    let (y, _) = x;
    y
}

#[inline]
pub fn snd<T, U>(x: (T, U)) -> U {
    let (_, y) = x;
    y
}
