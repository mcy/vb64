//! Helper macros.

use std::mem::MaybeUninit;

/// Type-checking guard for `array!()`.
pub(crate) const unsafe fn array_assume_init<T: Copy, const N: usize>(
  array: &[MaybeUninit<T>; N],
) -> [T; N] {
  (array as *const [MaybeUninit<T>; N])
    .cast::<[T; N]>()
    .read()
}

/// Constructs a new array of a given length by executing a "closure" on each
/// index.
macro_rules! array {
  ($N:expr; |$idx:ident| $body:expr) => {{
    use std::mem::MaybeUninit;
    let mut array = [MaybeUninit::uninit(); $N];
    let mut i = 0;
    while i < $N {
      array[i] = MaybeUninit::new({
        let $idx = i;
        $body
      });
      i += 1;
    }

    unsafe { $crate::macros::array_assume_init::<_, $N>(&array) }
  }};
}

/// Constructs a new vector of a given length by executing a "closure" on each
/// index.
macro_rules! simd {
  ($N:expr; |$idx:ident| $body:expr) => {
    std::simd::Simd::<_, $N>::from_array(array!($N; |$idx| $body))
  };
}

/// Like std::simd::swizzle!, but where the static indexing vector can depend
/// on a const parameter, e.g. an `array!()` call.
macro_rules! swizzle {
  ($N:ident; $x:expr, $index:expr) => {
    swizzle!($N; $x, Simd::splat(0), $index)
  };

  ($N:ident; $x:expr, $y:expr, $index:expr) => {{
    use std::simd::*;
    struct Swz;
    impl<const $N: usize> Swizzle2<$N, $N> for Swz
    where
      LaneCount<$N>: SupportedLaneCount,
    {
      const INDEX: [Which; $N] = {
        let index = $index;
        array!($N; |i| {
          let i = index[i];
          match i.checked_sub($N) {
            None => Which::First(i),
            Some(j) => Which::Second(j),
          }
        })
      };
    }

    Swz::swizzle2($x, $y)
  }};

  ($N:ident x $M:ident; $x:expr, $y:expr, $index:expr) => {{
    use std::simd::*;
    struct Swz;
    impl<const $N: usize, const $M: usize> Swizzle2<$N, $M> for Swz
    where
      LaneCount<$N>: SupportedLaneCount,
      LaneCount<$M>: SupportedLaneCount,
    {
      const INDEX: [Which; $M] = {
        let index = $index;
        array!($M; |i| {
          let i = index[i];
          match i.checked_sub($N) {
            None => Which::First(i),
            Some(j) => Which::Second(j),
          }
        })
      };
    }

    Swz::swizzle2($x, $y)
  }};
}
