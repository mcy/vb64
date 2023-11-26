//! Helper macros.

use std::mem::MaybeUninit;
use std::simd::LaneCount;
use std::simd::Simd;
use std::simd::SimdElement;
use std::simd::SupportedLaneCount;

/// Takes an "index table" and generates an inverted index, i.e. such that
/// `invert_index(x)[x[i]] == i` whenever both array accesses are in-bounds.
#[inline(always)]
pub const fn invert_index<const N: usize>(array: [usize; N]) -> [usize; N] {
  let mut out = [N; N];
  let mut i = 0;
  while i < N {
    if array[i] < N {
      out[array[i]] = i;
    }
    i += 1;
  }

  out
}

/// Generates a new vector by tiling `data` repeatedly.
#[inline(always)]
pub const fn tiled<T, const N: usize>(data: &[T]) -> Simd<T, N>
where
  T: SimdElement,
  LaneCount<N>: SupportedLaneCount,
{
  let mut out = [data[0]; N];
  let mut i = 0;
  while i < N {
    out[i] = data[i % data.len()];
    i += 1;
  }

  Simd::from_array(out)
}

/// Type-checking guard for `array!()`.
#[inline(always)]
pub const unsafe fn array_assume_init<T: Copy, const N: usize>(
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

    unsafe { $crate::util::array_assume_init::<_, $N>(&array) }
  }};
}

/// Like std::simd::swizzle!, but where the static indexing vector can depend
/// on a const parameter, e.g. an `array!()` call.
macro_rules! swizzle {
  ($N:ident; $x:expr, $index:expr) => {{
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
          if i >= $N {
            Which::Second(0)
          } else {
            Which::First(i)
          }
        })
      };
    }

    Swz::swizzle2($x, Simd::splat(0))
  }};
}
