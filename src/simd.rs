//! Core SIMD implementation.

use std::simd::prelude::*;
use std::simd::LaneCount;
use std::simd::SimdElement;
use std::simd::SupportedLaneCount;

/// Decodes `ascii` as base64. Returns the results of the decoding in the low
/// 3/4 of the returned vector, as well as whether decoding completed
/// successfully.
#[inline]
pub fn decode<const N: usize>(ascii: Simd<u8, N>) -> (Simd<u8, N>, bool)
where
  LaneCount<N>: SupportedLaneCount,
{
  // We need to convert each ASCII octet into a sextet, according to this match:
  //
  //    match c {
  //      A..=Z => c - 'A',       // c - 65 in 0x41..=0x5a
  //      a..=z => c - 'a' + 26,  // c - 71 in 0x61..=0x7a
  //      0..=9 => c - '0' + 52,  // c + 4  in 0x30..=0x39
  //      +     => c - '+' + 62,  // c + 19 in 0x2b
  //      /     => c - '/' + 63,  // c + 16 in 0x2f
  //    }

  // One approach is to use comparison masks to extract the pieces of the
  // input corresponding to each of the five cases above, and then map them
  // to the corresponding value we need to offset `ascii` by.

  /*
  use std::ops::RangeInclusive;
  let in_range = |bytes: Simd<u8, N>, range: RangeInclusive<u8>| {
    bytes.simd_ge(Simd::splat(*range.start()))
      & bytes.simd_le(Simd::splat(*range.end()))
  };

  let uppers = in_range(ascii, b'A'..=b'Z');
  let lowers = in_range(ascii, b'a'..=b'z');
  let digits = in_range(ascii, b'0'..=b'9');
  let pluses = ascii.simd_eq([b'+'; N].into());
  let slashes = ascii.simd_eq([b'/'; N].into());

  let valid = (uppers | lowers | digits | pluses | slashes).all();

  let sextets = ascii.cast::<i8>()
    + mask_splat(uppers, -65)
    + mask_splat(lowers, -71)
    + mask_splat(digits, 4)
    + mask_splat(pluses, 19)
    + mask_splat(slashes, 16);
  */

  // However, it turns out to be *almost twice as fast* to use a perfect hash!
  //
  // The function `|c| (c >> 4) - (c == '/')` is a perfect hash for
  // the match above, which maps the five categories as such:
  //
  //    match c {
  //      A..=Z => 4 or 5,
  //      a..=z => 6 or 7,
  //      0..=9 => 3,
  //      +     => 2,
  //      /     => 1,
  //    }
  //
  // We can then use a shuffle to select one of the corresponding offsets,
  // -65, -71, 4, 19, or 16, and add that to `ascii`.
  //
  // This perfect hash function is described at
  // https://github.com/WojciechMula/base64simd/issues/3.

  let hashes = (ascii >> Simd::splat(4))
    + Simd::simd_eq(ascii, Simd::splat(b'/'))
      .to_int()
      .cast::<u8>();

  let sextets = ascii
    + simd!(N; |i| [!0, 16, 19, 4, 191, 191, 185, 185][i % 8])
      .swizzle_dyn(hashes);

  // We also need to do a range check to reject invalid characters.

  const LO_LUT: Simd<u8, 16> = Simd::from_array([
    0b10101, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001,
    0b10001, 0b10001, 0b10011, 0b11010, 0b11011, 0b11011, 0b11011, 0b11010,
  ]);

  const HI_LUT: Simd<u8, 16> = Simd::from_array([
    0b10000, 0b10000, 0b00001, 0b00010, 0b00100, 0b01000, 0b00100, 0b01000,
    0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000,
  ]);

  let lo = swizzle::<16, N>(LO_LUT, ascii & Simd::splat(0x0f));
  let hi = swizzle::<16, N>(HI_LUT, ascii >> Simd::splat(4));
  let valid = (lo & hi).reduce_or() == 0;

  // Now we need to shift everything a little bit, since each byte has two high
  // bits it shouldn't that we need to delete. One thing we can do is to split
  // the vector into two alternating vectors, convert them to vectors of u16,
  // shift each lane by a specified amount, and then shuffle-OR them back
  // together. I learned this trick from Danila Kutenin.
  //
  // What we're basically going to do is the following. Below letters represent
  // the decoded message and dots are extraneous zeros. (Bits below are ordered
  // little-endian.)
  //
  // start:
  //  aaaaaa.. bbbbbb.. cccccc.. dddddd.. eeeeee.. ffffff.. gggggg.. hhhhhh..
  //
  // zext to u16:
  //  aaaaaa.......... bbbbbb.......... cccccc.......... dddddd..........
  //  eeeeee.......... ffffff.......... gggggg.......... hhhhhh..........
  //
  // u16 shift:
  //  ..aaaaaa|........ ....bbbb|bb...... ......cc|cccc.... dddddd..|........
  //  ..eeeeee|........ ....ffff|ff...... ......gg|gggg.... hhhhhh..|........
  //
  // u16 deinterleave
  //  ..aaaaaa|........ ......cc|cccc.... ..eeeeee|........ ......gg|gggg....
  //  ....bbbb|bb...... dddddd..|........ ....ffff|ff...... hhhhhh..|........
  //
  // u8 shuffles:
  //  ..aaaaaa|cccc.... ......cc|..eeeeee ....ffff|......gg ........|........
  //  bb......|....bbbb dddddd..|ff...... gggg....|hhhhhh.. ........|........
  //
  //
  // or:
  //  bbaaaaaa ccccbbbb ddddddcc ffeeeeee ggggffff hhhhhhgg ........ ........

  let sextets16 = sextets.cast::<u16>();
  let shifted = sextets16 << simd!(N; |i| [2, 4, 6, 0][i % 4]);

  let split = |x: Simd<u16, N>| {
    let lows = x.cast::<u8>();
    let highs = (x >> Simd::splat(8)).cast::<u8>();
    Simd::interleave(lows, highs).0
  };

  // Now we need to split `shifted` into two `u8` vectors of N lanes.
  let (a16, b16) = Simd::deinterleave(shifted, Simd::splat(0));
  let a8 = split(a16);
  let b8 = split(b16);

  // We're not quite done, because every other byte of the above vectors needs
  // to be shifted a multiple of bytes correlated with its index.
  //
  // Essentially, we need to delete one out of every four bytes (1 mod 4 and
  // 3 mod 4 bytes resp.) and also swap the two bytes that contain chunks of
  // the same sextet. In other words, the pattern is (0, 3, 2) + 4n and
  // (1, 0, 2) + 4n
  let a = swizzle!(N; a8, array!(N; |i| i / 3 * 4 + [0, 3, 2][i % 3]));
  let b = swizzle!(N; b8, array!(N; |i| i / 3 * 4 + [1, 0, 2][i % 3]));

  (a | b, valid)
}

/// Encodes the low 3/4 bytes of `data` as base64. The high quarter of the
/// input is ignored.
#[inline]
pub fn encode<const N: usize>(data: Simd<u8, N>) -> Simd<u8, N>
where
  LaneCount<N>: SupportedLaneCount,
{
  // First, we need to begin by undoing the "or" at the end of decode_simd.
  // This is a matter of applying a mask made of the 24 (little endian)
  // bits 00111111 11110000 00000011 = [251, 31, 196] repeating.
  let mask = simd!(N; |i| [0b11111100, 0b00001111, 0b11000000][i % 3]);

  // This is the inverse of the shuffle at the end of decode().
  let a8 = swizzle!(N; data & mask, array!(N; |i| {
    i / 4 * 3 + [0, N, 2, 1][i % 4]
  }));
  let b8 = swizzle!(N; data & !mask, array!(N; |i| {
    i / 4 * 3 + [1, 0, 2, N][i % 4]
  }));

  let join = |x: Simd<u8, N>| {
    let (lows, highs) = Simd::deinterleave(x, Simd::splat(0));
    lows.cast::<u16>() | highs.cast::<u16>() << Simd::splat(8)
  };

  let (shifted, _) = Simd::interleave(join(a8), join(b8));

  let sextets = (shifted >> simd!(N; |i| [2, 4, 6, 0][i % 4])).cast::<i8>();

  // Now we have what is essentially a u6 array that looks like this:
  //  aaaaaa.. bbbbbb.. cccccc.. dddddd.. eeeeee.. ffffff.. gggggg.. hhhhhh..

  let uppers = sextets.simd_lt(Simd::splat(26));
  let lowers = !uppers & sextets.simd_lt(Simd::splat(52));
  let digits = !uppers & !lowers & sextets.simd_lt(Simd::splat(62));
  let pluses = sextets.simd_eq(Simd::splat(62));
  let slashes = sextets.simd_eq(Simd::splat(63));

  let ascii = sextets
    - mask_splat(uppers, -65)
    - mask_splat(lowers, -71)
    - mask_splat(digits, 4)
    - mask_splat(pluses, 19)
    - mask_splat(slashes, 16);

  ascii.cast::<u8>()
}

/// Shorthand for mask.select(splat(val), splat(0)).
fn mask_splat<T, const N: usize>(mask: Mask<T::Mask, N>, val: T) -> Simd<T, N>
where
  T: SimdElement + Default,
  LaneCount<N>: SupportedLaneCount,
{
  mask.select(Simd::splat(val), Simd::splat(Default::default()))
}

/// Resizes a vector by either truncation or padding with zeroes.
fn resize<T, const N: usize, const M: usize>(v: Simd<T, N>) -> Simd<T, M>
where
  T: SimdElement + Default,
  LaneCount<N>: SupportedLaneCount,
  LaneCount<M>: SupportedLaneCount,
{
  let len = usize::min(N, M);
  let mut out = Simd::default();
  out.as_mut_array()[..len].copy_from_slice(&v.as_array()[..len]);
  out
}

/// Creates a new `M`-byte vector by treating each element of `indices` as an
/// index into `table`, which is treated as being padded to infinite length
/// with zero.
fn swizzle<const N: usize, const M: usize>(
  table: Simd<u8, N>,
  indices: Simd<u8, M>,
) -> Simd<u8, M>
where
  LaneCount<N>: SupportedLaneCount,
  LaneCount<M>: SupportedLaneCount,
{
  if N < M {
    Simd::swizzle_dyn(resize(table), indices)
  } else {
    resize(Simd::swizzle_dyn(table, resize(indices)))
  }
}
