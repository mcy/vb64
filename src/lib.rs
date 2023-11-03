//! Fast, SIMD base64 codecs.
//!
//! This crate implements base64 encoding and decoding as fast as possible.
//! To get maximum performance, compile with `-Ctarget-cpu=native` and
//! `-Zbuild-std`, which will ensure ideal instruction selection.
//!
//! The following is a perf comparison with the `base64` crate on a Zen 2
//! machine using AVX2 instructions; lower is better.
//! 
//! ![perf comparison with `base64`][graph-png]
//! 
//! On a Zen 2 machine and compiling with AVX2 support, decoding is between
//! 2x to 2.5x faster than `base64`, while encoding is around 1.2x to 1.5x
//! faster; with only SSSE3, decoding performance is even with `base64` and
//! encoding is much worse.
//!
//! It is relatively unlikely that base64 decoding is such a massive bottleneck
//! for your application that this matters, unless you're parsing base64 blobs
//! embedded in JSON; you may want to consider using a binary format like
//! Protobuf instead.
//! 
//! Also this crate uses `std::simd` so it requires nightly.
//!
//! # Constant time?? 👀
//! 
//! For decoding valid base64 (and for encoding any message), the
//! implementations are essentially constant-time, but mostly by accident, since
//! they are branchless and use shuffle-based lookup tables. Whether you
//! want to believe that this decodes your private key `.pem` files without
//! side-channel-leaking key material up to you (OpenSSL 100% leaks your private
//! keys this way).
//!
// The comedy of using base64 to encode an image of benchmark results from my
// base64 library is not lost on me.
#![doc = concat!("[graph-png]: data:image/png;base64,", include_str!("../images/graph.png.base64"))]

#![feature(portable_simd)]

use std::simd::LaneCount;
use std::simd::Simd;
use std::simd::SupportedLaneCount;

#[macro_use]
mod macros;

mod simd;

/// The error returned by all decode functions.
#[derive(Copy, Clone, Debug)]
pub struct Error;

/// Decodes some base64 `data` to a fresh vector.
pub fn decode(data: &[u8]) -> Result<Vec<u8>, Error> {
  let mut out = Vec::new();
  decode_to(data, &mut out)?;
  Ok(out)
}

/// Encodes arbitrary data as base64.
pub fn encode(data: &[u8]) -> String {
  let mut out = Vec::new();
  encode_to(data, &mut out);
  unsafe { String::from_utf8_unchecked(out) }
}

/// Decodes some base64 data as base64 and appends it to `out`.
pub fn decode_to(data: &[u8], out: &mut Vec<u8>) -> Result<(), Error> {
  if cfg!(target_feature = "avx2") {
    decode_tunable::<32>(data, out)
  } else {
    decode_tunable::<32>(data, out)
  }
}

/// Encodes arbitrary data as base64 and appends it to `out`.
pub fn encode_to(data: &[u8], out: &mut Vec<u8>) {
  if cfg!(target_feature = "avx2") {
    encode_tunable::<16>(data, out)
  } else {
    encode_tunable::<8>(data, out)
  }
}

fn decode_tunable<const N: usize>(
  mut data: &[u8],
  out: &mut Vec<u8>,
) -> Result<(), Error>
where
  LaneCount<N>: SupportedLaneCount,
{
  assert!(N % 4 == 0);

  if let Some(stripped) =
    data.strip_suffix(b"==").or_else(|| data.strip_suffix(b"="))
  {
    data = stripped;
  }

  if data.is_empty() {
    return Ok(());
  }

  // NOTE: Always a full N bytes of slop so we can do full SIMD stores.
  out.reserve(decoded_len(data.len()) + N);
  let mut raw_out = out.as_mut_ptr_range().end;

  let mut chunks = data.chunks_exact(N);
  let mut failed = false;
  while let Some(chunk) = chunks.next().filter(|_| !failed) {
    let ascii = Simd::from_slice(chunk);
    let (decoded, ok) = simd::decode(ascii);
    failed |= !ok;

    unsafe {
      raw_out.cast::<Simd<u8, N>>().write_unaligned(decoded);
      raw_out = raw_out.add(decoded_len(N));
    }
  }

  let rest = chunks.remainder();
  if !failed && !rest.is_empty() {
    // 'A' decodes as 0, so we can pretend the rest of the array is padded with
    // 'A's.
    let ascii = Simd::gather_or(rest, simd!(N; |i| i), Simd::splat(b'A'));
    let (decoded, ok) = simd::decode(ascii.into());
    failed |= !ok;

    unsafe {
      raw_out.cast::<Simd<u8, N>>().write_unaligned(decoded);
      raw_out = raw_out.add(decoded_len(rest.len()));
    }
  }

  if failed {
    return Err(Error);
  }

  unsafe {
    let new_len = raw_out.offset_from(out.as_ptr());
    out.set_len(new_len as usize);
  }

  Ok(())
}

fn encode_tunable<const N: usize>(data: &[u8], out: &mut Vec<u8>)
where
  LaneCount<N>: SupportedLaneCount,
{
  assert!(N % 4 == 0);
  let n3q = N / 4 * 3;

  if data.is_empty() {
    return;
  }

  // NOTE: Always a full N bytes of slop so we can do full SIMD stores.
  out.reserve(encoded_len(data.len()) + N);
  let mut raw_out = out.as_mut_ptr_range().end;

  // Can't use `[u8]::chunks` here, because we want 32-byte windows so we can
  // do full 32-byte loads, but we want them to overlap by 8 bytes; we also
  // want eight bytes of slop on the last chunk.
  //
  // There are two cases: either data.len() % 24 >= 8, or not; in the former
  // case, we can load every full chunk with a full load, but in the latter we
  // need an extra case to load less than 24 bytes.
  //
  // There is also a third, extra case where data.len() < 32, in which case
  // we need to not do pointer arithmetic below.
  let mut start = data.as_ptr();
  let end = unsafe {
    if data.len() % n3q >= (N - n3q) {
      start.add(data.len() - data.len() % n3q)
    } else if data.len() < N {
      start
    } else {
      start.add(data.len() - data.len() % n3q - n3q)
    }
  };

  while start != end {
    let chunk = unsafe { std::slice::from_raw_parts(start, N) };
    let encoded = simd::encode(Simd::from_slice(chunk));

    unsafe {
      start = start.add(n3q);

      raw_out.cast::<Simd<u8, N>>().write_unaligned(encoded);
      raw_out = raw_out.add(N);
    }
  }

  let end = data.as_ptr_range().end;
  while start < end {
    let chunk = unsafe {
      let rest = end.offset_from(start) as usize;
      std::slice::from_raw_parts(start, rest.min(n3q))
    };
    let encoded = simd::encode(Simd::gather_or_default(chunk, simd!(N; |i| i)));

    unsafe {
      start = start.add(chunk.len());

      raw_out.cast::<Simd<u8, N>>().write_unaligned(encoded);
      raw_out = raw_out.add(encoded_len(chunk.len()));
    }
  }

  unsafe {
    let new_len = raw_out.offset_from(out.as_ptr());
    out.set_len(new_len as usize);
  }

  match out.len() % 4 {
    2 => out.extend_from_slice(b"=="),
    3 => out.extend_from_slice(b"="),
    _ => {}
  }
}

fn decoded_len(input: usize) -> usize {
  input / 4 * 3
    + match input % 4 {
      1 | 2 => 1,
      3 => 2,
      _ => 0,
    }
}

fn encoded_len(input: usize) -> usize {
  input / 3 * 4
    + match input % 3 {
      1 => 2,
      2 => 3,
      _ => 0,
    }
}

#[cfg(test)]
mod tests {
  fn tests() -> Vec<(usize, &'static [u8], Vec<u8>)> {
    use base64::prelude::*;
    include_bytes!("test_vectors.txt")
      .split(|&b| b == b'\n')
      .enumerate()
      .map(|(i, b64)| (i, b64, BASE64_STANDARD.decode(b64).unwrap()))
      .collect()
  }

  #[test]
  fn decode() {
    for (i, enc, dec) in tests() {
      assert_eq!(crate::decode(enc).unwrap(), dec, "case {i}");
    }
  }

  #[test]
  fn encode() {
    for (i, enc, dec) in tests() {
      assert_eq!(crate::encode(&dec).as_bytes(), enc, "case {i}");
    }
  }

  #[test]
  fn alphabet() {
    for b in 0..255u8 {
      let res = crate::decode(&[b, b'=', b'=']);
      if b.is_ascii_alphanumeric() || b == b'+' || b == b'/' {
        assert!(res.is_ok(), "{b:#04x} is valid data");
      } else {
        assert!(res.is_err(), "{b:#04x} is not valid data");
      }
    }
  }

  #[test]
  #[ignore]
  fn keep_for_disassembly() {
    std::hint::black_box((super::decode as usize, super::encode as usize));
  }
}
