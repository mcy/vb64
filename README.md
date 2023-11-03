# vb64

Fast, SIMD base64 codecs.

This crate implements base64 encoding and decoding as fast as possible.
To get maximum performance, compile with `-Ctarget-cpu=native` and
`-Zbuild-std`, which will ensure ideal instruction selection.

The following is a perf comparison with the `base64` crate on a Zen 2
machine using AVX2 instructions; lower is better.

![perf comparison with `base64`](images/graph.png)

On a Zen 2 machine and compiling with AVX2 support, decoding is between
2x to 2.5x faster than `base64`, while encoding is around 1.2x to 1.5x
faster; with only SSSE3, decoding performance is even with `base64` and
encoding is much worse.

It is relatively unlikely that base64 decoding is such a massive bottleneck
for your application that this matters, unless you're parsing base64 blobs
embedded in JSON; you may want to consider using a binary format like
Protobuf instead.

Also this crate uses `std::simd` so it requires nightly.

License: Apache-2.0
