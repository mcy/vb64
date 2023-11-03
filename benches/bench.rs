use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;

fn tests() -> Vec<(usize, &'static [u8], Vec<u8>)> {
  use base64::prelude::*;
  include_bytes!("../src/test_vectors.txt")
    .split(|&b| b == b'\n')
    .enumerate()
    .skip(8)
    .map(|(i, b64)| (i, b64, BASE64_STANDARD.decode(b64).unwrap()))
    .collect()
}

fn decoded_len(input: usize) -> usize {
  input / 4 * 3
    + match input % 4 {
      1 | 2 => 1,
      3 => 2,
      _ => 0,
    }
}

fn decode(c: &mut Criterion) {
  let mut group = c.benchmark_group("decode");
  for (i, enc, _) in tests() {
    let len = decoded_len(
      std::str::from_utf8(enc)
        .unwrap()
        .trim_end_matches('=')
        .len(),
    );

    group
      .sample_size(500)
      .throughput(Throughput::Bytes(len as u64))
      .bench_with_input(BenchmarkId::new("vb64", i), enc, |b, enc| {
        b.iter(|| vb64::decode(enc))
      })
      .bench_with_input(BenchmarkId::new("baseline", i), enc, |b, enc| {
        use base64::prelude::*;
        b.iter(|| BASE64_STANDARD.decode(enc))
      });
  }
}

fn encode(c: &mut Criterion) {
  let mut group = c.benchmark_group("encode");
  for (i, _, dec) in tests() {
    group
      .throughput(Throughput::Bytes(dec.len() as u64))
      .bench_with_input(BenchmarkId::new("vb64", i), &dec, |b, dec| {
        b.iter(|| vb64::encode(dec))
      })
      .bench_with_input(BenchmarkId::new("baseline", i), &dec, |b, dec| {
        use base64::prelude::*;
        b.iter(|| BASE64_STANDARD.encode(dec))
      });
  }
  group.finish();
}

criterion::criterion_group!(benches, decode, encode);
criterion::criterion_main!(benches);
