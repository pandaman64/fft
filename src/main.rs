#![feature(test)]

extern crate num_complex;
use num_complex::{Complex32, ComplexDistribution as D};
use std::f32::consts::PI;

extern crate rand;
use rand::distributions::{Distribution, Uniform};
use rand::prelude::*;

fn dft(input: &[Complex32]) -> Vec<Complex32> {
    let n = input.len();
    let omega = Complex32::new(0.0, 2.0 * PI / (n as f32)).exp();

    let mut ret = Vec::with_capacity(n);

    for j in 0..n {
        ret.push(Complex32::new(0.0, 0.0));
        for k in 0..n {
            let jj = j as f32;
            let kk = k as f32;
            ret[j] += omega.powf(jj * kk) * input[k];
        }
    }

    for j in 0..n {
        ret[j] /= (n as f32).sqrt();
    }

    ret
}

fn fft(input: &[Complex32], initial: bool) -> Vec<Complex32> {
    let n = input.len();
    if n == 1 {
        return input.to_owned();
    }

    let omega = Complex32::new(0.0, 2.0 * PI / (n as f32)).exp();

    let mut even = Vec::with_capacity(n / 2);
    let mut odd = Vec::with_capacity(n / 2);

    for j in 0..(n / 2) {
        even.push(input[2 * j]);
        odd.push(input[2 * j + 1]);
    }

    let even = fft(&even, false);
    let odd = fft(&odd, false);

    let mut ret = Vec::with_capacity(n);

    for j in 0..(n / 2) {
        ret.push(even[j] + odd[j] * omega.powf(j as f32));
    }
    for j in 0..(n / 2) {
        ret.push(even[j] + odd[j] * omega.powf((n / 2 + j) as f32));
    }

    if initial {
        for x in ret.iter_mut() {
            *x /= (n as f32).sqrt();
        }
    }

    ret
}

fn fft_stride(input: &[Complex32], start: usize, stride: usize, output: &mut [Complex32]) {
    let n = input.len();
    let m = output.len();
    assert_eq!(n, m * stride);

    if n == stride {
        output[0] = input[start];
        return;
    }

    let omega = Complex32::new(0.0, 2.0 * PI / (m as f32)).exp();

    {
        let (even, odd) = output.split_at_mut(m / 2);
        fft_stride(input, start, stride * 2, even);
        fft_stride(input, start + stride, stride * 2, odd);

        for j in 0..(m / 2) {
            let tmp = odd[j];
            odd[j] = even[j] + omega.powf((m / 2 + j) as f32) * tmp;
            even[j] = even[j] + omega.powf(j as f32) * tmp;
        }
    }

    if stride == 1 {
        for x in output.iter_mut() {
            *x /= (n as f32).sqrt();
        }
    }
}

fn print_seq<T: std::fmt::Display>(input: &[T], start: usize, stride: usize) {
    if input.len() == 0 {
        println!("[]");
    } else {
        print!("[{:.4}", input[start]);
        for j in 1..(input.len() / stride) {
            print!(" {:.4}", input[start + j * stride]);
        }
        println!("]");
    }
}

fn distance(lhs: &[Complex32], rhs: &[Complex32]) -> f32 {
    let mut ret = 0.0;
    for (lhs, rhs) in lhs.iter().zip(rhs.iter()) {
        ret += (lhs - rhs).norm_sqr();
    }
    ret
}

fn random_test(bit_len: usize, count: usize, tolerance: f32) {
    let n = 2_usize.pow(bit_len as u32);
    let mut rng = thread_rng();
    let dist = D::new(Uniform::new(0.0, 1.0), Uniform::new(0.0, 1.0));
    let mut iter = dist.sample_iter(&mut rng);
    for _ in 0..count {
        let input = iter.by_ref().take(n).collect::<Vec<_>>();

        let dft = dft(&input);
        //let fft = fft(&input, true);
        let mut fft = vec![Complex32::new(0.0, 0.0); n];
        fft_stride(&input, 0, 1, &mut fft);

        assert!(distance(&dft, &fft) < tolerance);
    }
}

fn random_input(bit_len: usize) -> Vec<Complex32> {
    let n = 2_usize.pow(bit_len as u32);
    let mut rng = thread_rng();
    let dist = D::new(Uniform::new(0.0, 1.0), Uniform::new(0.0, 1.0));
    let iter = dist.sample_iter(&mut rng);
    iter.take(n).collect::<Vec<_>>()
}

#[test]
fn test_precision() {
    let tolerance = 0.01;
    random_test(2, 1000, tolerance);
    random_test(4, 1000, tolerance);
    random_test(6, 100, tolerance);
    random_test(8, 100, tolerance);
    random_test(10, 10, tolerance);
}

extern crate test;
use test::Bencher;

fn run_dft(b: &mut Bencher, bit_len: usize) {
    let input = random_input(bit_len);
    b.iter(|| {
        let input = test::black_box(&input);        
        dft(input);
    })
}

/*
fn run_fft(b: &mut Bencher, bit_len: usize) {
    let input = random_input(bit_len);
    b.iter(|| {
        let input = test::black_box(&input);        
        fft(input, true);
    })
}
*/

fn run_fft(b: &mut Bencher, bit_len: usize) {
    let input = random_input(bit_len);
    b.iter(|| {
        let input = test::black_box(&input);        
        let mut output = vec![Complex32::new(0.0, 0.0); input.len()];
        fft_stride(input, 0, 1, &mut output);
    })
}

#[bench]
fn bench_dft1(b: &mut Bencher) {
    run_dft(b, 1);
}
#[bench]
fn bench_fft1(b: &mut Bencher) {
    run_fft(b, 1);
}
#[bench]
fn bench_dft2(b: &mut Bencher) {
    run_dft(b, 2);
}
#[bench]
fn bench_fft2(b: &mut Bencher) {
    run_fft(b, 2);
}
#[bench]
fn bench_dft3(b: &mut Bencher) {
    run_dft(b, 3);
}
#[bench]
fn bench_fft3(b: &mut Bencher) {
    run_fft(b, 3);
}
#[bench]
fn bench_dft4(b: &mut Bencher) {
    run_dft(b, 4);
}
#[bench]
fn bench_fft4(b: &mut Bencher) {
    run_fft(b, 4);
}
#[bench]
fn bench_dft5(b: &mut Bencher) {
    run_dft(b, 5);
}
#[bench]
fn bench_fft5(b: &mut Bencher) {
    run_fft(b, 5);
}
#[bench]
fn bench_dft6(b: &mut Bencher) {
    run_dft(b, 6);
}
#[bench]
fn bench_fft6(b: &mut Bencher) {
    run_fft(b, 6);
}
#[bench]
fn bench_dft7(b: &mut Bencher) {
    run_dft(b, 7);
}
#[bench]
fn bench_fft7(b: &mut Bencher) {
    run_fft(b, 7);
}
#[bench]
fn bench_dft8(b: &mut Bencher) {
    run_dft(b, 8);
}
#[bench]
fn bench_fft8(b: &mut Bencher) {
    run_fft(b, 8);
}
#[bench]
fn bench_dft9(b: &mut Bencher) {
    run_dft(b, 9);
}
#[bench]
fn bench_fft9(b: &mut Bencher) {
    run_fft(b, 9);
}
#[bench]
fn bench_dft10(b: &mut Bencher) {
    run_dft(b, 10);
}
#[bench]
fn bench_fft10(b: &mut Bencher) {
    run_fft(b, 10);
}
#[bench]
fn bench_dft11(b: &mut Bencher) {
    run_dft(b, 11);
}
#[bench]
fn bench_fft11(b: &mut Bencher) {
    run_fft(b, 11);
}
/*
#[bench]
fn bench_dft12(b: &mut Bencher) {
    run_dft(b, 12);
}
#[bench]
fn bench_fft12(b: &mut Bencher) {
    run_fft(b, 12);
}
#[bench]
fn bench_dft13(b: &mut Bencher) {
    run_dft(b, 13);
}
#[bench]
fn bench_fft13(b: &mut Bencher) {
    run_fft(b, 13);
}
#[bench]
fn bench_dft14(b: &mut Bencher) {
    run_dft(b, 14);
}
#[bench]
fn bench_fft14(b: &mut Bencher) {
    run_fft(b, 14);
}
#[bench]
fn bench_dft15(b: &mut Bencher) {
    run_dft(b, 15);
}
#[bench]
fn bench_fft15(b: &mut Bencher) {
    run_fft(b, 15);
}
#[bench]
fn bench_dft16(b: &mut Bencher) {
    run_dft(b, 16);
}
#[bench]
fn bench_fft16(b: &mut Bencher) {
    run_fft(b, 16);
}
*/

fn main() {
    let input = (0..8).map(|x| Complex32::new(x as f32, 0.0)).collect::<Vec<_>>();
    print_seq(&dft(&input), 0, 1);
    let mut fft = vec![ 0.0.into(); 8 ];
    fft_stride(&input, 0, 1, &mut fft);
    print_seq(&fft, 0, 1);
}
