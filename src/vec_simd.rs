use cfg_if::cfg_if;
use ndarray::{ArrayView1, ArrayViewMut1, Array1};

#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

cfg_if! {
    if #[cfg(target_feature = "avx")] {
        /// Dot product: u · v
        ///
        /// This SIMD-vectorized function computes the dot product
        /// (BLAS sdot).
        pub fn dot(u: ArrayView1<f32>, v: ArrayView1<f32>) -> f32 {
            unsafe { dot_f32x8(u, v) }
        }
    } else if #[cfg(target_feature = "sse")] {
        /// Dot product: u · v
        ///
        /// This SIMD-vectorized function computes the dot product
        /// (BLAS sdot).
        pub fn dot(u: ArrayView1<f32>, v: ArrayView1<f32>) -> f32 {
            unsafe { dot_f32x4(u, v) }
        }
    } else {
        /// Unvectorized dot product: u · v
        pub fn dot(u: ArrayView1<f32>, v: ArrayView1<f32>) -> f32 {
            dot_unvectorized(u, v)
        }
    }
}

cfg_if! {
    if #[cfg(target_feature = "avx")] {
        /// Scaling: u = au
        ///
        /// This function performs SIMD-vectorized scaling (BLAS sscal).
        pub fn scale(u: ArrayViewMut1<f32>, a: f32) {
            unsafe { scale_f32x8(u, a) }
        }
    } else if #[cfg(target_feature = "sse")] {
        /// Scaling: u = au
        ///
        /// This function performs SIMD-vectorized scaling (BLAS sscal).
        pub fn scale(u: ArrayViewMut1<f32>, a: f32) {
            unsafe { scale_f32x4(u, a) }
        }
    } else {
        /// Unvectorized Scaling: u = au
        pub fn scale(u: ArrayViewMut1<f32>, a: f32) {
            scale_unvectorized(u, a)
        }
    }
}

cfg_if! {
    if #[cfg(target_feature = "avx")] {
        /// Scaled addition: *u = u + av*
        ///
        /// This function performs SIMD-vectorized scaled addition (BLAS saxpy).
        pub fn scaled_add(u: ArrayViewMut1<f32>, v: ArrayView1<f32>, a: f32) {
            unsafe { scaled_add_f32x8(u, v, a) }
        }
    } else if #[cfg(target_feature = "sse")] {
        /// Scaled addition: *u = u + av*
        ///
        /// This function performs SIMD-vectorized scaled addition (BLAS saxpy).
        pub fn scaled_add(u: ArrayViewMut1<f32>, v: ArrayView1<f32>, a: f32) {
            unsafe { scaled_add_f32x4(u, v, a) }
        }
    } else {
        /// Unvectorized scaled addition: *u = u + av*.
        pub fn scaled_add(u: ArrayViewMut1<f32>, v: ArrayView1<f32>, a: f32) {
            scaled_add_unvectorized(u, v, a)
        }
    }
}

cfg_if! {
    if #[cfg(target_feature = "avx")] {
        pub fn kld(u: ArrayView1<f32>, v: ArrayView1<f32>) -> f32 {
            unsafe { kld_f32x8(u, v) }
        }
    } else {
        pub fn kld(u: ArrayView1<f32>, v: ArrayView1<f32>) -> f32 {
            kld_unvectorized(u.as_slice().unwrap(), v.as_slice().unwrap())
        }
    }
}

#[allow(dead_code)]
unsafe fn dot_f32x4(u: ArrayView1<f32>, v: ArrayView1<f32>) -> f32 {
    assert_eq!(u.len(), v.len());

    let mut u = u
        .as_slice()
        .expect("Cannot apply SIMD instructions on non-contiguous data.");
    let mut v = &v
        .as_slice()
        .expect("Cannot apply SIMD instructions on non-contiguous data.")[..u.len()];

    let mut sums = _mm_setzero_ps();

    while u.len() >= 4 {
        let ux4 = _mm_loadu_ps(&u[0] as *const f32);
        let vx4 = _mm_loadu_ps(&v[0] as *const f32);

        sums = _mm_add_ps(_mm_mul_ps(ux4, vx4), sums);

        u = &u[4..];
        v = &v[4..];
    }

    sums = _mm_hadd_ps(sums, sums);
    sums = _mm_hadd_ps(sums, sums);

    _mm_cvtss_f32(sums) + dot_unvectorized(u, v)
}

#[cfg(target_feature = "avx")]
unsafe fn dot_f32x8(u: ArrayView1<f32>, v: ArrayView1<f32>) -> f32 {
    assert_eq!(u.len(), v.len());

    let mut u = u
        .as_slice()
        .expect("Cannot apply SIMD instructions on non-contiguous data.");
    let mut v = &v
        .as_slice()
        .expect("Cannot apply SIMD instructions on non-contiguous data.")[..u.len()];

    let mut sums = _mm256_setzero_ps();

    while u.len() >= 8 {
        let ux8 = _mm256_loadu_ps(&u[0] as *const f32);
        let vx8 = _mm256_loadu_ps(&v[0] as *const f32);

        // Future: support FMA?
        // sums = _mm256_fmadd_ps(a, b, sums);

        sums = _mm256_add_ps(_mm256_mul_ps(ux8, vx8), sums);

        u = &u[8..];
        v = &v[8..];
    }

    sums = _mm256_hadd_ps(sums, sums);
    sums = _mm256_hadd_ps(sums, sums);

    // Sum sums[0..4] and sums[4..8].
    let sums = _mm_add_ps(_mm256_castps256_ps128(sums), _mm256_extractf128_ps(sums, 1));

    _mm_cvtss_f32(sums) + dot_unvectorized(u, v)
}

#[cfg(target_feature = "avx")]
unsafe fn kld_f32x8(u: ArrayView1<f32>, v: ArrayView1<f32>) -> f32 {
    assert_eq!(u.len(), v.len());

    let mut u = u
        .as_slice()
        .expect("Cannot apply SIMD instructions on non-contiguous data.");
    let mut v = &v
        .as_slice()
        .expect("Cannot apply SIMD instructions on non-contiguous data.")[..u.len()];

    let L = (u.len() - (u.len() % 16)) as f32;
    let mut sums = _mm256_setzero_ps();

    let mut w = Array1::zeros(u.len());
    for i in 0..u.len()/2 {
        w[2*i] = -u[i*2+1].ln();
        w[2*i+1] = v[i*2+1].ln();
    }

    let mut w = w.as_slice().unwrap();

    let mask = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);

    while u.len() >= 16 {
        // load 16xf32 from first vector
        let lo = _mm256_loadu_ps(&u[0] as *const f32);
        let hi = _mm256_loadu_ps(&u[8] as *const f32);

        // unpack interleaved mu and sigma into two seperate lanes
        let lo_grouped = _mm256_permutevar8x32_ps(lo, mask);
        let hi_grouped = _mm256_permutevar8x32_ps(hi, mask);

        let mu1 = _mm256_permute2f128_ps(lo_grouped, hi_grouped, 0 | (2 << 4));
        let sigma1 = _mm256_permute2f128_ps(lo_grouped, hi_grouped, 1 | (3 << 4));

        // load 16xf32 from second vector
        let lo = _mm256_loadu_ps(&v[0] as *const f32);
        let hi = _mm256_loadu_ps(&v[8] as *const f32);

        let lo_grouped = _mm256_permutevar8x32_ps(lo, mask);
        let hi_grouped = _mm256_permutevar8x32_ps(hi, mask);

        let mu2 = _mm256_permute2f128_ps(lo_grouped, hi_grouped, 0 | (2 << 4));
        let sigma2 = _mm256_permute2f128_ps(lo_grouped, hi_grouped, 1 | (3 << 4));

        // load 16xf32 from second vector
        let sigma_log1 = _mm256_loadu_ps(&w[0] as *const f32);
        let sigma_log2 = _mm256_loadu_ps(&w[8] as *const f32);

        let sigma_div = _mm256_div_ps(sigma1, sigma2);

        //dbg!(&mu1);
        //dbg!(&mu2);
        //dbg!(&sigma1);
        let mu_diff = _mm256_sub_ps(mu1, mu2);
        let mu_diff_sq = _mm256_mul_ps(mu_diff, mu_diff);
        let mu_diff_sq_norm = _mm256_div_ps(mu_diff_sq, sigma1);

        //dbg!(&mu_diff_sq_norm);

        sums = _mm256_add_ps(sigma_div, sums);
        sums = _mm256_add_ps(mu_diff_sq_norm, sums);
        sums = _mm256_add_ps(sigma_log1, sums);
        sums = _mm256_add_ps(sigma_log2, sums);

        // Future: support FMA?
        // sums = _mm256_fmadd_ps(a, b, sums);

        u = &u[16..];
        v = &v[16..];
        w = &w[16..];
    }


    sums = _mm256_hadd_ps(sums, sums);
    sums = _mm256_hadd_ps(sums, sums);

    // Sum sums[0..4] and sums[4..8].
    let sums = _mm_add_ps(_mm256_castps256_ps128(sums), _mm256_extractf128_ps(sums, 1));
    let res = (_mm_cvtss_f32(sums) - L/2.0)/2.0 + kld_unvectorized(u, v);

    f32::max(res, 0.0)
}

pub fn dot_unvectorized(u: &[f32], v: &[f32]) -> f32 {
    assert_eq!(u.len(), v.len());
    u.iter().zip(v).map(|(&a, &b)| a * b).sum()
}

pub fn kld_unvectorized(mut u: &[f32], mut v: &[f32]) -> f32 {
    assert!(u.len() % 2 == 0 && v.len() % 2 == 0);
    assert_eq!(u.len(), v.len());

    let l = u.len() / 2;

    let mut sum = -(l as f32);
    for _ in 0..l {
        sum += u[1] / v[1] - (u[1]/v[1]).ln() + (u[0]-v[0]).powf(2.0) / u[1];

        u = &u[2..];
        v = &v[2..];
    }

    sum / 2.0
}

#[allow(dead_code, clippy::float_cmp)]
unsafe fn scaled_add_f32x4(mut u: ArrayViewMut1<f32>, v: ArrayView1<f32>, a: f32) {
    assert_eq!(u.len(), v.len());

    let mut u = u
        .as_slice_mut()
        .expect("Cannot apply SIMD instructions on non-contiguous data.");
    let mut v = &v
        .as_slice()
        .expect("Cannot apply SIMD instructions on non-contiguous data.")[..u.len()];

    if a == 1f32 {
        while u.len() >= 4 {
            let mut ux4 = _mm_loadu_ps(&u[0] as *const f32);
            let vx4 = _mm_loadu_ps(&v[0] as *const f32);
            ux4 = _mm_add_ps(ux4, vx4);
            _mm_storeu_ps(&mut u[0] as *mut f32, ux4);
            u = &mut { u }[4..];
            v = &v[4..];
        }
    } else {
        let ax4 = _mm_set1_ps(a);

        while u.len() >= 4 {
            let mut ux4 = _mm_loadu_ps(&u[0] as *const f32);
            let vx4 = _mm_loadu_ps(&v[0] as *const f32);
            ux4 = _mm_add_ps(ux4, _mm_mul_ps(vx4, ax4));
            _mm_storeu_ps(&mut u[0] as *mut f32, ux4);
            u = &mut { u }[4..];
            v = &v[4..];
        }
    }

    scaled_add_unvectorized(u, v, a);
}

#[cfg(target_feature = "avx")]
unsafe fn scaled_add_f32x8(mut u: ArrayViewMut1<f32>, v: ArrayView1<f32>, a: f32) {
    assert_eq!(u.len(), v.len());

    let mut u = u
        .as_slice_mut()
        .expect("Cannot apply SIMD instructions on non-contiguous data.");
    let mut v = &v
        .as_slice()
        .expect("Cannot apply SIMD instructions on non-contiguous data.")[..u.len()];

    if a == 1f32 {
        while u.len() >= 8 {
            let mut ux8 = _mm256_loadu_ps(&u[0] as *const f32);
            let vx8 = _mm256_loadu_ps(&v[0] as *const f32);

            ux8 = _mm256_add_ps(ux8, vx8);

            _mm256_storeu_ps(&mut u[0] as *mut f32, ux8);
            u = &mut { u }[8..];
            v = &v[8..];
        }
    } else {
        let ax8 = _mm256_set1_ps(a);

        while u.len() >= 8 {
            let mut ux8 = _mm256_loadu_ps(&mut u[0] as *const f32);
            let vx8 = _mm256_loadu_ps(&v[0] as *const f32);

            ux8 = _mm256_add_ps(ux8, _mm256_mul_ps(vx8, ax8));

            _mm256_storeu_ps(&mut u[0] as *mut f32, ux8);
            u = &mut { u }[8..];
            v = &v[8..];
        }
    }

    scaled_add_unvectorized(u, v, a);
}

#[allow(clippy::float_cmp)]
fn scaled_add_unvectorized(u: &mut [f32], v: &[f32], a: f32) {
    assert_eq!(u.len(), v.len());

    if a == 1f32 {
        for i in 0..u.len() {
            u[i] += v[i];
        }
    } else {
        for i in 0..u.len() {
            u[i] += v[i] * a;
        }
    }
}

#[allow(dead_code)]
unsafe fn scale_f32x4(mut u: ArrayViewMut1<f32>, a: f32) {
    let mut u = u
        .as_slice_mut()
        .expect("Cannot apply SIMD instructions on non-contiguous data.");

    let ax4 = _mm_set1_ps(a);

    while u.len() >= 4 {
        let mut ux4 = _mm_loadu_ps(&u[0] as *const f32);
        ux4 = _mm_mul_ps(ux4, ax4);
        _mm_storeu_ps(&mut u[0] as *mut f32, ux4);
        u = &mut { u }[4..];
    }

    scale_unvectorized(u, a);
}

#[cfg(target_feature = "avx")]
unsafe fn scale_f32x8(mut u: ArrayViewMut1<f32>, a: f32) {
    let mut u = u
        .as_slice_mut()
        .expect("Cannot apply SIMD instructions on non-contiguous data.");

    let ax8 = _mm256_set1_ps(a);

    while u.len() >= 8 {
        let mut ux8 = _mm256_loadu_ps(&mut u[0] as *const f32);
        ux8 = _mm256_mul_ps(ux8, ax8);
        _mm256_storeu_ps(&mut u[0] as *mut f32, ux8);
        u = &mut { u }[8..];
    }

    scale_unvectorized(u, a);
}

fn scale_unvectorized(u: &mut [f32], a: f32) {
    for c in u {
        *c *= a;
    }
}

/// Normalize a vector by its l2 norm.
///
/// The l2 norm is returned.
#[inline]
pub fn l2_normalize(v: ArrayViewMut1<f32>) -> f32 {
    let norm = dot(v.view(), v.view()).sqrt();
    scale(v, 1.0 / norm);
    norm
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    use crate::util::{all_close, array_all_close, close};

    use super::{
        dot_f32x4, dot_unvectorized, l2_normalize, scale_f32x4, scale_unvectorized,
        scaled_add_f32x4, scaled_add_unvectorized,
    };

    #[cfg(target_feature = "avx")]
    use super::{dot_f32x8, scale_f32x8, scaled_add_f32x8};

    #[test]
    fn add_unvectorized_test() {
        let u = &mut [1., 2., 3., 4., 5.];
        let v = &[5., 3., 3., 2., 1.];
        scaled_add_unvectorized(u, v, 1.0);
        assert!(all_close(u, &[6.0, 5.0, 6.0, 6.0, 6.0], 1e-5));
    }

    #[test]
    fn add_f32x4_test() {
        let mut u = Array1::random((102,), Uniform::new_inclusive(-1.0, 1.0));
        let v = Array1::random((102,), Uniform::new_inclusive(-1.0, 1.0));
        let mut check = u.clone();
        scaled_add_unvectorized(check.as_slice_mut().unwrap(), v.as_slice().unwrap(), 1.0);
        unsafe { scaled_add_f32x4(u.view_mut(), v.view(), 1.0) };
        assert!(array_all_close(check.view(), u.view(), 1e-5));
    }

    #[test]
    #[cfg(target_feature = "avx")]
    fn add_f32x8_test() {
        let mut u = Array1::random((102,), Uniform::new_inclusive(-1.0, 1.0));
        let v = Array1::random((102,), Uniform::new_inclusive(-1.0, 1.0));
        let mut check = u.clone();
        scaled_add_unvectorized(check.as_slice_mut().unwrap(), v.as_slice().unwrap(), 1.0);
        unsafe { scaled_add_f32x8(u.view_mut(), v.view(), 1.0) };
        assert!(array_all_close(check.view(), u.view(), 1e-5));
    }

    #[test]
    fn dot_f32x4_test() {
        let u = Array1::random((102,), Uniform::new_inclusive(-1.0, 1.0));
        let v = Array1::random((102,), Uniform::new_inclusive(-1.0, 1.0));
        assert!(close(
            unsafe { dot_f32x4(u.view(), v.view()) },
            dot_unvectorized(u.as_slice().unwrap(), v.as_slice().unwrap()),
            1e-5
        ));
    }

    #[test]
    #[cfg(target_feature = "avx")]
    fn dot_f32x8_test() {
        let u = Array1::random((102,), Uniform::new_inclusive(-1.0, 1.0));
        let v = Array1::random((102,), Uniform::new_inclusive(-1.0, 1.0));
        assert!(close(
            unsafe { dot_f32x8(u.view(), v.view()) },
            dot_unvectorized(u.as_slice().unwrap(), v.as_slice().unwrap()),
            1e-5
        ));
    }

    #[test]
    fn dot_unvectorized_test() {
        let u = [1f32, -2f32, -3f32];
        let v = [2f32, 4f32, -2f32];
        let w = [-1f32, 3f32, 2.5f32];

        assert!(close(dot_unvectorized(&u, &v), 0f32, 1e-5));
        assert!(close(dot_unvectorized(&u, &w), -14.5f32, 1e-5));
        assert!(close(dot_unvectorized(&v, &w), 5f32, 1e-5));
    }

    #[test]
    fn scaled_add_unvectorized_test() {
        let u = &mut [1., 2., 3., 4., 5.];
        let v = &[5., 3., 3., 2., 1.];
        scaled_add_unvectorized(u, v, 0.5);
        assert!(all_close(u, &[3.5, 3.5, 4.5, 5.0, 5.5], 1e-5));
    }

    #[test]
    fn scaled_add_f32x4_test() {
        let mut u = Array1::random((102,), Uniform::new_inclusive(-1.0, 1.0));
        let v = Array1::random((102,), Uniform::new_inclusive(-1.0, 1.0));
        let mut check = u.clone();
        scaled_add_unvectorized(check.as_slice_mut().unwrap(), v.as_slice().unwrap(), 2.5);
        unsafe { scaled_add_f32x4(u.view_mut(), v.view(), 2.5) };
        assert!(array_all_close(check.view(), u.view(), 1e-5));
    }

    #[test]
    #[cfg(target_feature = "avx")]
    fn scaled_add_f32x8_test() {
        let mut u = Array1::random((102,), Uniform::new_inclusive(-1.0, 1.0));
        let v = Array1::random((102,), Uniform::new_inclusive(-1.0, 1.0));
        let mut check = u.clone();
        scaled_add_unvectorized(check.as_slice_mut().unwrap(), v.as_slice().unwrap(), 2.5);
        unsafe { scaled_add_f32x8(u.view_mut(), v.view(), 2.5) };
        assert!(array_all_close(check.view(), u.view(), 1e-5));
    }

    #[test]
    fn scale_unvectorized_test() {
        let s = &mut [1., 2., 3., 4., 5.];
        scale_unvectorized(s, 0.5);
        assert!(all_close(s, &[0.5, 1.0, 1.5, 2.0, 2.5], 1e-5));
    }

    #[test]
    fn scale_f32x4_test() {
        let mut u = Array1::random((102,), Uniform::new_inclusive(-1.0, 1.0));
        let mut check = u.clone();
        scale_unvectorized(check.as_slice_mut().unwrap(), 2.);
        unsafe { scale_f32x4(u.view_mut(), 2.) };
        assert!(array_all_close(check.view(), u.view(), 1e-5));
    }

    #[test]
    #[cfg(target_feature = "avx")]
    fn scale_f32x8_test() {
        let mut u = Array1::random((102,), Uniform::new_inclusive(-1.0, 1.0));
        let mut check = u.clone();
        scale_unvectorized(check.as_slice_mut().unwrap(), 2.);
        unsafe { scale_f32x8(u.view_mut(), 2.) };
        assert!(array_all_close(check.view(), u.view(), 1e-5));
    }

    #[test]
    fn l2_normalize_test() {
        let mut u = Array1::from(vec![1., -2., -1., 3., -3., 1.]);
        assert!(close(l2_normalize(u.view_mut()), 5., 1e-5));
        assert!(all_close(
            &[0.2, -0.4, -0.2, 0.6, -0.6, 0.2],
            u.as_slice().unwrap(),
            1e-5
        ));

        // Normalization should result in a unit vector.
        let mut v = Array1::random((102,), Uniform::new_inclusive(-1.0, 1.0));
        l2_normalize(v.view_mut());
        assert!(close(v.dot(&v).sqrt(), 1.0, 1e-5));
    }
}
