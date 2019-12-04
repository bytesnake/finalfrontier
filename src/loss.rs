use std::iter::FromIterator;
use ndarray::prelude::*;
use packed_simd::{f32x16, f32x8, f32x4};

use crate::util;
use crate::vec_simd::dot;

/// Absolute activations to round in logistic regression.
///
/// Since the logistic function is asymptotic, there is always (a small)
/// gradient for larger activations. As a result, optimization of logistic
/// regression will not converge without e.g. regularization. In the
/// training of embeddings, this has the result of ever-increasing weights
/// (amplified by the optimization two vectors).
///
/// A simpler solution than regularization is to round the output of the
/// logistic function to 0 (negative activation) or 1 (positive activiation)
/// for large activations, to kill gradient.
///
/// This constant controls at what activation the logistic function should
/// round.
const LOGISTIC_ROUND_ACTIVATION: f32 = 10.0;

/// Return the loss and gradient of the co-occurence classification.
///
/// This function returns the negative log likelihood and gradient of
/// a training instance using the probability function *P(1|x) =
/// σ(u·v)*. `u` and `v` are word embeddings and `label` is the
/// target label, where a label of `1` means that the words co-occur
/// and a label of `0` that they do not.
///
/// This model is very similar to logistic regression, except that we
/// optimize both u and v.
///
/// The loss is as follows (y is used as the label):
///
/// log(P(y|x)) =
/// y log(P(1|x)) + (1-y) log(P(0|x)) =
/// y log(P(1|x)) + (1-y) log(1 - P(1|x)) =
/// y log(σ(u·v)) + (1-y) log(1 - σ(u·v)) =
/// y log(σ(u·v)) + (1-y) log(σ(-u·v))
///
/// We can simplify the first term:
///
/// y log(σ(u·v)) =
/// y log(1/e^{-u·v}) =
/// -y log(e^{-u·v})
///
/// Then we find the derivative with respect to v_1:
///
/// ∂/∂v_1 -y log(e^{-u·v}) =
/// -y σ(u·v) ∂/∂v_1(e^{-u·v}) =
/// -y σ(u·v) e^{-u·v} -u_1 =
/// y σ(-u·v) u_1 =
/// y (1 - σ(u·v)) u_1 =
/// (y - yσ(u·v)) u_1
///
/// Iff y = 1, then:
///
/// 1 - σ(u·v)
///
/// For the second term above, we also find the derivative:
///
/// ∂/∂v_1 -(1 - y) log(e^{u·v}) =
/// -(1 - y) σ(-u·v) ∂/∂v_1(e^{u·v}) =
/// -(1 - y) σ(-u·v) e^{u·v} ∂/∂v_1 u·v=
/// -(1 - y) σ(-u·v) e^{u·v} u_1 =
/// -(1 - y) σ(u·v) u_1 =
/// (-σ(u·v) + yσ(u·v)) u_1
///
/// When y = 0 then:
///
/// -σ(u·v)u_1
///
/// Combining both, the partial derivative of v_1 is: y - σ(u·v)u_1
///
/// We return y - σ(u·v) as the gradient, so that the caller can compute
/// the gradient for all components of u and v.
pub fn log_logistic_loss(u: ArrayView1<f32>, v: ArrayView1<f32>, label: bool) -> (f32, f32) {
    let dp = dot(u, v);
    let lf = logistic_function(dp);
    let grad = (label as usize) as f32 - lf;
    let loss = if label {
        -util::safe_ln(lf)
    } else {
        -util::safe_ln(1.0 - lf)
    };

    (loss, grad)
}

/// Compute the logistic function.
///
/// **σ(a) = 1 / (1 + e^{-a})**
fn logistic_function(a: f32) -> f32 {
    if a > LOGISTIC_ROUND_ACTIVATION {
        1.0
    } else if a < -LOGISTIC_ROUND_ACTIVATION {
        0.0
    } else {
        1.0 / (1.0 + (-a).exp())
    }
}

pub fn kld_loss(u: ArrayView1<f32>, v: ArrayView1<f32>, label: bool) -> (f32, Array1<f32>, Array1<f32>) {
    let mut kld = kld(u,v);

    let l = u.len() / 2;

    if kld.is_nan() || !kld.is_finite() {
        dbg!(&u);
        dbg!(&v);
        panic!("Nan");
    }

    let elu = |val:f32| -> f32 { if val > 0.0 { val } else { val.exp() + 1e-4 }};
    let elud = |val: f32| -> f32 { if val > 0.0 { 1.0 } else { val.exp() }};

    let delta1_iter = (0..u.len()).map(|i| {
        if i < l {
            (u[i] - v[i]) / elu(u[i+l])
        } else {
            elud(u[i]) * (elu(v[i]).recip() - 
                          elu(u[i]).recip() - 
                          ((u[i-l]-v[i-l]) / elu(u[i])).powf(2.0)) / 2.0
        }
    });

    let delta2_iter = (0..u.len()).map(|i| {
        if i < l {
            (v[i] - u[i]) / elu(u[i+l])
        } else {
            elud(v[i]) * (-elu(u[i])/(elu(v[i]).powf(2.0)) + elu(v[i]).recip())/2.0
        }
    });

    dbg!(&kld);
    if label {
        (
            kld.powf(2.0),
            Array1::from_iter(delta1_iter.map(|x| 2.0*x*kld)),
            Array1::from_iter(delta2_iter.map(|x| 2.0*x*kld))
        )
    } else {
        (
            (-kld).exp(),
            Array1::from_iter(delta1_iter.map(|x| -(-kld).exp()*x)),
            Array1::from_iter(delta2_iter.map(|x| -(-kld).exp()*x))
        )
    }
}

/*pub fn kld(u: ArrayView1<f32>, v: ArrayView1<f32>) -> f32 {
    let l = u.len() / 2;

    let mut sigma_ratio = &u.slice(s![l..]) / &v.slice(s![l..]);
    let trace_fac = sigma_ratio.sum();
    sigma_ratio.mapv_inplace(|x| util::safe_ln(x));

    let log_det = sigma_ratio.sum();
    let sq_diff = (&u.slice(s![..l]) - &v.slice(s![..l])).mapv_into(|x| x*x);
    let mu_diff_sq = (sq_diff / &u.slice(s![l..])).sum();
         
    return 0.5 * (trace_fac + mu_diff_sq - l as f32 - log_det);
}*/

pub fn kld(a: ArrayView1<f32>, b: ArrayView1<f32>) -> f32 {
	let init_vec = move |b: ArrayView1<f32>, off: usize| -> f32x16 {
            unsafe { f32x16::new(*b.uget(off),*b.uget(off+1),*b.uget(off+2),*b.uget(off+3),*b.uget(off+4),*b.uget(off+5),*b.uget(off+6),*b.uget(off+7),
                                *b.uget(off+8),*b.uget(off+9),*b.uget(off+10),*b.uget(off+11),*b.uget(off+12),*b.uget(off+13),*b.uget(off+14),*b.uget(off+15)) }
        };

    if a.len() != 32 || b.len() != 32 {
        panic!("Only support dimension of 32");
    }

	let sigma1 = init_vec(a, 16);
	let mu1 = init_vec(a, 0);

	let sigma2 = init_vec(b, 16);
	let mu2 = init_vec(b, 0);

	let div = sigma1 / sigma2;
	let trace_fac = div.sum();
	let log_det = ((div + 1e-14).ln()).sum();

	let mu_diff_sq = mu1-mu2;
	let mu_diff_sq = mu_diff_sq*mu_diff_sq;
	let mu_diff_sq = (mu_diff_sq / sigma1).sum();

	let sum = f32x4::new(trace_fac, mu_diff_sq, -16.0, -log_det);

    0.5 * sum.sum()
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;

    use crate::util::{all_close, close};

    use super::{log_logistic_loss, logistic_function};

    #[test]
    fn logistic_function_test() {
        let activations = &[
            -11.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 11.0,
        ];
        let outputs: Vec<_> = activations.iter().map(|&a| logistic_function(a)).collect();
        assert!(all_close(
            &[
                0.0, 0.00669, 0.01799, 0.04743, 0.11920, 0.26894, 0.5, 0.73106, 0.88080, 0.95257,
                0.982014, 0.99331, 1.0
            ],
            outputs.as_slice(),
            1e-5
        ));
    }

    #[test]
    fn log_logistic_loss_test() {
        let a = Array1::from_shape_vec((6,), vec![1., 1., 1., 0., 0., 0.]).unwrap();
        let a_orth = Array1::from_shape_vec((6,), vec![0., 0., 0., 1., 1., 1.]).unwrap();
        let a_opp = Array1::from_shape_vec((6,), vec![-1., -1., -1., 0., 0., 0.]).unwrap();

        let (loss, gradient) = log_logistic_loss(a.view(), a_orth.view(), true);
        assert!(close(loss, 0.69312, 1e-5));
        assert!(close(gradient, 0.5, 1e-5));

        let (loss, gradient) = log_logistic_loss(a.view(), a_orth.view(), false);
        assert!(close(loss, 0.69312, 1e-5));
        assert!(close(gradient, -0.5, 1e-5));

        let (loss, gradient) = log_logistic_loss(a.view(), a.view(), true);
        assert!(close(loss, 0.04858, 1e-5));
        assert!(close(gradient, 0.04742, 1e-5));

        let (loss, gradient) = log_logistic_loss(a.view(), a_opp.view(), false);
        assert!(close(loss, 0.04858, 1e-5));
        assert!(close(gradient, -0.04743, 1e-5));

        let (loss, gradient) = log_logistic_loss(a.view(), a_opp.view(), true);
        assert!(close(loss, 3.04838, 1e-5));
        assert!(close(gradient, 0.95257, 1e-5));
    }
}
