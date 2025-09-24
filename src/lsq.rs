use ndarray::{Array1, Array2};
use faer::prelude::*;
use faer::Side;

pub struct LsqOptions {
    pub lambda: f64,
    pub voltage_limit: Option<f64>,
}

impl Default for LsqOptions {
    fn default() -> Self {
        Self { lambda: 1e-3, voltage_limit: None }
    }
}

// one ridge solve using normal equations with adaptive lambda, no recursion
fn solve_once(a: &Array2<f64>, b: &Array1<f64>, mut lam: f64) -> (Vec<f64>, f64) {
    let m = a.nrows();
    let n = a.ncols();

    let amat = Mat::from_fn(m, n, |i, j| a[[i, j]]);
    let at = amat.transpose();
    let atb = &at * Mat::from_fn(m, 1, |i, _| b[i]);

    loop {
        let mut ata = &at * &amat;
        for i in 0..n { *ata.get_mut(i, i) += lam; }

        match ata.cholesky(Side::Lower) {
            Ok(ch) => {
                let x = ch.solve(atb.as_ref()).col(0).to_owned();
                let v = (0..n).map(|i| *x.get(i)).collect::<Vec<f64>>();
                return (v, lam);
            }
            Err(_) => {
                lam *= 10.0;
                if lam > 1e6 {
                    // give up gracefully
                    return (vec![0.0; n], lam);
                }
            }
        }
    }
}

// non recursive ridge least squares with a few fixed refinements
pub fn tikhonov(a: &Array2<f64>, b: &Array1<f64>, opts: &LsqOptions) -> Vec<f64> {
    // first solve
    let start_lam = if opts.lambda > 0.0 { opts.lambda } else { 1e-3 };
    let (mut v, mut used_lam) = solve_once(a, b, start_lam);

    // optional symmetric clamp
    if let Some(vlim) = opts.voltage_limit {
        for vi in v.iter_mut() {
            if *vi > vlim { *vi = vlim; }
            if *vi < -vlim { *vi = -vlim; }
        }
    }

    // compute residual r = b - A v
    let m = a.nrows();
    let n = a.ncols();
    let mut r = Array1::<f64>::zeros(m);
    let mut tmp = 0.0;
    for i in 0..m {
        tmp = 0.0;
        for j in 0..n { tmp += a[[i, j]] * v[j]; }
        r[i] = b[i] - tmp;
    }

    // two non recursive residual corrections with a small but bounded ridge
    let refine_lam = used_lam.min(1e-6).max(1e-9);
    for _ in 0..2 {
        let (dv, _) = solve_once(a, &r, refine_lam);
        for j in 0..n { v[j] += dv[j]; }

        // update residual
        for i in 0..m {
            tmp = 0.0;
            for j in 0..n { tmp += a[[i, j]] * v[j]; }
            r[i] = b[i] - tmp;
        }
    }

    v
}
