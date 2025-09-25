use ndarray::{Array1, Array2};
use rayon::prelude::*;
use sprs::{CsMat, TriMat};

// ---- helpers to convert and do matvecs on CSR/CSC ----
fn dense_to_csr(a: &Array2<f64>) -> CsMat<f64> {
    let m = a.nrows();
    let n = a.ncols();
    let mut tri = TriMat::new((m, n));
    for i in 0..m {
        for j in 0..n {
            let v = a[[i, j]];
            if v != 0.0 { tri.add_triplet(i, j, v); }
        }
    }
    tri.to_csr()
}

// y = A x  (CSR row-wise)
fn spmv_csr(a: &CsMat<f64>, x: &[f64]) -> Vec<f64> {
    let m = a.rows();
    (0..m).into_par_iter().map(|i| {
        let mut s = 0.0;
        if let Some(row) = a.outer_view(i) {
            for (j, v) in row.iter() { s += v * x[j]; }
        }
        s
    }).collect()
}

// y = A^T x  (CSC col-wise)
fn spmtv_csc(at: &CsMat<f64>, x: &[f64]) -> Vec<f64> {
    // at is CSC for A (i.e., CSR for A^T), but outer_view(j) iterates a column j of A
    let n = at.cols();
    (0..n).into_par_iter().map(|j| {
        let mut s = 0.0;
        if let Some(col) = at.outer_view(j) {
            for (i, v) in col.iter() { s += v * x[i]; }
        }
        s
    }).collect()
}

// BLAS-like tiny helpers
fn nrm2(x: &[f64]) -> f64 { x.iter().map(|t| t*t).sum::<f64>().sqrt() }
fn scal(x: &mut [f64], a: f64) { x.iter_mut().for_each(|t| *t *= a); }
fn axpy(y: &mut [f64], x: &[f64], a: f64) { for i in 0..y.len() { y[i] += a * x[i]; } }

pub struct LsqOptions {
    pub lambda: f64,                 // Tikhonov (λ >= 0)
    pub voltage_limit: Option<f64>,  // symmetric clamp
    pub iters: usize,                // LSQR iterations
    pub tol: f64,                    // early stop on |phi_bar|
}

impl Default for LsqOptions {
    fn default() -> Self {
        Self { lambda: 1e-2, voltage_limit: Some(5.0), iters: 400, tol: 1e-10 }
    }
}

/// LSQR on augmented matrix [A; sqrt(λ) I] with rhs [b; 0]
pub fn tikhonov(a_dense: &Array2<f64>, b: &Array1<f64>, opts: &LsqOptions) -> Vec<f64> {
    let a_csr = dense_to_csr(a_dense);
    let a_csc = a_csr.to_csc();            // for A^T * x
    let m0 = a_csr.rows();
    let n  = a_csr.cols();

    let sqrt_lam = opts.lambda.max(0.0).sqrt();

    // u is size m0 + n (augmented rows)
    // start with u = [b; 0]
    let mut u = Vec::with_capacity(m0 + n);
    u.extend_from_slice(b.as_slice().unwrap());
    u.resize(m0 + n, 0.0);

    // β = ||u||
    let mut beta = nrm2(&u);
    if beta != 0.0 { scal(&mut u, 1.0 / beta); }

    // v = A^T u_top + sqrt(λ) * u_bottom
    let (u_top, u_bot) = u.split_at(m0);
    let mut v = spmtv_csc(&a_csc, u_top);
    for j in 0..n { v[j] += sqrt_lam * u_bot[j]; }

    // α = ||v||, normalize
    let mut alpha = nrm2(&v);
    if alpha != 0.0 { scal(&mut v, 1.0 / alpha); }

    let mut w = v.clone();
    let mut x = vec![0.0; n];

    let mut phi_bar = beta;
    let mut rho_bar = alpha;

    for _ in 0..opts.iters {
        // u = [A v; sqrt(λ) v] - α [u_top; u_bot]
        let mut Av = spmv_csr(&a_csr, &v);
        // append sqrt(λ) v
        Av.extend(v.iter().map(|&t| sqrt_lam * t));
        // subtract α u
        axpy(&mut Av, &u, -alpha);
        u = Av;

        beta = nrm2(&u);
        if beta != 0.0 { scal(&mut u, 1.0 / beta); }

        // v = [A^T u_top + sqrt(λ) u_bot] - β v
        let (u_top2, u_bot2) = u.split_at(m0);
        let mut Atu = spmtv_csc(&a_csc, u_top2);
        for j in 0..n { Atu[j] += sqrt_lam * u_bot2[j]; }
        axpy(&mut Atu, &v, -beta);

        alpha = nrm2(&Atu);
        if alpha != 0.0 { scal(&mut Atu, 1.0 / alpha); }
        let v_new = Atu;

        // orthogonal update
        let rho   = (rho_bar * rho_bar + beta * beta).sqrt();
        let c     = rho_bar / rho;
        let s     = beta / rho;
        let theta = s * alpha;
        rho_bar   = -c * alpha;
        let phi   = c * phi_bar;
        phi_bar   = s * phi_bar;

        // x and w
        for j in 0..n {
            x[j] += (phi / rho) * w[j];
            w[j]  = v_new[j] - (theta / rho) * w[j];
        }

        v = v_new;

        if phi_bar.abs() < opts.tol { break; }
    }

    if let Some(vlim) = opts.voltage_limit {
        for xi in x.iter_mut() {
            if *xi > vlim { *xi = vlim; }
            if *xi < -vlim { *xi = -vlim; }
        }
    }
    x
}


/* helpers */
fn norm(x: &[f64]) -> f64 { x.iter().map(|t| t*t).sum::<f64>().sqrt() }

