#[cfg(test)]
mod tests {
    use ndarray::{Array2, Array1};
    use crate::lsq::{tikhonov, LsqOptions};

    #[test]
    fn lsqr_fits_small_system() {
        let a = Array2::from_shape_vec((4, 2), vec![
            1.0, 0.0,
            1.0, 1.0,
            1.0, 2.0,
            1.0, 3.0,
        ]).unwrap();
        let b = Array1::from(vec![1.0, 2.0, 2.9, 4.1]);  // y ≈ 1.03 + 1.02 x
        let opts = LsqOptions { lambda: 1e-3, voltage_limit: None, iters: 200, tol: 1e-10 };
        let v = tikhonov(&a, &b, &opts);
        assert!((v[0] - 1.03).abs() < 0.05);
        assert!((v[1] - 1.02).abs() < 0.05);
    }
}
