use crate::basis::TrapModel;
use crate::types::{Vec3, Hess};

/// eigenvalues of symmetric 3 by 3 given as Hess fields
pub fn eigenvalues(h: Hess) -> [f64; 3] {
    // construct full matrix and use analytic symmetric eig via cubic
    // for robustness here use a simple numeric reduction
    // small 3 by 3 symmetric Jacobi iterations
    let mut a = [
        [h.xx, h.xy, h.xz],
        [h.xy, h.yy, h.yz],
        [h.xz, h.yz, h.zz],
    ];
    let mut v = [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]];
    for _ in 0..25 {
        // find largest off diagonal
        let mut p = 0usize; let mut q = 1usize;
        let mut maxv = (a[0][1]).abs();
        let cands = [ (0,2), (1,2) ];
        for &(i,j) in &cands {
            if a[i][j].abs() > maxv { maxv = a[i][j].abs(); p = i; q = j; }
        }
        if maxv < 1e-12 { break; }
        let app = a[p][p]; let aqq = a[q][q]; let apq = a[p][q];
        let phi = 0.5 * ((aqq - app)/(apq)).atan();
        let c = phi.cos(); let s = phi.sin();

        // rotate A
        for k in 0..3 {
            let aik = a[p][k]; let aqk = a[q][k];
            a[p][k] = c*aik - s*aqk;
            a[q][k] = s*aik + c*aqk;
        }
        for k in 0..3 {
            let akp = a[k][p]; let akq = a[k][q];
            a[k][p] = c*akp - s*akq;
            a[k][q] = s*akp + c*akq;
        }
        // rotate V
        for k in 0..3 {
            let vkp = v[k][p]; let vkq = v[k][q];
            v[k][p] = c*vkp - s*vkq;
            v[k][q] = s*vkp + c*vkq;
        }
    }
    [a[0][0], a[1][1], a[2][2]]
}

/// given total Hessian and charge q and mass m, return secular freqs
pub fn secular_freqs(h: Hess, q: f64, m: f64) -> [f64; 3] {
    let ev = eigenvalues(h);
    // m omega^2 = q * lambda
    let mut w = [0.0; 3];
    for i in 0..3 {
        let val = (q*ev[i]/m).max(0.0);
        w[i] = val.sqrt();
    }
    w
}
