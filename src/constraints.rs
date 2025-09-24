use ndarray::{Array1, Array2};
use crate::basis::TrapModel;
use crate::types::{Vec3, Waypoint};

/// Build A v = b for one waypoint
/// rows are gx = -grf_x, gy = -grf_y, gz = -grf_z to enforce grad total zero
/// plus axial curvature equals target value
/// we treat axial_dir as unit and enforce u^T H_total u equals target curvature
pub fn build_constraints(model: &TrapModel, wp: &Waypoint, charge_q: f64, mass_m: f64)
    -> (Array2<f64>, Array1<f64>)
{
    let n = model.n_electrodes();
    let mut a = Array2::<f64>::zeros((4, n));
    let mut b = Array1::<f64>::zeros(4);

    let grf = model.rf.grad(wp.r);
    for j in 0..n {
        let bj = &model.dc[j];
        let gj = bj.grad(wp.r);
        a[[0, j]] = gj.x;
        a[[1, j]] = gj.y;
        a[[2, j]] = gj.z;
    }
    b[0] = -grf.x;
    b[1] = -grf.y;
    b[2] = -grf.z;

    let u = wp.axial_dir;
    let hrf = model.rf.hess(wp.r);
    let target_curv = mass_m * wp.omega_axial * wp.omega_axial / charge_q;

    // weight curvature higher so the solver honors it tightly
    let w_curv = 1e3;
    for j in 0..n {
        let hj = model.dc[j].hess(wp.r);
        a[[3, j]] = w_curv * hj.quad(u);
    }
    b[3] = w_curv * (target_curv - hrf.quad(u));

    (a, b)
}

