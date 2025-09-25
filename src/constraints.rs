use ndarray::{Array1, Array2};
use crate::basis::TrapModel;
use crate::types::{Vec3, Waypoint};

pub fn build_constraints(
    model: &TrapModel,
    wp: &Waypoint,
    charge_q: f64,
    mass_m: f64
) -> (Array2<f64>, Array1<f64>) {
    let n = model.n_electrodes();

    // 3 grad rows + 1 axial curvature + 2 radial floors = 6 rows
    let mut a = Array2::<f64>::zeros((6, n));
    let mut b = Array1::<f64>::zeros(6);

    // gradient = 0 at waypoint
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

    // axial curvature target (heavy weight)
    let u = wp.axial_dir;                      // assumed unit z
    let hrf = model.rf.hess(wp.r);
    let target_ax = mass_m * wp.omega_axial * wp.omega_axial / charge_q;
    let w_ax = 1e3;                            // heavy but not singular

    for j in 0..n {
        let hj = model.dc[j].hess(wp.r);
        a[[3, j]] = w_ax * hj.quad(u);
    }
    b[3] = w_ax * (target_ax - hrf.quad(u));

    // radial floors: keep H_xx and H_yy >= 0.2 * RF radial curvature
    let ex = Vec3 { x: 1.0, y: 0.0, z: 0.0 };
    let ey = Vec3 { x: 0.0, y: 1.0, z: 0.0 };
    let w_rad = 50.0;

    let floor_x = 0.2 * model.rf.hess(wp.r).quad(ex);
    for j in 0..n {
        let hj = model.dc[j].hess(wp.r);
        a[[4, j]] = w_rad * hj.quad(ex);
    }
    b[4] = w_rad * (floor_x - hrf.quad(ex));

    let floor_y = 0.2 * model.rf.hess(wp.r).quad(ey);
    for j in 0..n {
        let hj = model.dc[j].hess(wp.r);
        a[[5, j]] = w_rad * hj.quad(ey);
    }
    b[5] = w_rad * (floor_y - hrf.quad(ey));

    (a, b)
}
