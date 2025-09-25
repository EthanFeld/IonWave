// Shared helpers for tests

use ionwave::basis::{TrapModel, RfPseudo, GaussianBasis, PotentialBasis};
use ionwave::types::Vec3;

pub fn build_model(omega_axial: f64) -> TrapModel {
    // physical constants
    let q = 1.602e-19;
    let m = 2.84e-25;

    // rf curvature from target
    let target_curv = m * omega_axial * omega_axial / q;
    let rf = RfPseudo { kr: 1.0e10, kz: target_curv };

    // gentle dc basis set
    let mut dc: Vec<Box<dyn PotentialBasis>> = Vec::new();
    let sigma = 40e-6;
    let dc_scale = 0.002;
    let z_positions: Vec<f64> = (-4..=4).map(|k| k as f64 * 63e-6).collect();
    for (idx, zc) in z_positions.iter().enumerate() {
        let left = GaussianBasis { center: Vec3 { x: -50e-6, y: 0.0, z: *zc }, sigma, scale: dc_scale };
        let right = GaussianBasis { center: Vec3 { x:  50e-6, y: 0.0, z: *zc }, sigma, scale: dc_scale };
        dc.push(Box::new(left));
        dc.push(Box::new(right));
        if idx % 2 == 0 {
            let center = GaussianBasis { center: Vec3 { x: 0.0, y: 0.0, z: *zc + 0.5*63e-6 }, sigma, scale: 0.8 * dc_scale };
            dc.push(Box::new(center));
        }
    }

    TrapModel::new(Box::new(rf), dc, Some((0, 1)))
}

pub fn make_waypoints(n_wp: usize, omega_axial: f64) -> Vec<ionwave::types::Waypoint> {
    let z0 = 0.0;
    let dz = 63e-6;
    let axial_dir = Vec3 { x: 0.0, y: 0.0, z: 1.0 };
    (0..n_wp).map(|i| {
        let t = i as f64 / (n_wp as f64 - 1.0);
        let z = z0 + t * dz;
        ionwave::types::Waypoint { r: Vec3 { x: 0.0, y: 0.0, z }, omega_axial, axial_dir }
    }).collect()
}

pub fn freq_along_axis(h: ionwave::types::Hess, u: Vec3, q: f64, m: f64) -> f64 {
    let curv = h.quad(u);
    let w = (q * curv / m).max(0.0).sqrt();
    w
}
