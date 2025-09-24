use crate::basis::TrapModel;
use crate::constraints::build_constraints;
use crate::lsq::{tikhonov, LsqOptions};
use crate::types::{Waypoint, Result, IonwaveError};

/// Solve for a set of waypoints and return [n_wp][n_el] voltages
/// If left is true, swap the two columns that form the C2LR pair
pub fn solve_waveform(
    model: &TrapModel,
    waypoints: &[Waypoint],
    q_charge: f64,
    mass: f64,
    left: bool,
    opts: &LsqOptions,
) -> Result<Vec<Vec<f64>>> {
    let n_el = model.n_electrodes();
    if n_el == 0 { return Err(IonwaveError::InvalidInput("no electrodes".to_string())); }

    let pair = if let Some(p) = model.c2lr_pair { p } else { (0, 0) };

    let mut sols = Vec::with_capacity(waypoints.len());
    for wp in waypoints {
        let (mut a, b) = build_constraints(model, wp, q_charge, mass);
        if left {
            // swap the two rail columns to mirror the primitive
            let (i, j) = pair;
            for row in 0..a.nrows() {
                let tmp = a[[row, i]];
                a[[row, i]] = a[[row, j]];
                a[[row, j]] = tmp;
            }
        }
        let v = tikhonov(&a, &b, opts);
        sols.push(v);
    }
    Ok(sols)
}
