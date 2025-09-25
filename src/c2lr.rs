use crate::basis::TrapModel;
use crate::constraints::build_constraints;
use crate::lsq::{tikhonov, LsqOptions};
use crate::types::{Waypoint, Result, IonwaveError};
use rayon::prelude::*;

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
    let pair = model.c2lr_pair.unwrap_or((0, 0));

    let sols: Vec<Vec<f64>> = waypoints.par_iter().map(|wp| {
        let (mut a, b) = build_constraints(model, wp, q_charge, mass);
        if left {
            let (i, j) = pair;
            for row in 0..a.nrows() {
    let tmp = a[[row, i]];
    a[[row, i]] = a[[row, j]];
    a[[row, j]] = tmp;
}
        }
        tikhonov(&a, &b, opts)
    }).collect();

    Ok(sols)
}
