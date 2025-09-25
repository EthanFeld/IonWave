mod common;
use common::{build_model, make_waypoints, freq_along_axis};
use ionwave::c2lr::solve_waveform;
use ionwave::lsq::LsqOptions;
use ionwave::types::Vec3;

#[test]
fn c2lr_swap_holds() {
    let omega_axial = 2.0 * std::f64::consts::PI * 1.5e6;
    let model = build_model(omega_axial);
    let wps = make_waypoints(9, omega_axial);

    let q = 1.602e-19;
    let m = 2.84e-25;

    let opts = LsqOptions { lambda: 1e-2, voltage_limit: Some(5.0), ..Default::default() };
    let vr = solve_waveform(&model, &wps, q, m, false, &opts).expect("right");
    let vl = solve_waveform(&model, &wps, q, m, true,  &opts).expect("left");

    // first two electrodes are the rails so they should swap across all waypoints
    for i in 0..wps.len() {
        assert!((vr[i][0] - vl[i][1]).abs() < 1e-8);
        assert!((vr[i][1] - vl[i][0]).abs() < 1e-8);
    }
}

#[test]
fn axial_frequency_near_target() {
    let omega_axial = 2.0 * std::f64::consts::PI * 1.5e6;
    let model = build_model(omega_axial);
    let wps = make_waypoints(9, omega_axial);
    let q = 1.602e-19;
    let m = 2.84e-25;
    let opts = LsqOptions { lambda: 1e-2, voltage_limit: Some(5.0), ..Default::default() };
    let vr = solve_waveform(&model, &wps, q, m, false, &opts).expect("solve");

    let u = Vec3 { x: 0.0, y: 0.0, z: 1.0 };
    let mut max_dev_hz = 0.0;
    for (i, wp) in wps.iter().enumerate() {
        let h = model.hess_total(wp.r, &vr[i]);
        let w = freq_along_axis(h, u, q, m);
        let f_hz = w / (2.0*std::f64::consts::PI);
        let dev = (f_hz - omega_axial/(2.0*std::f64::consts::PI)).abs();
        if dev > max_dev_hz { max_dev_hz = dev; }
    }
    // allow a loose bound for the synthetic geometry then tighten once you tune weights
    assert!(max_dev_hz < 50e3, "max deviation was {} Hz", max_dev_hz);
}

#[test]
fn constraint_shape_is_six_by_n() {
    let omega_axial = 2.0 * std::f64::consts::PI * 1.5e6;
    let model = build_model(omega_axial);
    let wps = make_waypoints(1, omega_axial);
    let q = 1.602e-19;
    let m = 2.84e-25;
    let (a, b) = ionwave::constraints::build_constraints(&model, &wps[0], q, m);
    assert_eq!(a.nrows(), 6);
    assert_eq!(a.ncols(), model.n_electrodes());
    assert_eq!(b.len(), 6);
}
