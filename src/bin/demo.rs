use ionwave::basis::{TrapModel, RfPseudo, GaussianBasis, PotentialBasis};
use ionwave::c2lr::solve_waveform;
use ionwave::lsq::LsqOptions;
use ionwave::types::{Vec3, Waypoint};
use ionwave::dynamics::secular_freqs;
use ionwave::io::write_csv;

fn freq_along_axis(h: ionwave::types::Hess, u: ionwave::types::Vec3, q: f64, m: f64) -> f64 {
    let curv = h.quad(u);
    let w = (q * curv / m).max(0.0).sqrt();
    w
}

fn main() -> anyhow::Result<()> {
    // physical constants
    let q = 1.602e-19;     // C
    let m = 2.84e-25;      // kg approx 171Yb

    // target axial frequency and matching rf curvature
    let omega_axial = 2.0 * std::f64::consts::PI * 1.5e6; // rad s
    let target_curv = m * omega_axial * omega_axial / q;
    
    // choose axial along z
    let axial_dir = Vec3 { x: 0.0, y: 0.0, z: 1.0 };

    // rf pseudopotential surrogate
    // kr is large so radial modes are well confined
    let rf = RfPseudo { kr: 1.0e10, kz: target_curv };


    // build dc bases
    // arrange gaussian lobes near z axis, staggered in x for steering
    let mut dc: Vec<Box<dyn PotentialBasis>> = Vec::new();
    let sigma = 40e-6;
    let dc_scale = 0.0001;    // gentle dc influence
    let z_positions: Vec<f64> = (-4..=4).map(|k| k as f64 * 63e-6).collect();
    for (idx, zc) in z_positions.iter().enumerate() {
        let left = GaussianBasis {
            center: Vec3 { x: -50e-6, y: 0.0, z: *zc },
            sigma,
            scale: dc_scale,
        };
        let right = GaussianBasis {
            center: Vec3 { x:  50e-6, y: 0.0, z: *zc },
            sigma,
            scale: dc_scale,
        };
        dc.push(Box::new(left));
        dc.push(Box::new(right));

        if idx % 2 == 0 {
            let center = GaussianBasis {
                center: Vec3 { x: 0.0, y: 0.0, z: *zc + 0.5*63e-6 },
                sigma,
                scale: 0.8 * dc_scale,
            };
            dc.push(Box::new(center));
        }
    }

    // define the C2LR pair as the first two electrodes we added
    let c2lr_pair = Some((0usize, 1usize));
    let model = TrapModel::new(Box::new(rf), dc, c2lr_pair);

    // waypoints for a single C2LR segment of 63 micrometers in z
    let n_wp = 15usize;
    let z0 = 0.0;
    let dz = 63e-6;
    let mut waypoints: Vec<Waypoint> = Vec::new();
    for i in 0..n_wp {
        let t = i as f64 / (n_wp as f64 - 1.0);
        let z = z0 + t*dz;
        waypoints.push(Waypoint {
            r: Vec3 { x: 0.0, y: 0.0, z },
            omega_axial,
            axial_dir,
        });
    }

    // solver options
let opts = LsqOptions { 
    lambda: 1e-2, 
    voltage_limit: Some(5.0), 
    ..Default::default() 
};
    // solve right and left segments
    let volts_right = solve_waveform(&model, &waypoints, q, m, false, &opts)
        .expect("solve right");
    let volts_left  = solve_waveform(&model, &waypoints, q, m, true,  &opts)
        .expect("solve left");

    // report axial frequency along the chosen axis
    let u = axial_dir;
    let h0 = model.hess_total(waypoints[0].r, &volts_right[0]);
    let w0 = freq_along_axis(h0, u, q, m);

    let hend = model.hess_total(waypoints[n_wp-1].r, &volts_right[n_wp-1]);
    let wend = freq_along_axis(hend, u, q, m);

    // also compute a max deviation over all waypoints
    let mut max_dev_hz = 0.0;
    for (i, wp) in waypoints.iter().enumerate() {
        let h = model.hess_total(wp.r, &volts_right[i]);
        let w = freq_along_axis(h, u, q, m);
        let axial_hz = w / (2.0*std::f64::consts::PI);
        let dev = (axial_hz - omega_axial/(2.0*std::f64::consts::PI)).abs();
        if dev > max_dev_hz { max_dev_hz = dev; }
    }
    println!("max axial deviation {:.3} kHz", max_dev_hz / 1e3);

    println!("electrodes: {}", model.n_electrodes());
    println!("target axial {:.3} MHz", omega_axial/(2.0*std::f64::consts::PI)/1e6);
    println!("axial at start {:.3} MHz, at end {:.3} MHz",
        w0/(2.0*std::f64::consts::PI)/1e6,
        wend/(2.0*std::f64::consts::PI)/1e6
    );

    // write csv files
    write_csv("target/out/waveforms.csv", &volts_right)?;
    println!("wrote target/out/waveforms.csv with {} rows and {} columns", volts_right.len(), volts_right[0].len());
    write_csv("target/out/waveforms_left.csv", &volts_left)?;
    println!("wrote target/out/waveforms_left.csv");

    Ok(())
}
