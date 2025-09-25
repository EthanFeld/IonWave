use ionwave::basis::{GaussianBasis, PotentialBasis};
use ionwave::types::{Vec3, Hess};

fn approx_eq(a: f64, b: f64, rtol: f64, atol: f64) -> bool {
    (a - b).abs() <= atol + rtol * b.abs().max(a.abs())
}

fn num_grad(basis: &impl PotentialBasis, r: Vec3, h: f64) -> Vec3 {
    // central differences on phi
    let fxp = basis.phi(Vec3 { x: r.x + h, ..r });
    let fxm = basis.phi(Vec3 { x: r.x - h, ..r });
    let fyp = basis.phi(Vec3 { y: r.y + h, ..r });
    let fym = basis.phi(Vec3 { y: r.y - h, ..r });
    let fzp = basis.phi(Vec3 { z: r.z + h, ..r });
    let fzm = basis.phi(Vec3 { z: r.z - h, ..r });
    Vec3 {
        x: (fxp - fxm) / (2.0 * h),
        y: (fyp - fym) / (2.0 * h),
        z: (fzp - fzm) / (2.0 * h),
    }
}

fn num_hess(basis: &impl PotentialBasis, r: Vec3, h: f64) -> Hess {
    // second partials by central differences
    let f      = basis.phi(r);
    let fxx_p  = basis.phi(Vec3 { x: r.x + h, ..r });
    let fxx_m  = basis.phi(Vec3 { x: r.x - h, ..r });
    let fyy_p  = basis.phi(Vec3 { y: r.y + h, ..r });
    let fyy_m  = basis.phi(Vec3 { y: r.y - h, ..r });
    let fzz_p  = basis.phi(Vec3 { z: r.z + h, ..r });
    let fzz_m  = basis.phi(Vec3 { z: r.z - h, ..r });

    let d2x = (fxx_p - 2.0 * f + fxx_m) / (h * h);
    let d2y = (fyy_p - 2.0 * f + fyy_m) / (h * h);
    let d2z = (fzz_p - 2.0 * f + fzz_m) / (h * h);

    // mixed partials by 4 point stencil
    let fxy_pp = basis.phi(Vec3 { x: r.x + h, y: r.y + h, ..r });
    let fxy_pm = basis.phi(Vec3 { x: r.x + h, y: r.y - h, ..r });
    let fxy_mp = basis.phi(Vec3 { x: r.x - h, y: r.y + h, ..r });
    let fxy_mm = basis.phi(Vec3 { x: r.x - h, y: r.y - h, ..r });
    let dxy = (fxy_pp - fxy_pm - fxy_mp + fxy_mm) / (4.0 * h * h);

    let fxz_pp = basis.phi(Vec3 { x: r.x + h, z: r.z + h, ..r });
    let fxz_pm = basis.phi(Vec3 { x: r.x + h, z: r.z - h, ..r });
    let fxz_mp = basis.phi(Vec3 { x: r.x - h, z: r.z + h, ..r });
    let fxz_mm = basis.phi(Vec3 { x: r.x - h, z: r.z - h, ..r });
    let dxz = (fxz_pp - fxz_pm - fxz_mp + fxz_mm) / (4.0 * h * h);

    let fyz_pp = basis.phi(Vec3 { y: r.y + h, z: r.z + h, ..r });
    let fyz_pm = basis.phi(Vec3 { y: r.y + h, z: r.z - h, ..r });
    let fyz_mp = basis.phi(Vec3 { y: r.y - h, z: r.z + h, ..r });
    let fyz_mm = basis.phi(Vec3 { y: r.y - h, z: r.z - h, ..r });
    let dyz = (fyz_pp - fyz_pm - fyz_mp + fyz_mm) / (4.0 * h * h);

    Hess { xx: d2x, yy: d2y, zz: d2z, xy: dxy, xz: dxz, yz: dyz }
}

#[test]
fn gaussian_grad_and_hess_match_finite_differences() {
    // a representative Gaussian basis from your demo geometry
    let basis = GaussianBasis {
        center: Vec3 { x: -50e-6, y: 0.0, z: 0.0 },
        sigma: 40e-6,
        scale: 2e-3, // small scale keeps numbers well conditioned
    };
    // pick a point not exactly at the center
    let r = Vec3 { x: -40e-6, y: 10e-6, z: 12e-6 };

    // step size tuned to sigma scale
    let h = 1e-7; // 0.1 micron

    let g_analytic = basis.grad(r);
    let g_numeric = num_grad(&basis, r, h);

    // tolerances
    let rtol_g = 3e-3;
    let atol_g = 1e-9;

    assert!(approx_eq(g_analytic.x, g_numeric.x, rtol_g, atol_g), "grad x mismatch: analytic {} numeric {}", g_analytic.x, g_numeric.x);
    assert!(approx_eq(g_analytic.y, g_numeric.y, rtol_g, atol_g), "grad y mismatch: analytic {} numeric {}", g_analytic.y, g_numeric.y);
    assert!(approx_eq(g_analytic.z, g_numeric.z, rtol_g, atol_g), "grad z mismatch: analytic {} numeric {}", g_analytic.z, g_numeric.z);

    let h_analytic = basis.hess(r);
    let h_numeric = num_hess(&basis, r, h);

    let rtol_h = 1e-2;
    let atol_h = 1e-7;

    assert!(approx_eq(h_analytic.xx, h_numeric.xx, rtol_h, atol_h), "hess xx mismatch: analytic {} numeric {}", h_analytic.xx, h_numeric.xx);
    assert!(approx_eq(h_analytic.yy, h_numeric.yy, rtol_h, atol_h), "hess yy mismatch: analytic {} numeric {}", h_analytic.yy, h_numeric.yy);
    assert!(approx_eq(h_analytic.zz, h_numeric.zz, rtol_h, atol_h), "hess zz mismatch: analytic {} numeric {}", h_analytic.zz, h_numeric.zz);
    assert!(approx_eq(h_analytic.xy, h_numeric.xy, rtol_h, atol_h), "hess xy mismatch: analytic {} numeric {}", h_analytic.xy, h_numeric.xy);
    assert!(approx_eq(h_analytic.xz, h_numeric.xz, rtol_h, atol_h), "hess xz mismatch: analytic {} numeric {}", h_analytic.xz, h_numeric.xz);
    assert!(approx_eq(h_analytic.yz, h_numeric.yz, rtol_h, atol_h), "hess yz mismatch: analytic {} numeric {}", h_analytic.yz, h_numeric.yz);
}
