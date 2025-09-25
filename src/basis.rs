// src/basis.rs

use crate::types::{Vec3, Hess};

pub trait PotentialBasis: Send + Sync {
    fn phi(&self, r: Vec3) -> f64;
    fn grad(&self, r: Vec3) -> Vec3;
    fn hess(&self, r: Vec3) -> Hess;
}

pub struct RfPseudo { pub kr: f64, pub kz: f64 }
impl PotentialBasis for RfPseudo {
    fn phi(&self, r: Vec3) -> f64 { 0.5 * (self.kr * (r.x*r.x + r.y*r.y) + self.kz * r.z*r.z) }
    fn grad(&self, r: Vec3) -> Vec3 { Vec3 { x: self.kr*r.x, y: self.kr*r.y, z: self.kz*r.z } }
    fn hess(&self, _r: Vec3) -> Hess {
        Hess { xx: self.kr, yy: self.kr, zz: self.kz, xy: 0.0, xz: 0.0, yz: 0.0 }
    }
}

pub struct GaussianBasis { pub center: Vec3, pub sigma: f64, pub scale: f64 }

impl PotentialBasis for GaussianBasis {
    fn phi(&self, r: Vec3) -> f64 {
        let dx = r.x - self.center.x;
        let dy = r.y - self.center.y;
        let dz = r.z - self.center.z;
        let s2 = self.sigma * self.sigma;
        self.scale * (-0.5 * (dx*dx + dy*dy + dz*dz) / s2).exp()
    }
    fn grad(&self, r: Vec3) -> Vec3 {
        let dx = r.x - self.center.x;
        let dy = r.y - self.center.y;
        let dz = r.z - self.center.z;
        let s2 = self.sigma * self.sigma;
        let p = self.phi(r);
        Vec3 { x: -p * dx / s2, y: -p * dy / s2, z: -p * dz / s2 }
    }
    fn hess(&self, r: Vec3) -> Hess {
        let dx = r.x - self.center.x;
        let dy = r.y - self.center.y;
        let dz = r.z - self.center.z;
        let s2 = self.sigma * self.sigma;
        let s4 = s2 * s2;
        let p = self.phi(r);
        let xx = p * ((dx*dx - s2) / s4);
        let yy = p * ((dy*dy - s2) / s4);
        let zz = p * ((dz*dz - s2) / s4);
        let xy = p * ((dx*dy) / s4);
        let xz = p * ((dx*dz) / s4);
        let yz = p * ((dy*dz) / s4);
        Hess { xx, yy, zz, xy, xz, yz }
    }
}

pub struct TrapModel {
    pub rf: Box<dyn PotentialBasis>,
    pub dc: Vec<Box<dyn PotentialBasis>>,
    pub c2lr_pair: Option<(usize, usize)>,
}

impl TrapModel {
    pub fn new(rf: Box<dyn PotentialBasis>, dc: Vec<Box<dyn PotentialBasis>>, c2lr_pair: Option<(usize, usize)>) -> Self {
        Self { rf, dc, c2lr_pair }
    }
    pub fn n_electrodes(&self) -> usize { self.dc.len() }
    pub fn grad_total(&self, r: Vec3, v: &[f64]) -> Vec3 {
        let mut g = self.rf.grad(r);
        for (i, b) in self.dc.iter().enumerate() {
            let gi = b.grad(r);
            g.x += v[i]*gi.x; g.y += v[i]*gi.y; g.z += v[i]*gi.z;
        }
        g
    }
    pub fn hess_total(&self, r: Vec3, v: &[f64]) -> Hess {
        let mut h = self.rf.hess(r);
        for (i, b) in self.dc.iter().enumerate() {
            let hi = b.hess(r).scale(v[i]);
            h = h.add(hi);
        }
        h
    }
}
