use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn dot(self, o: Vec3) -> f64 { self.x*o.x + self.y*o.y + self.z*o.z }
    pub fn norm(self) -> f64 { self.dot(self).sqrt() }
    pub fn unit(self) -> Vec3 {
        let n = self.norm();
        if n == 0.0 { self } else { self / n }
    }
}

use std::ops::{Add, Sub, Mul, Div};
impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, o: Vec3) -> Vec3 { Vec3 { x: self.x+o.x, y: self.y+o.y, z: self.z+o.z } }
}
impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, o: Vec3) -> Vec3 { Vec3 { x: self.x-o.x, y: self.y-o.y, z: self.z-o.z } }
}
impl Mul<f64> for Vec3 {
    type Output = Vec3;
    fn mul(self, s: f64) -> Vec3 { Vec3 { x: self.x*s, y: self.y*s, z: self.z*s } }
}
impl Div<f64> for Vec3 {
    type Output = Vec3;
    fn div(self, s: f64) -> Vec3 { Vec3 { x: self.x/s, y: self.y/s, z: self.z/s } }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Hess {
    pub xx: f64, pub yy: f64, pub zz: f64,
    pub xy: f64, pub xz: f64, pub yz: f64,
}

impl Hess {
    pub fn quad(self, u: Vec3) -> f64 {
        // u^T H u
        let x = u.x; let y = u.y; let z = u.z;
        self.xx*x*x + self.yy*y*y + self.zz*z*z + 2.0*(self.xy*x*y + self.xz*x*z + self.yz*y*z)
    }
    pub fn add(self, o: Hess) -> Hess {
        Hess {
            xx: self.xx + o.xx, yy: self.yy + o.yy, zz: self.zz + o.zz,
            xy: self.xy + o.xy, xz: self.xz + o.xz, yz: self.yz + o.yz
        }
    }
    pub fn scale(self, s: f64) -> Hess {
        Hess { xx: self.xx*s, yy: self.yy*s, zz: self.zz*s, xy: self.xy*s, xz: self.xz*s, yz: self.yz*s }
    }
}

#[derive(Clone, Debug)]
pub struct Waypoint {
    pub r: Vec3,
    pub omega_axial: f64,      // target angular frequency
    pub axial_dir: Vec3,       // unit vector
}

#[derive(Clone, Debug)]
pub struct SolveReport {
    pub cond_est: f64,
    pub objective: f64,
}

#[derive(thiserror::Error, Debug)]
pub enum IonwaveError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("solver failure: {0}")]
    Solver(String),
}

pub type Result<T> = std::result::Result<T, IonwaveError>;
