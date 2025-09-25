# Ionwave

Ionwave is a Rust framework for computing and validating ion transport waveforms in segmented linear Paul traps, the core architecture of many trapped-ion quantum computers.

It combines physics modeling, numerical optimization, and modern Rust engineering practices to produce DC electrode voltage sequences that smoothly transport ions while preserving stable confinement.

---

## What this repository does

- **Trap model**
  - RF pseudopotential surrogate provides radial confinement.
  - Gaussian DC electrode basis functions approximate segmented electrodes along the trap axis.

- **Waveform solver**
  - Builds constraints at each transport waypoint:
    - Zero electric field at the ion position.
    - Target axial frequency enforced for motional stability.
    - Radial curvature floors to prevent loss of confinement.
  - Solves a constrained least-squares problem using a custom LSQR solver with Tikhonov regularization.

- **Output**
  - Produces voltage sequences for each electrode as CSV files.

In short: Ionwave automatically finds the electrode voltages needed to move an ion smoothly along the trap without significantly changing its oscillation frequency.

---

## Potential uses

- **Quantum computing hardware**
  - Generate shuttling, splitting, and merging waveforms for trapped-ion qubits.
  - Support scalable ion trap architectures by automating waveform design.

- **Research and simulation**
  - Prototype electrode configurations without running full finite-element models.
  - Benchmark numerical optimization techniques such as autodifferentiation and sparse solvers.
  - Explore waveform stability and axial frequency preservation across transport paths.

---
## Notable Features

- **Physics modeling**
  - RF pseudopotential surrogate for radial confinement
  - Gaussian DC electrode basis functions to approximate segmented electrodes
  - Analytic gradients and Hessians for fast and accurate computations
  - Constraint builder enforcing:
    - Zero electric field at the ion position
    - Target axial curvature / frequency
    - Radial curvature floors for stable confinement

- **Numerical solvers**
  - Custom LSQR implementation in Rust
  - Tikhonov regularization for stability
  - Sparse matrix support via `sprs`
  - Parallelized mat-vec operations with `rayon`

- **Engineering practices**
  - Unit tests for gradients, Hessians, and constraints
  - Integration tests for axial frequency stability and electrode symmetry
  - Gradient/Hessian finite-difference validation
  - C2LR symmetry test (rail swap check)

- **Outputs and analysis**
  - CSV voltage waveforms for all electrodes
  - Example binary (`demo.rs`) that builds and solves a trap geometry


- **Scalability and extensibility**
  - Designed for traps with hundreds of electrodes
  - Modular basis functions (extend with autodiff or FEM solutions)
  - Parallel and sparse-ready algorithms for HPC use
