# Coined Quantum Walks

> ⚠️ **Project status**  
> This repository contains the **original implementation** of my Coined Quantum Walks (CQWs) project.  
> I am currently working on a **full rewrite** with cleaner abstractions, improved explanations, and more optimised numerical routines.

This project accompanies the research work:

**“Properties of Localized Multi-Dimensional Disordered Coined Quantum Walks”**  
*Adam Tang — Department of Physics, University of Warwick*

The codebase implements numerical simulations of **discrete-time coined quantum walks** in **one, two, and three dimensions**, with a focus on **quenched disorder**, **Anderson localization**, and quantum information measures such as **entanglement entropy**, **state fidelity**, and **participation values**.

---

## Scientific Overview

Coined Quantum Walks (CQWs) are quantum analogues of classical random walks, consisting of a **walker** and a **quantum coin** evolving on a discrete lattice. Due to quantum superposition and interference, CQWs exhibit **ballistic spreading** in the absence of disorder.

This project investigates how **quenched disorder in the coin operator** modifies the transport properties of CQWs across dimensions:

- **1D, 2D, and 3D lattice geometries**
- **Hadamard-based coins** and their disordered generalisations
- **Anderson localization** and suppression of ballistic transport
- Scaling of localization behaviour with dimensionality

The simulations confirm:

- Anderson localization in **1D, 2D, and 3D**
- No metal–insulator transition within the explored parameter ranges
- Dimension-dependent behaviour of **entanglement entropy**
- Robust decay of **state fidelity** under even weak disorder

High-dimensional simulations (especially 3D) require **large memory systems** and were executed on **Linux-based HPC clusters** using SLURM.

---

## Repository Structure

```text
Coined-Quantum-Walks/
├── src/                     # Core simulation scripts
│   ├── 1D_runs/             # 1D quantum walk simulations
│   │   ├── run_1d_example.py
│   │   ├── 1d_ee.py         # Entanglement entropy
│   │   ├── 1d_fidelity.py   # State fidelity
│   │   └── 1D_PV.py         # Participation values
│   │
│   ├── 2D_runs/             # 2D quantum walk simulations
│   │   ├── run_2d_example.py
│   │   ├── 2d_ee.py
│   │   ├── 2d_fidelity.py
│   │   └── 2d_pv.py
│   │
│   └── 3D_runs/             # 3D quantum walk simulations
│       ├── run_3d_example.py
│       ├── 3d_ee.py
│       ├── 3d_fidelity.py
│       └── 3d_pv.py
│
├── plotters/                # Post-processing and plotting scripts
│   ├── 1D_plotting/
│   │   ├── 1D_EE_plot.py
│   │   ├── 1D_SF_plot.py
│   │   └── 1D_PV_plot.py
│   │
│   ├── 2D_plotting/
│   │   ├── 2D_EE_plot.py
│   │   ├── 2D_SF_plot.py
│   │   └── 2D_PV_plot.py
│   │
│   └── 3D_plotting/
│       ├── 3D_EE_plot.py
│       ├── 3D_SF_plot.py
│       └── 3D_PV_plot.py
│
├── HPC_jobs/                # SLURM job scripts for cluster execution
│   ├── 1D_jobs/
│   │   ├── 1D_EE_job.sh
│   │   ├── 1D_SF_job.sh
│   │   └── 1D_PV_job.sh
│   │
│   ├── 2D_jobs/
│   │   ├── 2D_EE.sh
│   │   ├── 2D_SF.sh
│   │   └── 2D_PV.sh
│   │
│   └── 3D_jobs/
│       ├── 3D_EE.sh
│       ├── 3D_SF.sh
│       └── 3D_PV.sh
│
├── output/
│   ├── data/                # Raw numerical output
│   └── logs/                # Simulation and SLURM logs
│
├── Coined_Quantum_Walks.pdf # Full theory, methodology, and results
├── LICENSE                  # MIT License
└── README.md
```

## Key Quantities Computed

The simulations measure several physically meaningful quantities that characterise the
dynamical and informational properties of coined quantum walks under disorder.

### Probability Distributions

Used to identify **ballistic**, **diffusive**, and **localized** transport regimes by examining
the spatial probability density of the walker as a function of time and disorder strength.

### State Fidelity

Measures the deviation between a disordered quantum walk and the corresponding
disorder-free (Hadamard) walk:

$F = \left| \langle \psi_W(t) \mid \psi_0(t) \rangle \right|^2$

where $\psi_W(t)$ is the state evolved with disorder strength $W$, and
$\psi_0(t)$ is the reference disorder-free state.

### Entanglement Entropy

Quantifies the entanglement between the **coin** and **walker** subsystems via the reduced
coin density matrix $\rho_c$:

$S = -\mathrm{tr}\left( \rho_c \ln \rho_c \right)$

This measure reveals how disorder and dimensionality affect quantum correlations.

### Participation Value

Measures the spatial spread and degree of localization of the walk:

$P = \frac{1}{D} \sum_i \frac{1}{|\psi_i|^4}$

where $D$ is the system size.  
As $P \to 1$, the walk is highly delocalized; as $P \to 0$, the walk is strongly localized.

---

## Running the Code

### Local (small systems)

Example scripts for modest lattice sizes can be run directly:

```bash
python src/1D_runs/run_1d_example.py
python src/2D_runs/run_2d_example.py
```

### HPC / Large Systems

For large-scale 2D and 3D simulations, SLURM job scripts are provided for execution on
high-performance computing clusters:

```bash
sbatch HPC_jobs/3D_jobs/3D_EE.sh
```

> ⚠️ **Warning**  
> 3D simulations require **very large amounts of RAM** and are not suitable for most  
> laptops or personal machines.

---

## Dependencies

- Python 3.9+
- NumPy
- SciPy
- Matplotlib
- (Optional) `multiprocessing` / `joblib` for parallel execution

Exact package versions are not pinned in this legacy version of the project.

---

## Notes on Code Quality

This repository reflects an **early-stage research implementation**:

- Scripts are dimension-specific rather than fully modular
- Memory usage is not fully optimised, especially in 3D
- Some logic is duplicated across dimensional cases

A **clean, modular rewrite** with improved performance and structure is currently in progress.

---

## Citation

If you use or reference this work, please cite:

Adam Tang, *Properties of Localized Multi-Dimensional Disordered Coined Quantum Walks*,  
Department of Physics, University of Warwick.
