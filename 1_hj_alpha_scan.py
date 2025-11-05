#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alpha-scan for Hamilton-Jacobi residual
================================================
This script reproduces the alpha-scan residual test described in the paper.

It evaluates the mass-weighted L2 residual of the Hamilton-Jacobi equation
    R_HJ(alpha) = sqrt( int_{rho>eps} rho [ S_t + |grad S|^2/(2m) + V + Q_alpha ]^2 dx ) / int_{rho>eps} rho dx,
with Q_alpha = -alpha (Delta sqrt(rho))/sqrt(rho), masking a tiny density floor eps near nodes.
We test three 1D systems with units m=hbar=omega=1 so alpha* = hbar^2/(2m) = 0.5:

1) Harmonic oscillator (HO) ground state (stationary: S_t = -E0, E0 = 1/2)
2) HO first excited state (stationary: S_t = -E1, E1 = 3/2)
3) Free Gaussian packet with momentum k in {0, 1.5} and V=0
   For the packet we compute S_t numerically from Hpsi/psi with H = -(hbar^2/2m)Delta + V.

Outputs
-------
- Console summary of minima and sample points around alpha*
- CSV file 'alpha_scan_results.csv' with rows: system,alpha,R_HJ
- NPZ file  'alpha_scan_data.npz' with raw arrays for reproducibility
- Optional plots with --plot flag (uses matplotlib; no custom colours/styles)

Usage
-----
    python hj_alpha_scan.py --N 20001 --xmin -12 --xmax 12 --eps 1e-12 --plot
Defaults are chosen to match the manuscript-grade resolution.
"""
import argparse
import sys
import numpy as np
from typing import Optional

def _version_banner():
    print(f"Python {sys.version.split()[0]} | NumPy {np.__version__}")

try:
    import matplotlib.pyplot as plt
    _HAVE_PLT = True
except Exception:
    _HAVE_PLT = False

# ---------- Numerics: finite differences (2nd order, conservative) ----------

def gradient_central(f: np.ndarray, dx: float) -> np.ndarray:
    """Second-order central gradient with one-sided second-order at boundaries."""
    if f.size < 3:
        raise ValueError("gradient_central requires at least 3 points.")
    g = np.empty_like(f, dtype=float)
    g[1:-1] = (f[2:] - f[:-2]) * (0.5/dx)
    g[0] = (-3.0*f[0] + 4.0*f[1] - f[2]) / (2.0*dx)
    g[-1] = (3.0*f[-1] - 4.0*f[-2] + f[-3]) / (2.0*dx)
    return g

def laplacian_1d(f: np.ndarray, dx: float) -> np.ndarray:
    """Second-order Laplacian in 1D with second-order one-sided boundaries."""
    if f.size < 5:
        raise ValueError("laplacian_1d requires at least 5 points for one-sided second-order stencils.")
    lap = np.empty_like(f, dtype=float)
    lap[1:-1] = (f[2:] - 2.0*f[1:-1] + f[:-2]) / (dx*dx)
    lap[0]  = (2.0*f[0] - 5.0*f[1] + 4.0*f[2] - f[3]) / (dx*dx)
    lap[-1] = (2.0*f[-1] - 5.0*f[-2] + 4.0*f[-3] - f[-4]) / (dx*dx)
    return lap

# ---------- Physics helpers ----------

def ho_eigenstate(n: int, x: np.ndarray) -> np.ndarray:
    """
    1D harmonic oscillator eigenfunctions in units m=omega=hbar=1:
    psi0 = pi^{-1/4} exp(-x^2/2)
    psi1 = sqrt(2) x psi0
    Returns complex array (real-valued here).
    """
    norm0 = np.pi**(-0.25)
    psi0 = norm0 * np.exp(-0.5*x*x)
    if n == 0:
        return psi0.astype(complex)
    elif n == 1:
        return (np.sqrt(2.0) * x * psi0).astype(complex)
    else:
        raise ValueError("Only n=0 or n=1 are implemented for this test.")

def ho_energy(n: int) -> float:
    """E_n for HO in units m=omega=hbar=1."""
    return n + 0.5

def free_gaussian_packet(x: np.ndarray, sigma: float, x0: float, k: float) -> np.ndarray:
    """Minimum-uncertainty Gaussian packet at t=0 with central wavenumber k."""
    A = (np.pi**0.25 * np.sqrt(sigma))**-1
    phase = 1j * k * x
    return (A * np.exp(-0.5*((x - x0)/sigma)**2) * np.exp(phase)).astype(complex)

# ---------- Residual machinery ----------

def safe_phase_unwrap(psi: np.ndarray) -> np.ndarray:
    """Unwrap phase of psi robustly."""
    return np.unwrap(np.angle(psi))

def quantum_potential(alpha: float, rho: np.ndarray, dx: float, tiny: float) -> np.ndarray:
    """Q_alpha = -alpha (Delta sqrt(rho))/sqrt(rho) with small positive floor to avoid division by ~0."""
    sr = np.sqrt(np.maximum(rho, 0.0))
    lap_sr = laplacian_1d(sr, dx)
    denom = np.maximum(sr, tiny)
    return -alpha * (lap_sr / denom)

def hj_residual(alpha: float,
                x: np.ndarray,
                psi: np.ndarray,
                V: np.ndarray,
                m: float,
                hbar: float,
                eps_rho: float,
                use_stationary_E: Optional[float] = None) -> float:
    """
    Compute R_HJ(alpha) for a given state psi and potential V.
    If use_stationary_E is not None, sets S_t = -E (useful for exact eigenstates).
    Else, estimates S_t = -Re( (Hpsi)/psi ).
    """
    dx = x[1] - x[0]
    rho = np.abs(psi)**2
    mask = rho > eps_rho
    if not np.any(mask):
        raise ValueError("Density mask is empty. Decrease --eps or adjust the state/domain.")
    tiny = max(1e-300, eps_rho*1e-3)

    # current j = (hbar/m) * Im(psi* dpsi/dx), then v = j/rho, so dSdx = m v.
    dpsi_dx = gradient_central(psi.real, dx) + 1j * gradient_central(psi.imag, dx)
    j = (hbar / m) * np.imag(np.conjugate(psi) * dpsi_dx)
    v = np.zeros_like(j, dtype=float)
    v[mask] = j[mask] / rho[mask]
    dSdx = m * v
    kin = 0.5 * (dSdx ** 2) / m

    # Quantum potential
    Q = quantum_potential(alpha, rho, dx, tiny)

    # S_t term
    if use_stationary_E is not None:
        S_t = -float(use_stationary_E) * np.ones_like(x)
    else:
        lap_psi = laplacian_1d(psi.real, dx) + 1j*laplacian_1d(psi.imag, dx)
        Hpsi = -(hbar*hbar/(2.0*m)) * lap_psi + V * psi
        z = np.zeros_like(psi, dtype=complex)
        z[mask] = Hpsi[mask] / psi[mask]
        S_t = np.zeros_like(x, dtype=float)
        S_t[mask] = -np.real(z[mask])

    # Residual density
    Rpt = S_t + kin + V + Q
    num = np.sqrt(np.trapz(rho[mask] * (Rpt[mask] ** 2), x[mask]))
    den = np.trapz(rho[mask], x[mask])
    if den <= 0.0:
        raise ValueError("Zero or negative normalisation in masked region.")
    return float(num / den)

# ---------- Experiment runners ----------

def run_alpha_scan(x, m, hbar, V, psi, alphas, eps_rho, label, E_stationary=None):
    rows = []
    for a in alphas:
        R = hj_residual(a, x, psi, V, m, hbar, eps_rho, use_stationary_E=E_stationary)
        rows.append((label, a, R))
    arr = np.array(rows, dtype=object)
    # Identify minimum
    i_min = int(np.argmin(arr[:, 2].astype(float)))
    a_min = float(arr[i_min, 1])
    R_min = float(arr[i_min, 2])
    return arr, a_min, R_min

def pretty_table(arr, nmax=10):
    """Return a simple aligned string table for console output."""
    head = f"{'alpha':>10} | {'R_HJ':>14}"
    sep = "-"*len(head)
    lines = [head, sep]
    n = min(nmax, len(arr))
    for _, a, R in arr[:n]:
        lines.append(f"{a:10.6f} | {R:14.6e}")
    return "\n".join(lines)

# ---------- Main CLI ----------

def main():
    parser = argparse.ArgumentParser(description="Alpha-scan HJ residual reproducibility script (Test 1)." )
    parser.add_argument("--xmin", type=float, default=-12.0, help="Domain minimum (default: -12)")
    parser.addArgument = parser.add_argument  # optional alias
    parser.add_argument("--xmax", type=float, default=12.0, help="Domain maximum (default: 12)")
    parser.add_argument("--N", type=int, default=20001, help="Number of grid points (default: 20001)")
    parser.add_argument("--eps", type=float, default=1e-12, help="Density mask floor eps (default: 1e-12)")
    parser.add_argument("--sigma", type=float, default=1.0, help="Width of Gaussian packet (default: 1.0)")
    parser.add_argument("--x0", type=float, default=0.0, help="Centre of Gaussian packet (default: 0.0)")
    parser.add_argument("--klist", type=str, default="0.0,1.5", help="Comma list of k for packet (default: '0.0,1.5')")
    parser.add_argument("--alphalist", type=str, default="0.45,0.475,0.5,0.525,0.55", help="Comma list of alpha values to sample")
    parser.add_argument("--alphagrid", type=int, default=101, help="Dense alpha grid points around 0.5 for min estimation (default: 101)")
    parser.add_argument("--plot", action="store_true", help="Show/save plots of R_HJ(alpha)")
    args = parser.parse_args()

    # Physical units: m=hbar=omega=1 -> alpha* = hbar^2/(2m) = 0.5
    m = 1.0
    hbar = 1.0
    alpha_star = hbar*hbar/(2.0*m)

    # Grid
    x = np.linspace(args.xmin, args.xmax, args.N, dtype=float)
    dx = x[1] - x[0]
    if args.N < 5:
        raise ValueError("N must be at least 5 for second-order stencils.")

    # Potentials
    V_ho = 0.5 * m * x*x
    V_free = np.zeros_like(x)

    # Systems
    systems = []

    # HO ground state (stationary, S_t = -E0)
    psi0 = ho_eigenstate(0, x)
    E0 = ho_energy(0)
    systems.append(("HO_n0", psi0, V_ho, E0))

    # HO first excited (stationary, S_t = -E1) - has a node, masking handles it
    psi1 = ho_eigenstate(1, x)
    E1 = ho_energy(1)
    systems.append(("HO_n1", psi1, V_ho, E1))

    # Free Gaussian packets with k list (numerical S_t via Hpsi/psi)
    klist = [float(s.strip()) for s in args.klist.split(",") if s.strip()]
    for kval in klist:
        psi_g = free_gaussian_packet(x, sigma=args.sigma, x0=args.x0, k=kval)
        systems.append((f"FREE_packet_k={kval:g}", psi_g, V_free, None))

    # alpha samples: coarse list + dense grid around alpha*
    alist_coarse = [float(s.strip()) for s in args.alphalist.split(",") if s.strip()]
    a_dense = np.linspace(alpha_star - 0.1, alpha_star + 0.1, args.alphagrid, dtype=float)

    # Storage
    all_rows = []

    print("\nAlpha-scan HJ residual (Test 1)\n============================")
    print(f"Grid: N={args.N}, dx={dx:.3e}, domain=[{args.xmin},{args.xmax}]")
    print(f"Mask floor eps={args.eps:g}, expected Fisher alpha*={alpha_star}\n")

    # Run each system
    for label, psi, V, Estat in systems:
        print(f"System: {label}")
        # --- mass conservation check ---
        mass = np.trapz(np.abs(psi)**2, x)
        assert abs(mass - 1.0) < 1e-6, f"Mass drift in {label}: {mass:.6e} - extend domain or refine grid"
        # Dense scan for min location
        arr_dense, a_min, R_min = run_alpha_scan(x, m, hbar, V, psi, a_dense, args.eps, label, Estat)
        # Coarse values for paper-style table print
        arr_coarse, _, _ = run_alpha_scan(x, m, hbar, V, psi, np.array(alist_coarse), args.eps, label, Estat)

        # Accumulate rows
        all_rows.append(arr_dense)
        all_rows.append(arr_coarse)

        # Console output
        print(f"  Estimated minimum: alpha_min ~= {a_min:.6f}  (R_HJ ~= {R_min:.3e})")
        print("  Sample around alpha*:\n" + pretty_table(arr_coarse))
        # Quadratic growth check via least-squares on R^2 vs (alpha-alpha*)^2
        A = (arr_dense[:,1].astype(float) - alpha_star)**2
        y = arr_dense[:,2].astype(float)**2
        c = np.polyfit(A, y, 1)  # y ~= c1*A + c0 (intercept should be ~0)
        print(f"  Quadratic check (R^2 vs (alpha-alpha*)^2): slope ~= {c[0]:.3e}, intercept ~= {c[1]:.3e}\n")

    # Save CSV and NPZ
    all_arr = np.concatenate(all_rows, axis=0)
    out_csv = "alpha_scan_results.csv"
    header = "system,alpha,R_HJ"
    np.savetxt(out_csv,
               X=np.column_stack([all_arr[:,0], all_arr[:,1].astype(str), all_arr[:,2].astype(str)]),
               fmt=["%s","%s","%s"],
               delimiter=",",
               header=header,
               comments="")
    np.savez_compressed("alpha_scan_data.npz",
                        data=all_arr,
                        description="Rows of [system(str), alpha(float), R_HJ(float)]",
                        note="Includes both dense and coarse scans for each system.")

    print(f"Saved CSV: {out_csv}")
    print(f"Saved NPZ: alpha_scan_data.npz")

    # Optional plots
    if args.plot:
        if not _HAVE_PLT:
            print("matplotlib not available; skipping plots.", file=sys.stderr)
        else:
            # One figure per unique system for dense scan
            print("Generating plots...")
            for label, psi, V, Estat in systems:
                arr_dense, a_min, R_min = run_alpha_scan(x, m, hbar, V, psi, a_dense, args.eps, label, Estat)
                a_vals = arr_dense[:,1].astype(float)
                R_vals = arr_dense[:,2].astype(float)

                plt.figure()
                plt.plot(a_vals, R_vals, marker="o", linestyle="-")
                plt.axvline(alpha_star, linestyle="--")
                plt.title(f"R_HJ(alpha) - {label}")
                plt.xlabel("alpha")
                plt.ylabel("R_HJ")
                plt.tight_layout()
                png_name = f"alpha_scan_{label.replace('=','_').replace(',','_')}.png"
                plt.savefig(png_name, dpi=160)
                print(f"  Saved: {png_name}")
            print("Done.")

if __name__ == "__main__":
    main()
