#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entropy-production / reversibility test (Test 3)
================================================
Verifies that adding diffusion (ρ_t = D Δρ) breaks time-reversal symmetry.

Theory
------
For the diffusion equation ρ_t = D Δρ with constant D ≥ 0:
    dH/dt = D ∫ |∇ρ|² / ρ dx  ≥  0,  where  H = -∫ ρ ln ρ dx
    d/dt ∫ ρ ln ρ dx = -D ∫ |∇ρ|² / ρ dx

Hence H grows monotonically unless D = 0, which corresponds to exact reversibility.

Expected outcome
----------------
- D = 0:  all rates ≈ 0  (reversible)
- D > 0:  dH/dt > 0  and  d(∫ρ lnρ)/dt < 0
- “Direct PDE” and Fisher forms agree to roundoff once boundary flux is negligible.
"""
from __future__ import annotations
import numpy as np
import argparse

# ---------- Finite-difference operators ----------

def gradient_central(f: np.ndarray, dx: float) -> np.ndarray:
    """Second-order central gradient with one-sided boundaries."""
    g = np.empty_like(f, dtype=np.float64)
    g[1:-1] = (f[2:] - f[:-2]) * (0.5 / dx)
    g[0]    = (-3.0*f[0] + 4.0*f[1] - f[2]) / (2.0*dx)
    g[-1]   = ( 3.0*f[-1] - 4.0*f[-2] + f[-3]) / (2.0*dx)
    return g

def laplacian_1d(f: np.ndarray, dx: float) -> np.ndarray:
    """Second-order Laplacian with 4-point one-sided boundaries."""
    lap = np.empty_like(f, dtype=np.float64)
    lap[1:-1] = (f[2:] - 2.0*f[1:-1] + f[:-2]) / (dx*dx)
    lap[0]    = (2.0*f[0] - 5.0*f[1] + 4.0*f[2] - f[3]) / (dx*dx)
    lap[-1]   = (2.0*f[-1] - 5.0*f[-2] + 4.0*f[-3] - f[-4]) / (dx*dx)
    return lap

# ---------- Entropy-production evaluation ----------

def entropy_rates(x: np.ndarray, rho: np.ndarray, D: float) -> tuple[float, float, float, float, float]:
    """
    Compute theoretical and direct entropy-production rates.
    Returns (dH/dt, dSsh/dt (RHS), dSsh/dt (direct), Fisher I, boundary_flux_estimate).
    """
    dx   = float(x[1] - x[0])
    tiny = 1e-300

    grad_rho = gradient_central(rho, dx)
    I = np.trapz((grad_rho**2) / np.maximum(rho, tiny), x)  # Fisher integral

    if not np.isfinite(I):
        raise FloatingPointError("Non-finite Fisher integral — check grid or sigma/domain.")

    Hdot_RHS     = D * I
    Ssh_dot_RHS  = -D * I

    lap_rho      = laplacian_1d(rho, dx)
    Ssh_dot_dir  = np.trapz((np.log(np.maximum(rho, tiny)) + 1.0) * (D * lap_rho), x)

    # Boundary flux estimate: D * [(1+ln ρ) ∂x ρ] |_{xmin}^{xmax}
    # Using one-sided derivative consistent with gradient_central ends.
    dρdx_left  = ( -3.0*rho[0] + 4.0*rho[1] - rho[2]) / (2.0*dx)
    dρdx_right = (  3.0*rho[-1] - 4.0*rho[-2] + rho[-3]) / (2.0*dx)
    flux = D * ((1.0 + np.log(max(rho[-1], tiny))) * dρdx_right
                - (1.0 + np.log(max(rho[0],  tiny))) * dρdx_left)

    return Hdot_RHS, Ssh_dot_RHS, Ssh_dot_dir, I, flux

# ---------- Main CLI ----------

def main():
    parser = argparse.ArgumentParser(description="Entropy-production / reversibility test (Test 3)")
    parser.add_argument("--xmin", type=float, default=-20.0, help="Domain min")
    parser.add_argument("--xmax", type=float, default= 20.0, help="Domain max")
    parser.add_argument("--N",    type=int,   default=20001, help="Grid points (>= 5)")
    parser.add_argument("--sigma", type=float, default=1.0, help="Gaussian width")
    parser.add_argument("--Dlist", type=str, default="0,1e-3,5e-3",
                        help="Comma-separated diffusion constants")
    parser.add_argument("--renorm", action="store_true", help="Renormalise ρ to unit mass on the grid")
    parser.add_argument("--show-boundary", action="store_true", help="Print boundary flux estimate")
    args = parser.parse_args()

    if args.N < 5:
        raise ValueError("N must be at least 5 for 4-point one-sided boundary stencils.")

    # Grid
    x  = np.linspace(args.xmin, args.xmax, args.N, dtype=np.float64)
    dx = float(x[1] - x[0])

    # Normalised Gaussian ρ(x) (analytic prefactor), then optional discrete renorm
    sigma = float(args.sigma)
    rho = (np.pi**0.5 * sigma)**-1 * np.exp(-(x**2) / (sigma**2))
    rho = np.maximum(rho, 0.0)

    mass = np.trapz(rho, x)
    if args.renorm and mass > 0:
        rho /= mass
        mass = np.trapz(rho, x)

    # Sanity check
    if abs(mass - 1.0) > 1e-6:
        raise AssertionError(f"Mass drift {mass:.6e}. Consider --renorm, wider domain or larger N.")

    print("\nEntropy-production / reversibility test (Test 3)")
    print("=================================================")
    print(f"Grid: N={args.N}, dx={dx:.3e}, domain=[{args.xmin},{args.xmax}]")
    print(f"Gaussian σ={sigma}, mass={mass:.12f}")
    print(f"Options: renorm={args.renorm}, show_boundary={args.show_boundary}\n")

    Dvals = [float(s.strip()) for s in args.Dlist.split(",") if s.strip()]
    hdr = f"{'D':>8} | {'Fisher I':>14} | {'dH/dt':>14} | {'d(∫ρlnρ)/dt (RHS)':>22} | {'direct PDE':>16} | {'|diff|':>10}"
    if args.show_boundary:
        hdr += f" | {'boundary flux':>16}"
    print(hdr)
    print("-" * len(hdr))

    for D in Dvals:
        Hdot, Ssh_rhs, Ssh_dir, I, flux = entropy_rates(x, rho, D)
        diff = abs(Ssh_rhs - Ssh_dir)

        # Clean tiny roundoff for printing
        def z(v): return 0.0 if abs(v) < 1e-15 else v

        row = f"{D:8.1e} | {I:14.12e} | {z(Hdot):14.12e} | {z(Ssh_rhs):22.12e} | {z(Ssh_dir):16.12e} | {diff:10.3e}"
        if args.show_boundary:
            row += f" | {z(flux):16.12e}"
        print(row)

    print("\nInterpretation:")
    print("  D = 0      →  all derivatives ≈ 0  →  exact reversibility")
    print("  D > 0      →  dH/dt = D·I > 0,  d(∫ρlnρ)/dt < 0  →  entropy production")
    print("  Agreement to roundoff requires negligible boundary flux and adequate resolution.")

if __name__ == '__main__':
    main()
