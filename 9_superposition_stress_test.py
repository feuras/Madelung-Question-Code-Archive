#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Projective Superposition Stress-Test (Test 9)
=============================================

Purpose
-------
Operationally test linear superposition in the psi-picture to validate the Fisher
regulariser as the unique complexifier that preserves linearity. We evolve two
initial states separately and together and measure the projective residual

    R = || psi_plus(T) - (psi1(T) + psi2(T))/sqrt(2) ||_2,

which must be zero for a truly linear flow (up to solver tolerance). Any deviation
from the Fisher regulariser is modelled here by a small state-dependent local
curvature proxy that induces nonlinearity, producing R > 0 even at tiny strength beta.

Method
------
1D split-step Fourier method (Strang), harmonic potential V(x) = 1/2 m omega^2 x^2.
Units: hbar = m = 1 (non-dimensional). High resolution with optional refinement
study to demonstrate numerical robustness.

Outputs
-------
- CSV file with residuals and parameters.
- PNG figure of residual vs beta (if --plot).
- Stdout report including convergence checks if --refine > 0.

Usage
-----
    python 9_superposition_stress_test.py --T 30 --dt 0.05 --N 4096 --L 80 \
        --betas 0,0.005,0.01,0.02,0.05 --omega 0.1 --sigma 1.2 \        --x01 -6 --x02 6 --p01 0.8 --p02 -0.8 \        --save-prefix results/test9 --plot --refine 1

The linear Schrodinger case corresponds to beta = 0. Any beta > 0 uses the
nonlinear proxy.

Reproducibility
---------------
The method is deterministic given the grid. We normalise wavefunctions at each
time step in the nonlinear branch to avoid norm drift without affecting the
residual diagnostic.

License
-------
MIT
"""
import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from numpy.fft import fft, ifft, fftfreq

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

hbar = 1.0
m = 1.0

@dataclass
class Params:
    N: int
    L: float
    dt: float
    T: float
    omega: float
    sigma: float
    x01: float
    x02: float
    p01: float
    p02: float
    betas: List[float]
    refine: int
    save_prefix: str
    do_plot: bool
    nonfisher: str
    gamma: float

def gaussian(x, x0, p0, sigma):
    return (1.0 / (np.pi * sigma**2)**0.25) * np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * p0 * (x - x0) / hbar)

def normalize(psi, x):
    nrm2 = np.trapz(np.abs(psi)**2, x)
    if nrm2 <= 0:
        raise ValueError("Non-positive norm encountered.")
    return psi / math.sqrt(nrm2)

def setup_grid(N: int, L: float):
    dx = 2 * L / N
    x = np.linspace(-L, L - dx, N)
    k = fftfreq(N, d=dx) * 2 * np.pi
    dk = k[1] - k[0]
    return x, k, dx, dk

def potential(x, omega):
    return 0.5 * m * (omega**2) * x**2

def kinetic_symbol(k):
    return (hbar**2 / (2 * m)) * (k**2)

def evolve_linear(psi0, x, Vx, K, dt, nsteps):
    psi = psi0.copy()
    T_half = np.exp(-1j * K * dt / (2 * hbar))
    V_full = np.exp(-1j * Vx * dt / hbar)
    for _ in range(nsteps):
        psi = ifft(T_half * fft(psi))
        psi = V_full * psi
        psi = ifft(T_half * fft(psi))
    return normalize(psi, x)

def evolve_nonlinear(psi0, x, Vx, K, dt, nsteps, beta=0.02,
                     nonfisher="rho-gamma", gamma=0.5, eps=1e-12):
    psi = psi0.copy()
    dx = x[1] - x[0]
    T_half = np.exp(-1j * K * dt / (2 * hbar))
    for _ in range(nsteps):
        # half kinetic
        psi = ifft(T_half * fft(psi))
        # potential + non-Fisher proxy
        rho = np.abs(psi)**2
        if nonfisher == "drho2-rho2":
            drho = (np.roll(rho, -1) - np.roll(rho, 1)) / (2 * dx)
            nl = beta * (drho**2 / (np.maximum(rho, eps)**2))
        elif nonfisher == "rho-gamma":
            nl = beta * (np.maximum(rho, eps)**gamma)
        else:
            raise ValueError(f"unknown nonfisher model: {nonfisher}")
        psi = np.exp(-1j * (Vx + nl) * dt / hbar) * psi
        # half kinetic
        psi = ifft(T_half * fft(psi))
        psi = normalize(psi, x)
    return psi

def superposition_residual(psi1_T, psi2_T, psi_plus_T, x):
    combo = normalize((psi1_T + psi2_T) / np.sqrt(2.0), x)
    diff = psi_plus_T - combo
    return math.sqrt(np.trapz(np.abs(diff)**2, x))

def run_once(P: Params):
    # grid
    x, k, dx, dk = setup_grid(P.N, P.L)
    Vx = potential(x, P.omega)
    K = kinetic_symbol(k)

    # initial states
    psi1_0 = gaussian(x, P.x01, P.p01, P.sigma)
    psi2_0 = gaussian(x, P.x02, P.p02, P.sigma)
    psi_plus_0 = (psi1_0 + psi2_0) / math.sqrt(2.0)

    psi1_0 = normalize(psi1_0, x)
    psi2_0 = normalize(psi2_0, x)
    psi_plus_0 = normalize(psi_plus_0, x)

    nsteps = int(round(P.T / P.dt))
    if nsteps < 1:
        raise ValueError("T/dt must be >= 1 step.")

    # linear baseline
    psi1_lin = evolve_linear(psi1_0, x, Vx, K, P.dt, nsteps)
    psi2_lin = evolve_linear(psi2_0, x, Vx, K, P.dt, nsteps)
    psi_plus_lin = evolve_linear(psi_plus_0, x, Vx, K, P.dt, nsteps)
    R_lin = superposition_residual(psi1_lin, psi2_lin, psi_plus_lin, x)

    # nonlinear sweeps
    results = []
    for b in P.betas:
        if abs(b) < 1e-15:
            R = R_lin
            model = "linear"
        else:
            psi1_T = evolve_nonlinear(psi1_0, x, Vx, K, P.dt, nsteps,
                                      beta=b, nonfisher=P.nonfisher, gamma=P.gamma)
            psi2_T = evolve_nonlinear(psi2_0, x, Vx, K, P.dt, nsteps,
                                      beta=b, nonfisher=P.nonfisher, gamma=P.gamma)
            psi_plus_T = evolve_nonlinear(psi_plus_0, x, Vx, K, P.dt, nsteps,
                                          beta=b, nonfisher=P.nonfisher, gamma=P.gamma)
            R = superposition_residual(psi1_T, psi2_T, psi_plus_T, x)
            model = "nonlinear"
        results.append((model, b, R))

    return x, results, dict(dx=dx)

def refine_study(P: Params) -> Tuple[List[Tuple[str, float, float]], List[Tuple[str, float, float]]]:
    # base run
    _, base_results, _ = run_once(P)

    if P.refine <= 0:
        return base_results, []

    # refined grid: N -> 2N, dt -> dt/2
    Pref = Params(
        N=2 * P.N, L=P.L, dt=P.dt / 2.0, T=P.T,
        omega=P.omega, sigma=P.sigma,
        x01=P.x01, x02=P.x02, p01=P.p01, p02=P.p02,
        betas=P.betas, refine=0, save_prefix=P.save_prefix, do_plot=P.do_plot,
        nonfisher=P.nonfisher, gamma=P.gamma
    )

    _, refined_results, _ = run_once(Pref)
    return base_results, refined_results

def main():
    ap = argparse.ArgumentParser(description="Projective Superposition Stress-Test (Test 9)")
    ap.add_argument("--N", type=int, default=4096, help="grid points")
    ap.add_argument("--L", type=float, default=80.0, help="half-domain length; total length is 2L")
    ap.add_argument("--dt", type=float, default=0.05, help="time step")
    ap.add_argument("--T", type=float, default=30.0, help="final time")
    ap.add_argument("--omega", type=float, default=0.1, help="harmonic frequency")
    ap.add_argument("--sigma", type=float, default=1.2, help="initial Gaussian width")
    ap.add_argument("--x01", type=float, default=-6.0, help="initial center of psi1")
    ap.add_argument("--x02", type=float, default=+6.0, help="initial center of psi2")
    ap.add_argument("--p01", type=float, default=+0.8, help="initial momentum of psi1")
    ap.add_argument("--p02", type=float, default=-0.8, help="initial momentum of psi2")
    ap.add_argument("--betas", type=str, default="0,0.005,0.01,0.02,0.05", help="comma-separated beta values; 0 equals linear")
    ap.add_argument("--refine", type=int, default=1, help="0 none; 1 single refinement N->2N, dt->dt/2")
    ap.add_argument("--save-prefix", type=str, default="results/test9", help="prefix for outputs (CSV, PNG)")
    ap.add_argument("--plot", action="store_true", help="save a PNG plot of residual vs beta")
    ap.add_argument("--nonfisher", type=str, default="rho-gamma",
                    choices=["rho-gamma", "drho2-rho2"],
                    help="non-Fisher curvature proxy: rho-gamma uses nl = beta*rho^gamma; drho2-rho2 uses nl = beta*(|∇rho|^2/rho^2)")
    ap.add_argument("--gamma", type=float, default=0.5,
                    help="exponent for --nonfisher rho-gamma (nl = beta*rho^gamma)")
    args = ap.parse_args()

    betas = [float(s) for s in args.betas.split(",") if len(s.strip()) > 0]
    P = Params(
        N=args.N, L=args.L, dt=args.dt, T=args.T,
        omega=args.omega, sigma=args.sigma,
        x01=args.x01, x02=args.x02, p01=args.p01, p02=args.p02,
        betas=betas, refine=args.refine,
        save_prefix=args.save_prefix, do_plot=bool(args.plot),
        nonfisher=args.nonfisher, gamma=args.gamma
    )

    os.makedirs(os.path.dirname(P.save_prefix), exist_ok=True)

    base, refined = refine_study(P)

    # Write CSV
    import csv
    csv_path = f"{P.save_prefix}_residuals.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "beta", "nonfisher", "gamma", "R_superposition", "grid"])
        for model, b, R in base:
            w.writerow([model, f"{b:.8g}", P.nonfisher, f"{P.gamma:.6g}", f"{R:.16e}", "base"])
        for model, b, R in refined:
            w.writerow([model, f"{b:.8g}", P.nonfisher, f"{P.gamma:.6g}", f"{R:.16e}", "refined"])

    # Stdout report
    def fmt_rows(rows):
        return "\n".join([f"  {model:9s} beta={b:>8g}  R={R:.6e}" for model, b, R in rows])

    print("\nProjective Superposition Stress-Test (Test 9)")
    print("Parameters:", json.dumps({
        "N": P.N, "L": P.L, "dt": P.dt, "T": P.T,
        "omega": P.omega, "sigma": P.sigma,
        "x01": P.x01, "x02": P.x02, "p01": P.p01, "p02": P.p02,
        "betas": P.betas, "refine": P.refine,
        "nonfisher": P.nonfisher, "gamma": P.gamma
    }, indent=2))

    print("\nBase grid results:")
    print(fmt_rows(base))

    if refined:
        print("\nRefined grid results:")
        print(fmt_rows(refined))

        # Convergence indicator: linear case should converge to zero; nonlinear should be stable.
        # We compute |R_base - R_refined| for each beta.
        diffs = []
        mapping_ref = {(b, m): R for (m, b, R) in refined for m in [m]}
        for m, b, Rb in base:
            Rr = mapping_ref.get((b, m), None)
            if Rr is not None:
                diffs.append((m, b, abs(Rb - Rr)))
        if diffs:
            print("\nGrid refinement deltas |R_base - R_refined|:")
            for m, b, d in diffs:
                print(f"  {m:9s} beta={b:>8g}  dR={d:.6e}")

    print(f"\nCSV written to: {csv_path}")

    # Plot
    if P.do_plot:
        if not HAS_MPL:
            print("matplotlib not available; skipping plot.")
        else:
            # collapse to a single series beta -> R using base results
            xs = [b for _, b, _ in base]
            ys = [R for _, _, R in base]
            # keep ordering by beta
            order = np.argsort(xs)
            xs = np.array(xs)[order]
            ys = np.array(ys)[order]

            plt.figure(figsize=(6,4))
            plt.plot(xs, ys, marker="o")
            plt.xlabel("nonlinearity strength beta")
            plt.ylabel("superposition residual R")
            plt.title("Projective superposition stress-test")
            plt.tight_layout()
            png_path = f"{P.save_prefix}_residuals.png"
            plt.savefig(png_path, dpi=160)
            print(f"PNG written to: {png_path}")

if __name__ == "__main__":
    main()
