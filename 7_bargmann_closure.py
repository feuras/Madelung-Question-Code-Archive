#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bargmann algebra closure test (exact per paper)
===============================================

We verify, numerically and directly on the hydrodynamic variables (rho, S),
that the canonical Poisson structure and Fisher-regularised Hamiltonian
realise the Bargmann (centrally extended Galilei) algebra:

  {H, P} = 0,       {H, K} = -P,       {P, K} = - m ∫ rho dx,

with:
  H[ρ,S] = ∫ [ ρ |∂x S|^2 / (2m) + V ρ + α |∂x sqrt(ρ)|^2 ] dx,
  P[ρ,S] = ∫ ρ ∂x S dx,
  K[ρ,S] = m ∫ x ρ dx - t P,

and α = ℏ^2 / (2m). We take V(x) = 0 (translation-invariant) and periodic BCs.

Poisson bracket on functionals F, G:
  {F,G} = ∫ [ (δF/δρ)(δG/δS) - (δF/δS)(δG/δρ) ] dx

Discretisation: uniform grid, periodic second-order finite differences.

This script prints the bracket values, the expected values, and absolute/relative
errors. Convergence is O(dx^2); at high resolution errors hit machine floor.
"""

import numpy as np
import argparse

# ------------------------- finite differences (periodic) ------------------------

def dx_periodic(f: np.ndarray, dx: float) -> np.ndarray:
    """Central difference with periodic wrap: f_x."""
    return (np.roll(f, -1) - np.roll(f, 1)) / (2.0 * dx)

def dxx_periodic(f: np.ndarray, dx: float) -> np.ndarray:
    """Second derivative with periodic wrap: f_xx."""
    return (np.roll(f, -1) - 2.0 * f + np.roll(f, 1)) / (dx * dx)

def trapz_uniform(f: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal rule on uniform grid."""
    return float(np.trapz(f, x))

# ------------------------- Fisher quotient Q = -R_xx/R -------------------------

def fisher_Q(rho: np.ndarray, dx: float, eps: float = 1e-14) -> np.ndarray:
    """Compute Q = - ∂xx sqrt(rho) / sqrt(rho) with positivity floor."""
    rho_eff = np.maximum(rho, eps)
    R = np.sqrt(rho_eff)
    R_xx = dxx_periodic(R, dx)
    return - R_xx / R

# ------------------------- functional derivatives ------------------------------

def dH_dS(rho: np.ndarray, S: np.ndarray, m: float, dx: float) -> np.ndarray:
    """δH/δS = - ∂x( rho * S_x / m )."""
    Sx = dx_periodic(S, dx)
    flux = rho * Sx / m
    return - dx_periodic(flux, dx)

def dH_drho(rho: np.ndarray, S: np.ndarray, m: float, alpha: float,
            V: np.ndarray, dx: float) -> np.ndarray:
    """δH/δρ = |S_x|^2/(2m) + V - alpha * Q."""
    Sx = dx_periodic(S, dx)
    return (Sx**2) / (2.0 * m) + V - alpha * fisher_Q(rho, dx)

def dP_drho(S: np.ndarray, dx: float) -> np.ndarray:
    """δP/δρ = S_x."""
    return dx_periodic(S, dx)

def dP_dS(rho: np.ndarray, dx: float) -> np.ndarray:
    """δP/δS = - ∂x ρ (periodic IBP)."""
    return - dx_periodic(rho, dx)

def dK_drho(x: np.ndarray, S: np.ndarray, m: float, t: float, dx: float) -> np.ndarray:
    """δK/δρ = m x - t S_x."""
    Sx = dx_periodic(S, dx)
    return m * x - t * Sx

def dK_dS(rho: np.ndarray, t: float, dx: float) -> np.ndarray:
    """δK/δS = t ∂x ρ (since δK/δS = -t δP/δS)."""
    return t * dx_periodic(rho, dx)

# ------------------------- Poisson bracket -------------------------------------

def bracket(F_rho: np.ndarray, F_S: np.ndarray,
            G_rho: np.ndarray, G_S: np.ndarray, x: np.ndarray) -> float:
    """{F,G} = ∫ (F_rho G_S - F_S G_rho) dx."""
    integrand = F_rho * G_S - F_S * G_rho
    return trapz_uniform(integrand, x)

# ------------------------- main ------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Bargmann algebra closure test (exact per paper)")
    ap.add_argument("--L", type=float, default=20.0, help="Half-domain size; domain is [-L, L)")
    ap.add_argument("--N", type=int, default=20001, help="Grid points (periodic, uniform)")
    ap.add_argument("--m", type=float, default=1.3, help="Mass m")
    ap.add_argument("--hbar", type=float, default=1.0, help="Planck constant ℏ")
    ap.add_argument("--sigma", type=float, default=1.5, help="Gaussian width for ρ")
    ap.add_argument("--k", type=float, default=0.7, help="Linear phase slope in S")
    ap.add_argument("--t", type=float, default=0.0, help="Time parameter in K")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for optional tiny perturbations")
    ap.add_argument("--perturb", action="store_true", help="Add tiny smooth perturbations to S for generality")
    args = ap.parse_args()

    # Grid and parameters
    L = args.L
    N = args.N
    x = np.linspace(-L, L, N, endpoint=False)
    dx = x[1] - x[0]

    m = args.m
    hbar = args.hbar
    alpha = hbar**2 / (2.0 * m)
    t = args.t

    # Translation-invariant potential
    V = np.zeros_like(x)

    # Normalised Gaussian density (periodic truncation is safe at this L and sigma)
    sigma = args.sigma
    rho = np.exp(-(x**2) / (sigma**2))
    rho /= trapz_uniform(rho, x)  # ∫ρ dx = 1

    # Phase S: linear term + optional small smooth perturbation
    S = args.k * x
    if args.perturb:
        rng = np.random.default_rng(args.seed)
        # add a tiny smooth bump (damped cubic) to avoid accidental symmetries
        S += 0.03 * x**3 * np.exp(-0.1 * x**2) * (1.0 + 0.05 * rng.standard_normal())

    # Functional derivatives
    H_rho = dH_drho(rho, S, m, alpha, V, dx)
    H_S   = dH_dS(rho, S, m, dx)

    P_rho = dP_drho(S, dx)
    P_S   = dP_dS(rho, dx)

    K_rho = dK_drho(x, S, m, t, dx)
    K_S   = dK_dS(rho, t, dx)

    # Poisson brackets
    HP = bracket(H_rho, H_S, P_rho, P_S, x)     # {H,P}
    HK = bracket(H_rho, H_S, K_rho, K_S, x)     # {H,K}
    PK = bracket(P_rho, P_S, K_rho, K_S, x)     # {P,K}

    # Charges and expectations
    Sx = dx_periodic(S, dx)
    P_val = trapz_uniform(rho * Sx, x)
    mass_int = trapz_uniform(rho, x)            # should be 1
    expected_HP = 0.0
    expected_HK = -P_val
    expected_PK = -m * mass_int

    # Errors
    err_HP = abs(HP - expected_HP)
    err_HK = abs(HK - expected_HK)
    err_PK = abs(PK - expected_PK)

    rel_HP = err_HP / (abs(expected_HP) + 1e-30)  # expected zero: show abs too
    rel_HK = err_HK / (abs(expected_HK) + 1e-30)
    rel_PK = err_PK / (abs(expected_PK) + 1e-30)

    # Report
    print("\nBargmann algebra closure test (exact per paper)")
    print("=================================================")
    print(f"Grid: N={N}, dx={dx:.6e}, domain=[{-L},{L})")
    print(f"m={m}, hbar={hbar}, alpha=ℏ²/(2m)={alpha:.12g}, t={t}")
    print(f"rho: Gaussian σ={sigma}, normalised ∫ρ dx={mass_int:.15g}")
    print(f"S: linear slope k={args.k}" + (" + smooth perturbation" if args.perturb else ""))
    print("\nComputed brackets:")
    print(f"  {{H,P}} = {HP:.15e}   (expected 0)")
    print(f"  {{H,K}} = {HK:.15e}   (expected = -P = {expected_HK:.15e})")
    print(f"  {{P,K}} = {PK:.15e}   (expected = -m ∫ρ dx = {expected_PK:.15e})")
    print("\nCharges:")
    print(f"  P = ∫ρ S_x dx = {P_val:.15e}")
    print(f"  ∫ρ dx = {mass_int:.15e}")
    print("\nAbsolute errors:")
    print(f"  |{{H,P}}-0|   = {err_HP:.15e}")
    print(f"  |{{H,K}}+P|   = {err_HK:.15e}")
    print(f"  |{{P,K}}+m|   = {err_PK:.15e}")
    print("\nRelative errors:")
    print(f"  rel {{H,P}} (abs since expected=0) = {rel_HP:.3e}")
    print(f"  rel {{H,K}}                       = {rel_HK:.3e}")
    print(f"  rel {{P,K}}                       = {rel_PK:.3e}")

    # Simple pass/fail suggestion (tolerances tuned for N~2e4; relax for coarser grids)
    tol_abs = 1e-11
    tol_rel = 1e-10
    ok = (err_HP < tol_abs) and (rel_HK < tol_rel) and (rel_PK < tol_rel)
    print("\nConclusion:", "PASS" if ok else "CHECK TOLERANCES / RESOLUTION")

if __name__ == "__main__":
    main()
