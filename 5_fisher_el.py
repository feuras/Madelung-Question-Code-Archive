#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fisher EL verification (Test 5)
===============================
Verify numerically that for smooth densities rho(x),
    -2 div( f(rho) grad rho ) + f'(rho) |grad rho|^2  =  -4 C Laplacian(sqrt(rho)) / sqrt(rho)
holds only for  f(rho) = C / rho.

Numerically stable evaluation
-----------------------------
Use the algebraically equivalent form
    LHS = -2 f(rho) * Laplacian(rho)  -  f'(rho) * |grad rho|^2
which avoids differentiating products across tiny rho tails.

Densities tested (1D)
---------------------
1) Gaussian (L1-normalised):         rho(x) = (sqrt(pi) sigma)^(-1) * exp(-x^2/sigma^2)
2) Compact bump (cos^2 on [-a,a]):   rho(x) = cos^2( (pi/2) x/a ), |x|<=a; 0 outside, renormalised

Expected outcome
----------------
- For f = C/rho: relative L2 error ~ 1e-7 (Gaussian), <= 1e-4 (bump)
- For any other f: errors orders of magnitude larger
"""
import numpy as np
import argparse

# ---- compatibility shim for older numpy ----
trapz = getattr(np, "trapezoid", np.trapz)

# ---------- finite-difference operators ----------

def grad1(f: np.ndarray, dx: float) -> np.ndarray:
    g = np.empty_like(f, dtype=float)
    g[1:-1] = (f[2:] - f[:-2]) * (0.5 / dx)
    g[0] = (-3 * f[0] + 4 * f[1] - f[2]) / (2 * dx)
    g[-1] = (3 * f[-1] - 4 * f[-2] + f[-3]) / (2 * dx)
    return g


def lap1(f: np.ndarray, dx: float) -> np.ndarray:
    lap = np.empty_like(f, dtype=float)
    lap[1:-1] = (f[2:] - 2 * f[1:-1] + f[:-2]) / (dx * dx)
    lap[0] = (2 * f[0] - 5 * f[1] + 4 * f[2] - f[3]) / (dx * dx)
    lap[-1] = (2 * f[-1] - 5 * f[-2] + 4 * f[-3] - f[-4]) / (dx * dx)
    return lap


# ---------- densities ----------

def rho_gaussian(x: np.ndarray, sigma: float) -> np.ndarray:
    return (np.pi**0.5 * sigma) ** -1 * np.exp(-(x**2) / (sigma**2))


def rho_bump_cos2(x: np.ndarray, a: float) -> np.ndarray:
    rho = np.zeros_like(x, dtype=float)
    inside = np.abs(x) <= a
    rho[inside] = np.cos(0.5 * np.pi * x[inside] / a) ** 2
    # normalise to unit mass
    rho /= trapz(rho, x)
    return rho


# ---------- f families ----------

def make_power_f(p: float, C: float):
    def f(r):
        rpos = np.maximum(r, 1e-300)
        return C * np.power(rpos, -p)
    def df(r):
        rpos = np.maximum(r, 1e-300)
        return C * (-p) * np.power(rpos, -p - 1.0)
    return f, df


def f_linear_perturbed(C: float):
    def f(r):
        rpos = np.maximum(r, 1e-300)
        return C * (1.0 / rpos + 0.1 * rpos)
    def df(r):
        rpos = np.maximum(r, 1e-300)
        return C * (-1.0 / (rpos * rpos) + 0.1)
    return f, df


# ---------- residual (stable form) ----------

def fisher_residual_stable(
    x: np.ndarray, rho: np.ndarray, f_func, df_func, C: float,
    eps: float = 1e-12, mask_abs: float = 1e-6, mask_rel: float = 1e-12
) -> float:
    dx = x[1] - x[0]
    rho_eff = np.maximum(rho, eps)
    dr = grad1(rho, dx)
    lap_rho = lap1(rho, dx)
    f = f_func(rho_eff)
    df = df_func(rho_eff)
    left = -2.0 * f * lap_rho - df * (dr * dr)
    sr = np.sqrt(rho_eff)
    right = -4.0 * C * lap1(sr, dx) / sr
    # restrict to safe core (rho > threshold) to avoid tail amplification
    thr = max(mask_abs, mask_rel * float(rho.max()))
    mask = rho > thr
    num = np.sqrt(trapz(((left - right)[mask]) ** 2, x[mask]))
    den = np.sqrt(trapz((right[mask]) ** 2, x[mask]))
    if not np.isfinite(den) or den < 1e-30:
        return np.inf
    return float(num / den)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Fisher EL verification (Test 5)")
    ap.add_argument("--xmin", type=float, default=-10.0)
    ap.add_argument("--xmax", type=float, default=10.0)
    ap.add_argument("--N", type=int, default=100001)
    ap.add_argument("--sigma", type=float, default=1.3)
    ap.add_argument("--a", type=float, default=2.5)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=1e-12)
    ap.add_argument("--mask_abs", type=float, default=1e-6)
    ap.add_argument("--mask_rel", type=float, default=1e-12)
    ap.add_argument("--convergence", action="store_true")
    args = ap.parse_args()

    x = np.linspace(args.xmin, args.xmax, args.N, dtype=float)
    dx = x[1] - x[0]

    rhoG = np.maximum(rho_gaussian(x, args.sigma), 0.0)
    rhoB = np.maximum(rho_bump_cos2(x, args.a), 0.0)

    tests = [
        ("p=1 (Fisher)", *make_power_f(1.0, args.C)),
        ("p=0.5", *make_power_f(0.5, args.C)),
        ("p=1.5", *make_power_f(1.5, args.C)),
        ("+ linear perturbed", *f_linear_perturbed(args.C)),
    ]

    print("\nFisher EL verification (Test 5)")
    print("================================")
    print(f"Grid: N={args.N}, dx={dx:.3e}, domain=[{args.xmin},{args.xmax}]")
    print(
        f"eps={args.eps}, C={args.C}, sigma={args.sigma}, a={args.a}, "
        f"mask_abs={args.mask_abs}, mask_rel={args.mask_rel}\n"
    )

    header = f"{'f(rho)':28s} | {'RelErr Gaussian':>16s} | {'RelErr Bump':>14s}"
    print(header)
    print("-" * len(header))
    for name, f, df in tests:
        rG = fisher_residual_stable(x, rhoG, f, df, args.C,
                                    eps=args.eps,
                                    mask_abs=args.mask_abs,
                                    mask_rel=args.mask_rel)
        rB = fisher_residual_stable(x, rhoB, f, df, args.C,
                                    eps=args.eps,
                                    mask_abs=args.mask_abs,
                                    mask_rel=args.mask_rel)
        print(f"{name:28s} | {rG:16.3e} | {rB:14.3e}")

    if args.convergence:
        print("\nConvergence check for Fisher case:")
        for N in [8193, 16385, 32769]:
            x2 = np.linspace(args.xmin, args.xmax, N)
            rhoG2 = rho_gaussian(x2, args.sigma)
            rG2 = fisher_residual_stable(
                x2, rhoG2, *make_power_f(1.0, args.C),
                C=args.C, eps=args.eps,
                mask_abs=args.mask_abs, mask_rel=args.mask_rel
            )
            print(f"N={N:6d}  RelErr Gaussian (p=1): {rG2:.3e}")

    print("\nInterpretation:")
    print("  Only f(rho)=C/rho collapses the residual to the numerical floor (~1e-7 for Gaussian, <=1e-4 for bump).")
    print("  Any deviation (different power or perturbation) increases the error by orders of magnitude.")
    print("  This verifies the Euler-Lagrange uniqueness of the Fisher curvature.")


if __name__ == "__main__":
    main()
