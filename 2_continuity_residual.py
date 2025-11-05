#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Continuity residual baseline (Test 2)
====================================
Verifies that the numerical scheme satisfies the continuity identity

    R_cont = || rho_t + div(rho * gradS / m) ||_L2  /  || rho ||_L2

with rho_t computed from the Hermitian Hamiltonian H to respect the
sign convention in the paper:

    rho_t = (2/hbar) * Im(psi.conj() * (H psi))

Systems:
  1) HO ground state (stationary, real)
  2) HO first excited (stationary, real with node)
  3) Free Gaussian packet at t=0 (not stationary; continuity still holds)

Expected: R_cont at numerical floor for all three when grid is adequate.
"""
import sys, numpy as np, argparse
np.seterr(all="raise")

# ---------- Finite difference operators ----------

def gradient_central(f: np.ndarray, dx: float) -> np.ndarray:
    """Second-order central gradient with one-sided boundaries."""
    g = np.empty_like(f, dtype=float)
    g[1:-1] = (f[2:] - f[:-2]) * (0.5/dx)
    g[0]  = (-3.0*f[0] + 4.0*f[1] - f[2]) / (2.0*dx)
    g[-1] = ( 3.0*f[-1] - 4.0*f[-2] + f[-3]) / (2.0*dx)
    return g

def laplacian_central(f: np.ndarray, dx: float) -> np.ndarray:
    """Second derivative with second-order accuracy and one-sided boundaries."""
    L = np.empty_like(f, dtype=complex if np.iscomplexobj(f) else float)
    invdx2 = 1.0/(dx*dx)
    L[1:-1] = (f[2:] - 2.0*f[1:-1] + f[:-2]) * invdx2
    # One-sided second-order second derivative at boundaries
    # f''(x0) ≈ (2f0 - 5f1 + 4f2 - f3)/dx^2
    L[0]  = (2.0*f[0]  - 5.0*f[1]  + 4.0*f[2]  - f[3])  * invdx2
    L[-1] = (2.0*f[-1] - 5.0*f[-2] + 4.0*f[-3] - f[-4]) * invdx2
    return L

# ---------- Potentials and wavefunctions ----------

def V_free(x: np.ndarray) -> np.ndarray:
    return np.zeros_like(x)

def V_ho(x: np.ndarray) -> np.ndarray:
    # m = omega = hbar = 1 in closed forms below, so V = x^2 / 2
    return 0.5 * x*x

def ho_eigenstate(n: int, x: np.ndarray) -> np.ndarray:
    """Harmonic oscillator eigenstates for m=omega=hbar=1."""
    norm0 = np.pi**(-0.25)
    psi0 = norm0 * np.exp(-0.5*x*x)
    if n == 0:
        return psi0.astype(complex)
    if n == 1:
        return (np.sqrt(2.0) * x * psi0).astype(complex)
    raise ValueError("Only n=0 and n=1 implemented.")

def free_gaussian_packet(x: np.ndarray, sigma: float, x0: float, k: float) -> np.ndarray:
    """Free Gaussian packet at t=0 with plane-wave phase."""
    A = (np.pi**0.25 * np.sqrt(sigma))**-1
    return (A * np.exp(-0.5*((x - x0)/sigma)**2) * np.exp(1j * k * x)).astype(complex)

# ---------- Hamiltonian and residual ----------

def apply_H(psi: np.ndarray, x: np.ndarray, m: float, hbar: float, Vx: np.ndarray) -> np.ndarray:
    dx = x[1] - x[0]
    lap = laplacian_central(psi, dx)
    return -(hbar*hbar)/(2.0*m) * lap + Vx * psi

def continuity_residual(x: np.ndarray, psi: np.ndarray, m: float, hbar: float, Vx: np.ndarray) -> float:
    """R_cont = || rho_t + div j ||_L2 / ||rho||_L2 with rho_t from H, j from current."""
    dx = x[1] - x[0]
    # Ensure normalisation on the given grid
    mass = np.trapezoid(np.abs(psi)**2, x)
    psi = psi / np.sqrt(mass)

    rho = np.abs(psi)**2

    # Current j = (hbar/m) Im(psi* dpsi/dx)
    dpsi_dx = gradient_central(psi.real, dx) + 1j * gradient_central(psi.imag, dx)
    j = (hbar/m) * np.imag(np.conjugate(psi) * dpsi_dx)
    div_j = gradient_central(j.astype(float), dx)

    # rho_t from Hermitian H, sign matches paper's sign note
    Hpsi = apply_H(psi, x, m, hbar, Vx)
    rho_t = (2.0/hbar) * np.imag(np.conjugate(psi) * Hpsi)

    num = np.sqrt(np.trapezoid((rho_t + div_j)**2, x))
    den = np.sqrt(np.trapezoid(rho**2, x))
    if not np.isfinite(den) or den <= 0.0:
        raise FloatingPointError(f"Invalid denominator in R_cont: {den}")
    return float(num / den)

# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Continuity residual baseline (Test 2)")
    parser.add_argument("--xmin", type=float, default=-12.0)
    parser.add_argument("--xmax", type=float, default=12.0)
    parser.add_argument("--N", type=int, default=20001)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--x0", type=float, default=0.0)
    parser.add_argument("--k", type=float, default=1.0)
    args = parser.parse_args()

    m = 1.0
    hbar = 1.0
    x = np.linspace(args.xmin, args.xmax, args.N, dtype=float)
    dx = x[1] - x[0]

    systems = [
        ("HO_ground", ho_eigenstate(0, x), V_ho(x)),
        ("HO_excited", ho_eigenstate(1, x), V_ho(x)),
        ("FREE_packet", free_gaussian_packet(x, sigma=args.sigma, x0=args.x0, k=args.k), V_free(x)),
    ]

    print("\nContinuity residual baseline (Test 2)\n=====================================")
    print(f"Grid: N={args.N}, dx={dx:.3e}, domain=[{args.xmin},{args.xmax}]\n")

    for name, psi, Vx in systems:
        R = continuity_residual(x, psi, m, hbar, Vx)
        print(f"System: {name:12s}   R_cont = {R:.3e}")

    print("\nExpected: R_cont at numerical floor, confirming the discrete continuity law.")

if __name__ == "__main__":
    main()
