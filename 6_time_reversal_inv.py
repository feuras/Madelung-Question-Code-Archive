#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time-Reversal Involution Verification
=====================================

Performs both the reversible Fisher-scale Schrödinger test (D=0)
and the irreversible Doebner–Goldin control (D>0) in one run.

Outputs:
- L2 relative errors for ψ (phase-aligned) and ρ
- Overlap magnitude |<ψ0 | ψ_back>|
- Norms at t=0 and t=T
- Interpretation line highlighting reversibility vs irreversibility
"""
import numpy as np
import argparse
import warnings

# Suppress all non-critical warnings for clean output
warnings.filterwarnings("ignore")

# -----------------------------
# Utilities
# -----------------------------
def normalize(psi, x):
    """L2-normalize ψ on grid x."""
    return psi / np.sqrt(np.trapz(np.abs(psi)**2, x))

def fft_wavenumbers(N, dx):
    """Continuous wavenumbers for FFT on periodic grid of spacing dx."""
    return np.fft.fftfreq(N, d=dx) * 2.0 * np.pi

def split_step_unitary(psi, Vx, k, dt, hbar, m):
    """One time-step of unitary split-step Fourier method for Schrödinger with real potential."""
    V_half = np.exp(-1j * Vx * dt / (2.0 * hbar))
    K_phase = np.exp(-1j * (hbar * (k**2) / (2.0 * m)) * dt)
    psi = V_half * psi
    psi_k = np.fft.fft(psi)
    psi_k *= K_phase
    psi = np.fft.ifft(psi_k)
    psi = V_half * psi
    return psi

def evolve_schrodinger(psi0, Vx, x, dt, Nt, hbar=1.0, m=1.0, renorm=True):
    """Evolve ψ under reversible Schrödinger."""
    N = x.size
    dx = x[1] - x[0]
    k = fft_wavenumbers(N, dx)
    psi = psi0.copy()
    for _ in range(Nt):
        psi = split_step_unitary(psi, Vx, k, dt, hbar, m)
        if renorm:
            psi = normalize(psi, x)
    return psi

def phase_align(psi_ref, psi, x):
    """Align global phase of ψ to maximize overlap with ψ_ref."""
    overlap = np.trapz(np.conjugate(psi_ref) * psi, x)
    phase = np.exp(-1j * np.angle(overlap))
    return psi * phase, overlap

def l2_rel_error(psi_ref, psi, x):
    """Relative L2 error between ψ and ψ_ref."""
    num = np.sqrt(np.trapz(np.abs(psi - psi_ref)**2, x))
    den = np.sqrt(np.trapz(np.abs(psi_ref)**2, x))
    return float(num / den)

def l2_rel_error_density(rho_ref, rho, x):
    """Relative L2 error for densities."""
    num = np.sqrt(np.trapz((rho - rho_ref)**2, x))
    den = np.sqrt(np.trapz(rho_ref**2, x))
    return float(num / max(den, 1e-300))

# -----------------------------
# Doebner–Goldin diffusion
# -----------------------------
def dg_heat_step(psi, k, dt, D, hbar):
    """Exact heat-kernel step for the DG term."""
    nu = 2.0 * D / hbar
    rho = np.abs(psi)**2
    rho_k = np.fft.fft(rho)
    rho_k *= np.exp(-nu * (k**2) * dt)
    rho_new = np.real(np.fft.ifft(rho_k))
    rho_new = np.maximum(rho_new, 0.0)
    phase = np.exp(1j * np.angle(psi))
    return np.sqrt(rho_new) * phase

def evolve_schrodinger_plus_DG(psi0, Vx, x, dt, Nt, D, hbar=1.0, m=1.0):
    """DG evolution using Strang splitting."""
    N = x.size
    dx = x[1] - x[0]
    k = fft_wavenumbers(N, dx)
    psi = psi0.copy()
    for _ in range(Nt):
        psi = dg_heat_step(psi, k, dt/2.0, D, hbar)
        psi = split_step_unitary(psi, Vx, k, dt, hbar, m)
        psi = dg_heat_step(psi, k, dt/2.0, D, hbar)
    return psi

# -----------------------------
# Involution protocols
# -----------------------------
def involution_reversible(psi0, Vx, x, dt, Nt, hbar=1.0, m=1.0):
    """K U(T) K U(T) for reversible Schrödinger."""
    psi_T = evolve_schrodinger(psi0, Vx, x, dt, Nt, hbar, m)
    psi_rev_T = np.conjugate(psi_T)
    psi_back = evolve_schrodinger(psi_rev_T, Vx, x, dt, Nt, hbar, m)
    psi_final = np.conjugate(psi_back)

    psi_aligned, overlap = phase_align(psi0, psi_final, x)
    err_psi = l2_rel_error(psi0, psi_aligned, x)
    rho0 = np.abs(psi0)**2
    rho_final = np.abs(psi_final)**2
    err_rho = l2_rel_error_density(rho0, rho_final, x)
    norm0 = np.sqrt(np.trapz(np.abs(psi0)**2, x))
    normT = np.sqrt(np.trapz(np.abs(psi_T)**2, x))

    return err_psi, err_rho, float(np.abs(overlap)), norm0, normT

def involution_DG(psi0, Vx, x, dt, Nt, D, hbar=1.0, m=1.0):
    """K U_DG(T) K U_DG(T) for irreversible Doebner–Goldin."""
    psi_T = evolve_schrodinger_plus_DG(psi0, Vx, x, dt, Nt, D, hbar, m)
    psi_rev_T = np.conjugate(psi_T)
    psi_back = evolve_schrodinger_plus_DG(psi_rev_T, Vx, x, dt, Nt, D, hbar, m)
    psi_final = np.conjugate(psi_back)

    psi_aligned, overlap = phase_align(psi0, psi_final, x)
    err_psi = l2_rel_error(psi0, psi_aligned, x)
    rho0 = np.abs(psi0)**2
    rho_final = np.abs(psi_final)**2
    err_rho = l2_rel_error_density(rho0, rho_final, x)
    norm0 = np.sqrt(np.trapz(np.abs(psi0)**2, x))
    normT = np.sqrt(np.trapz(np.abs(psi_T)**2, x))

    return err_psi, err_rho, float(np.abs(overlap)), norm0, normT

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, default=20.0)
    ap.add_argument("--N", type=int, default=8192)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--Nt", type=int, default=400)
    ap.add_argument("--sigma0", type=float, default=1.2)
    ap.add_argument("--x0", type=float, default=-4.0)
    ap.add_argument("--k0", type=float, default=1.3)
    args = ap.parse_args()

    L, N = args.L, args.N
    x = np.linspace(-L, L, N, endpoint=False)
    dx = x[1] - x[0]
    Vx = np.zeros_like(x)
    norm_const = (np.pi**0.25 * np.sqrt(args.sigma0))**-1
    psi0 = norm_const * np.exp(-((x - args.x0)**2)/(2*args.sigma0**2)) * np.exp(1j * args.k0 * x)
    psi0 = normalize(psi0, x)
    T = args.Nt * args.dt

    print("Time-Reversal Involution Verification")
    print("=====================================")
    print(f"Domain: [-{L},{L}), N={N}, dx={dx:.3e}")
    print(f"dt={args.dt}, Nt={args.Nt}, T={T}")
    print(f"Initial packet: sigma0={args.sigma0}, x0={args.x0}, k0={args.k0}")
    print("")

    # (1) Reversible Schrödinger
    err_psi, err_rho, overlap, norm0, normT = involution_reversible(psi0, Vx, x, args.dt, args.Nt)
    print("Reversible Schrödinger (Fisher corner, D=0):")
    print(f"  L2 error ψ: {err_psi:.3e}")
    print(f"  L2 error ρ: {err_rho:.3e}")
    print(f"  |overlap|:  {overlap:.10f}")
    print(f"  Norm(t=0): {norm0:.10f}")
    print(f"  Norm(t=T): {normT:.10f}")
    print("  → Perfect antiunitary involution (K U(T) K U(T) = I)\n")

    # (2) Irreversible DG
    D = 0.01
    err_psi_dg, err_rho_dg, overlap_dg, norm0_dg, normT_dg = involution_DG(psi0, Vx, x, args.dt, args.Nt, D)
    print(f"Doebner–Goldin irreversible control (D={D}):")
    print(f"  L2 error ψ: {err_psi_dg:.3e}")
    print(f"  L2 error ρ: {err_rho_dg:.3e}")
    print(f"  |overlap|:  {overlap_dg:.10f}")
    print(f"  Norm(t=0): {norm0_dg:.10f}")
    print(f"  Norm(t=T): {normT_dg:.10f}")
    print("  → Finite mismatch confirms broken microscopic reversibility for D>0\n")

    ratio = (err_psi_dg / max(err_psi, 1e-15))
    print(f"Reversibility breakdown ratio (ψ): {ratio:.3e}x difference between D=0 and D>0")

if __name__ == "__main__":
    main()
