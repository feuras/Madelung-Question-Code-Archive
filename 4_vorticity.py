#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vorticity and Circulation Quantisation (Test 4)
===============================================

Reproduce quantised circulation of hydrodynamic vortices in the
Madelung form of the Schrödinger equation:

    ψ(r,θ) = f(r) * exp(i n θ)
    v = j / ρ,  j = (ħ/m) Im(ψ* ∇ψ)

Verify:
    ∮ v · dl = 2π n ħ / m
    ∬ (∇×v)_z dA = 2π n ħ / m

Both confirm δ-supported vorticity and integer-quantised circulation.
"""
import numpy as np
import argparse

# ---------- Finite-difference operators ----------

def gradient_central_2d(F, dx, dy):
    Fx = np.empty_like(F, float)
    Fy = np.empty_like(F, float)
    Fx[:, 1:-1] = (F[:, 2:] - F[:, :-2]) * (0.5 / dx)
    Fy[1:-1, :] = (F[2:, :] - F[:-2, :]) * (0.5 / dy)
    Fx[:, 0]  = (-3*F[:,0] + 4*F[:,1] - F[:,2]) / (2*dx)
    Fx[:,-1]  = ( 3*F[:,-1] - 4*F[:,-2] + F[:,-3]) / (2*dx)
    Fy[0,:]   = (-3*F[0,:] + 4*F[1,:] - F[2,:]) / (2*dy)
    Fy[-1,:]  = ( 3*F[-1,:] - 4*F[-2,:] + F[-3,:]) / (2*dy)
    return Fx, Fy

def curl_z(vx, vy, dx, dy):
    dvy_dx, _ = gradient_central_2d(vy, dx, dy)
    _, dvx_dy = gradient_central_2d(vx, dx, dy)
    return dvy_dx - dvx_dy

# ---------- Vortex field ----------

def vortex_wavefunction(X, Y, n, sigma):
    R = np.sqrt(X*X + Y*Y)
    Theta = np.arctan2(Y, X)
    f = (R**abs(n)) * np.exp(-R**2 / (2*sigma**2))
    return f * np.exp(1j * n * Theta)

def velocity_from_psi(psi, dx, dy, hbar, m):
    dψr_dx, dψr_dy = gradient_central_2d(psi.real, dx, dy)
    dψi_dx, dψi_dy = gradient_central_2d(psi.imag, dx, dy)
    dψ_dx = dψr_dx + 1j * dψi_dx
    dψ_dy = dψr_dy + 1j * dψi_dy
    rho = np.abs(psi)**2
    jx = (hbar/m) * np.imag(np.conjugate(psi) * dψ_dx)
    jy = (hbar/m) * np.imag(np.conjugate(psi) * dψ_dy)
    eps = 1e-14 * rho.max()
    mask = rho > eps
    vx = np.zeros_like(rho)
    vy = np.zeros_like(rho)
    vx[mask] = jx[mask] / rho[mask]
    vy[mask] = jy[mask] / rho[mask]
    return vx, vy, rho

# ---------- Integrals ----------

def path_integral_circulation(vx, vy, x, y, radius, num_pts=4096):
    t = np.linspace(0, 2*np.pi, num_pts, endpoint=False)
    eps = 1e-12 * radius
    xs = (radius - eps) * np.cos(t)
    ys = (radius - eps) * np.sin(t)
    dxg = x[1] - x[0]
    dyg = y[1] - y[0]
    ix = (xs - x[0]) / dxg
    iy = (ys - y[0]) / dyg
    i0 = np.clip(ix.astype(int), 0, len(x)-2)
    j0 = np.clip(iy.astype(int), 0, len(y)-2)
    fx = ix - i0
    fy = iy - j0
    def bilin(F):
        return ((1-fx)*(1-fy)*F[j0, i0] +
                fx*(1-fy)*F[j0, i0+1] +
                (1-fx)*fy*F[j0+1, i0] +
                fx*fy*F[j0+1, i0+1])
    vxs = bilin(vx)
    vys = bilin(vy)
    dx_dl = -radius * np.sin(t)
    dy_dl =  radius * np.cos(t)
    return np.trapz(vxs*dx_dl + vys*dy_dl, t)

# ---------- Main ----------

def main():
    p = argparse.ArgumentParser(description="Vorticity and circulation quantisation (Test 4)")
    p.add_argument("--L", type=float, default=6.0)
    p.add_argument("--N", type=int, default=1001)
    p.add_argument("--sigma", type=float, default=1.2)
    p.add_argument("--nlist", type=str, default="-2,-1,0,1,2")
    p.add_argument("--radii", type=str, default="0.6,1.2,2.4,3.6")
    p.add_argument("--hbar", type=float, default=1.0)
    p.add_argument("--ħ", dest="hbar", type=float)
    p.add_argument("--m", type=float, default=1.0)
    p.add_argument("--tol", type=float, default=1e-2)
    args = p.parse_args()

    n_list = [int(s) for s in args.nlist.split(",") if s.strip()]
    radii = [float(s) for s in args.radii.split(",") if s.strip()]
    x = np.linspace(-args.L, args.L, args.N)
    y = np.linspace(-args.L, args.L, args.N)
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    X, Y = np.meshgrid(x, y, indexing="xy")

    print(f"\nVorticity and Circulation Quantisation (Test 4)")
    print("="*49)
    print(f"Grid: {args.N}×{args.N}, dx=dy={dx:.4f}, L={args.L}, σ={args.sigma}")
    print(f"hbar={args.hbar}, m={args.m}\n")

    hdr = f"{'n':>3} | {'R':>4} | {'target(2πnħ/m)':>18} | {'∮v·dl':>14} | {'err':>9} | {'∬ω_z dA':>14} | {'err':>9}"
    print(hdr)
    print("-"*len(hdr))

    two_pi_hbar_over_m = 2*np.pi*args.hbar/args.m
    ok_all = True

    for n in n_list:
        psi = vortex_wavefunction(X, Y, n, args.sigma)
        vx, vy, rho = velocity_from_psi(psi, dx, dy, args.hbar, args.m)
        omega_z = curl_z(vx, vy, dx, dy)

        for R in radii:
            circ = path_integral_circulation(vx, vy, x, y, R)
            mask = (X**2 + Y**2) <= (R**2)
            area = np.sum(omega_z[mask])*dx*dy
            target = two_pi_hbar_over_m*n
            err1 = circ - target
            err2 = area - target
            print(f"{n:3d} | {R:4.1f} | {target:18.12f} | {circ:14.9f} | {err1:9.3e} | {area:14.9f} | {err2:9.3e}")
            if abs(err1) > args.tol or abs(err2) > args.tol:
                ok_all = False

    print("\nResult:")
    print("  Circulation and vorticity integrals quantised to 2πnħ/m.")
    print(f"  {'PASS' if ok_all else 'FAIL'} (tolerance ±{args.tol})")

if __name__ == "__main__":
    main()
