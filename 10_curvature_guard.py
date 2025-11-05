#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
appendix_curvature_guard.py  (rigour-enhanced)

Scope and limits:
- Spatial dimension is 2D for Fisher/curvature checks (periodic box), serving as a clean testbed.
- Curved background is a smooth diagonal conformal metric g_ij = a(x,y)^2 δ_ij (no off-diagonals).
- Klein–Gordon diagnostic is 1D in space + time, linear, with periodic BCs.
- All operators are matched (central differences and their exact adjoints) under periodic BCs.
- No statements about dynamics on curved spacetime are made; these are geometric and diagnostic checks.

This script exits with nonzero status if any check exceeds strict tolerances.
"""

import numpy as np
import math
import sys
from dataclasses import dataclass

np.set_printoptions(precision=6, suppress=True)

# ------------------------- Discrete operator helpers (periodic) -------------------------
def Dx(u, dx):
    return (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2*dx)

def Dy(u, dy):
    return (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2*dy)

def DxT(v, dx):
    # exact adjoint in periodic L2: D^T = -D
    return -(np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2*dx)

def DyT(v, dy):
    return -(np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2*dy)

def Lap(u, dx, dy):
    return ((np.roll(u, -1, axis=0) - 2*u + np.roll(u, 1, axis=0))/dx**2 +
            (np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1))/dy**2)

# ------------------------- Grids -------------------------
def grid2d(nx=128, ny=128, Lx=1.0, Ly=1.0):
    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)
    dx = x[1]-x[0]; dy = y[1]-y[0]
    X, Y = np.meshgrid(x, y, indexing='ij')
    return X, Y, dx, dy

# ------------------------- Positivity-safe perturbation -------------------------
def make_eta_safe(rho, eta, dx, dy, sqrtg=None, eps=1e-4):
    # zero-mean perturbation (Euclidean or curved measure), then scale so that rho ± eps*eta ≥ rho_min/2
    if sqrtg is None:
        eta = eta - eta.mean()
    else:
        w = sqrtg
        eta = eta - (np.sum(eta*w)*dx*dy)/np.sum(w*dx*dy)
    # normalise magnitude
    if eta.std() == 0:
        return np.zeros_like(eta)
    eta = eta * (0.1*np.mean(rho)/eta.std())
    # positivity guard
    rho_min = float(rho.min())
    if rho_min <= 0:
        raise ValueError("rho must be strictly positive in these checks.")
    # compute scale to ensure positivity with margin 1/2
    max_step = min((rho_min/2.0) / (eps*float(np.max(np.abs(eta))) + 1e-16), 1.0)
    return eta * max_step

# ------------------------- 1) Flat Fisher variation -------------------------
def fisher_flat_F(rho, dx, dy):
    u = np.sqrt(rho)  # positivity enforced by make_eta_safe
    return np.sum(Dx(u, dx)**2 + Dy(u, dy)**2) * dx * dy

def fisher_flat_var_discrete(rho, dx, dy):
    u = np.sqrt(rho)
    gx, gy = Dx(u, dx), Dy(u, dy)
    dF_du = 2*(DxT(gx, dx) + DyT(gy, dy))          # exact discrete adjoint
    return dF_du / (2*u)                            # chain rule du/dρ = 1/(2√ρ)

def check_flat_fisher(seed=0, nx=128, ny=128):
    rng = np.random.default_rng(seed)
    X, Y, dx, dy = grid2d(nx, ny)
    rho = np.exp(-((X-0.31)**2 + (Y-0.67)**2)/0.02) + 0.4*np.exp(-((X-0.72)**2 + (Y-0.28)**2)/0.03) + 0.05
    eta = make_eta_safe(rho, rng.standard_normal(rho.shape), dx, dy, sqrtg=None, eps=1e-4)
    eps = 1e-4
    fd = (fisher_flat_F(rho + eps*eta, dx, dy) - fisher_flat_F(rho - eps*eta, dx, dy))/(2*eps)
    var = fisher_flat_var_discrete(rho, dx, dy)
    inner = np.sum(var*eta)*dx*dy
    rel = abs(fd - inner)/max(1e-16, abs(fd))
    return fd, inner, rel

# ------------------------- 2) Curved Fisher variation on conformal metric -------------------------
def conformal_metric(X, Y, amp=0.25):
    a = 1.0 + amp*np.sin(2*np.pi*X)*np.sin(2*np.pi*Y)
    gxx = a**2; gyy = a**2; gxy = np.zeros_like(a)
    gdet = gxx*gyy - gxy**2
    return a, gxx, gyy, gxy, gdet

def covariant_grad(u, gxx, gyy, dx, dy):
    invgxx, invgyy = 1.0/gxx, 1.0/gyy
    ux, uy = Dx(u, dx), Dy(u, dy)
    return invgxx*ux, invgyy*uy

def covariant_div(vx, vy, gdet, dx, dy):
    sqrtg = np.sqrt(gdet)
    Vx, Vy = sqrtg*vx, sqrtg*vy
    return (Dx(Vx, dx) + Dy(Vy, dy))/sqrtg

def box_g(u, gxx, gyy, gdet, dx, dy):
    vx, vy = covariant_grad(u, gxx, gyy, dx, dy)
    return covariant_div(vx, vy, gdet, dx, dy)

def fisher_curved_F(rho, gxx, gyy, gdet, dx, dy):
    u = np.sqrt(rho)
    ux, uy = Dx(u, dx), Dy(u, dy)
    invgxx, invgyy = 1.0/gxx, 1.0/gyy
    sqrtg = np.sqrt(gdet)
    integrand = (invgxx*ux**2 + invgyy*uy**2)*sqrtg
    return np.sum(integrand)*dx*dy

def fisher_curved_var_discrete(rho, gxx, gyy, gdet, dx, dy):
    u = np.sqrt(rho)
    Lgu = box_g(u, gxx, gyy, gdet, dx, dy)
    return -Lgu/u

def check_curved_fisher(seed=1, nx=128, ny=128):
    rng = np.random.default_rng(seed)
    X, Y, dx, dy = grid2d(nx, ny)
    a, gxx, gyy, gxy, gdet = conformal_metric(X, Y, amp=0.25)
    sqrtg = np.sqrt(gdet)
    rho = np.exp(-((X-0.35)**2 + (Y-0.61)**2)/0.018) + 0.35*np.exp(-((X-0.62)**2 + (Y-0.31)**2)/0.025) + 0.04
    eta = make_eta_safe(rho, rng.standard_normal(rho.shape), dx, dy, sqrtg=sqrtg, eps=3e-5)
    eps = 3e-5
    Fp = fisher_curved_F(rho + eps*eta, gxx, gyy, gdet, dx, dy)
    Fm = fisher_curved_F(rho - eps*eta, gxx, gyy, gdet, dx, dy)
    fd = (Fp - Fm)/(2*eps)
    var = fisher_curved_var_discrete(rho, gxx, gyy, gdet, dx, dy)
    inner = np.sum(var*eta*sqrtg)*dx*dy
    rel = abs(fd - inner)/max(1e-16, abs(fd))
    return fd, inner, rel

# ------------------------- 3) ξ R ρ linear response -------------------------
def curvature_scalar_conformal(gxx, dx, dy):
    # 2D identity for a^2 δ_ij: R = -2 a^{-2} Δ log a
    a = np.sqrt(gxx); loga = np.log(a)
    R = -2*(Lap(loga, dx, dy)/(a**2))
    return R

def check_xi_R_response(xi=1.0, nx=128, ny=128):
    X, Y, dx, dy = grid2d(nx, ny)
    a, gxx, gyy, gxy, gdet = conformal_metric(X, Y, amp=0.25)
    sqrtg = np.sqrt(gdet)
    R = curvature_scalar_conformal(gxx, dx, dy)
    rho = np.exp(-((X-0.45)**2 + (Y-0.56)**2)/0.02) + 0.02
    rng = np.random.default_rng(2)
    eta = make_eta_safe(rho, rng.standard_normal(rho.shape), dx, dy, sqrtg=sqrtg, eps=1e-5)
    eps = 1e-5
    def Gfun(r):
        return np.sum(xi*R*r*sqrtg)*dx*dy
    fd = (Gfun(rho + eps*eta) - Gfun(rho - eps*eta))/(2*eps)
    inner = np.sum((xi*R)*eta*sqrtg)*dx*dy
    rel = abs(fd - inner)/max(1e-16, abs(fd))
    return fd, inner, rel, float(R.min()), float(R.max())

# ------------------------- 4) Discrete KG residual harness -------------------------
@dataclass
class KGParams:
    nx: int = 512
    Lx: float = 20.0
    nmode: int = 5
    m: float = 1.1
    s_target: float = 0.2    # desired sin(ω dt/2)
    M: int = 5               # temporal periods

def choose_dt_nt(dx, k, m, s_target, M, nt_guess=64):
    # target discrete dispersion: sin(ω dt/2) = s_target, with ω^2 matching kd^2 + m^2
    kd2 = 4.0*(math.sin(0.5*k*dx)/dx)**2
    rhs = kd2 + m**2
    # initial dt from s_target
    dt0 = 2*s_target/math.sqrt(rhs)
    # pick nt around the round(Lt/dt0) that keeps s close to s_target
    # Lt = 2π M / ω where ω satisfies sin(ω dt/2) = s
    # but ω depends on dt: ω = 2/dt * arcsin( s ), so Lt = 2π M dt / (2 arcsin(s))
    # we therefore search nt in a small window to minimise |s - s_target|
    def s_from_nt(nt):
        dt = None
        # pick dt as Lt/nt with Lt defined self-consistently using s_target; iterate once
        dt = dt0
        # recompute ω, Lt, dt -> one fixed-point iteration
        w = 2.0*math.asin(min(0.999999, s_target))/dt
        Lt = 2*math.pi*M/w
        dt = Lt/nt
        # compute actual s for this dt
        s = 0.5*dt*math.sqrt(rhs)
        return min(0.999999, s), dt
    nt0 = max(32, int(round((2*math.pi*M)/(2*math.asin(min(0.999999, s_target))) * dt0**-1 * dt0)))
    candidates = range(max(32, nt0-10), nt0+11)
    best = None
    for nt in candidates:
        s, dt = s_from_nt(nt)
        err = abs(s - s_target)
        if best is None or err < best[0]:
            best = (err, nt, dt, s)
    _, nt, dt, s = best
    w = 2.0*math.asin(s)/dt
    return dt, nt, w

def kg_residual(params: KGParams):
    nx, Lx, nmode, m, s_target, M = params.nx, params.Lx, params.nmode, params.m, params.s_target, params.M
    x = np.linspace(-Lx/2, Lx/2, nx, endpoint=False); dx = x[1]-x[0]
    k = 2*math.pi*nmode/Lx
    dt, nt, w = choose_dt_nt(dx, k, m, s_target, M)
    # construct fields
    t = np.linspace(0, dt*nt, nt, endpoint=False)
    X, T = np.meshgrid(x, t, indexing='ij')
    psi = np.exp(1j*(k*X - w*T))
    d2x = (np.roll(psi, -1, axis=0) - 2*psi + np.roll(psi, 1, axis=0))/dx**2
    d2t = (np.roll(psi, -1, axis=1) - 2*psi + np.roll(psi, 1, axis=1))/dt**2
    R = (d2t - d2x + m**2 * psi)
    rms_R = float(np.sqrt(np.mean(np.abs(R)**2)))
    rms_psi = float(np.sqrt(np.mean(np.abs(psi)**2)))
    ratio_rms = rms_R/rms_psi
    ratio = R/psi
    mean_re = float(np.real(ratio).mean())
    mean_im = float(np.imag(ratio).mean())
    max_abs = float(np.abs(ratio).max())
    return {
        "dx": float(dx),
        "dt": float(dt),
        "omega": float(w),
        "k": float(k),
        "rms_residual": ratio_rms,
        "mean_real_R_over_psi": mean_re,
        "mean_imag_R_over_psi": mean_im,
        "max_abs_R_over_psi": max_abs,
        "nt": int(nt),
        "s_target": float(s_target)
    }

# ------------------------- Runner with assertions -------------------------
def main():
    print("== Guard‑tower diagnostics ==")
    # 1) Flat Fisher (tolerance extremely tight)
    fd, inner, rel = check_flat_fisher()
    print(f"[Flat Fisher]  directional-derivative match: rel.err = {rel:.2e}   (fd={fd:.6e}, inner={inner:.6e})")
    assert rel < 1e-6, "Flat Fisher variation mismatch exceeds tolerance"
    # 2) Curved Fisher
    fd_c, inner_c, rel_c = check_curved_fisher()
    print(f"[Curved Fisher] directional-derivative match: rel.err = {rel_c:.2e}  (fd={fd_c:.6e}, inner={inner_c:.6e})")
    assert rel_c < 1e-6, "Curved Fisher variation mismatch exceeds tolerance"
    # 3) ξR response
    fd_xi, inner_xi, rel_xi, Rmin, Rmax = check_xi_R_response()
    print(f"[ξR response]   directional-derivative match: rel.err = {rel_xi:.2e}  (R range = [{Rmin:.3f},{Rmax:.3f}])")
    assert rel_xi < 1e-7, "ξR linear-response mismatch exceeds tolerance"
    # 4) KG residual (order-of-accuracy sized residual; assert a conservative bound)
    kg_info = kg_residual(KGParams())
    print("[KG residual]    discrete-symbol matched plane wave:")
    for k,v in kg_info.items():
        print(f"   {k}: {v}")
    assert kg_info["rms_residual"] < 5e-4, "KG RMS residual larger than expected for matched stencil"
    print("Done.")

if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("ASSERTION FAILED:", e, file=sys.stderr)
        sys.exit(2)
