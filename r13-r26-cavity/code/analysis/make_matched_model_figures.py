#!/usr/bin/env python3
"""Reproduce the matched DSMC/R13/R26 comparison figures.

The script uses one common DSMC cell-centre grid, one declared seven-cell
uniform filter, and a fixed five-percent activity threshold.  It writes both
vector PDF and 600 dpi PNG outputs together with machine-readable metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import uniform_filter

ORDER = ("rho", "u", "v", "T", "qx", "qy", "sxx", "sxy", "syy",
         "Rxx", "Rxy", "Ryy", "mxxx", "mxxy", "mxyy", "myyy", "Delta")


def interp(state, sx, sy, tx, ty):
    yy, xx = np.meshgrid(ty, tx, indexing="ij")
    pts = np.column_stack((yy.ravel(), xx.ravel()))
    out = np.empty((len(ty), len(tx), state.shape[-1]))
    for k in range(state.shape[-1]):
        out[..., k] = RegularGridInterpolator((sy, sx), state[..., k],
                                               bounds_error=True)(pts).reshape(len(ty), len(tx))
    return out


def load_dsmc(root):
    meta = json.loads((root / "case_metadata.json").read_text())
    a = np.loadtxt(root / "grid.final.00200000", skiprows=9)
    a = a[np.lexsort((a[:, 1], a[:, 2]))]
    nx = int(meta["nx"])
    vals = a[:, 3:].reshape(nx, nx, -1)
    x = np.unique(a[:, 1]) / meta["length_m"]
    y = np.unique(a[:, 2]) / meta["length_m"]
    c0 = math.sqrt(1.380649e-23 * meta["wall_temperature_K"] / meta["argon_mass_kg"])
    rho0 = meta["number_density_m-3"] * meta["argon_mass_kg"]
    q0 = rho0 * c0**3
    state = np.zeros((nx, nx, 6))
    state[..., 0] = vals[..., 0] / meta["number_density_m-3"]
    state[..., 1:3] = vals[..., 1:3] / c0
    state[..., 3] = vals[..., 4] / meta["wall_temperature_K"]
    state[..., 4:6] = vals[..., 5:7] / q0
    return state, x, y, meta, c0


def load_r26(root, c0):
    with np.load(root / "last_accepted_state.npz", allow_pickle=False) as z:
        state, x, y = z["state"].copy(), z["x"].copy(), z["y"].copy()
    ratio = math.sqrt(208.0 * 300.0) / c0
    state[..., 1:3] *= ratio
    state[..., 4:6] *= ratio**3
    return state, x, y


def load_r13(run_root, source_root):
    report = json.loads((run_root / "report.json").read_text())
    sys.path.insert(0, str(source_root))
    from rana_original_reference_solver import RanaOriginalConfig, eliminated_wall_state
    allowed = RanaOriginalConfig.__dataclass_fields__
    cfg = RanaOriginalConfig(**{k: v for k, v in report["configuration"].items() if k in allowed})
    interior = np.load(run_root / "state.npy", allow_pickle=False)
    walls = {s: eliminated_wall_state(interior, cfg, side=s)
             for s in ("left", "right", "bottom", "top")}
    ny, nx, nf = interior.shape
    full = np.zeros((ny + 2, nx + 2, nf)); full[1:-1, 1:-1] = interior
    full[1:-1, 0], full[1:-1, -1] = walls["left"], walls["right"]
    full[0, 1:-1], full[-1, 1:-1] = walls["bottom"], walls["top"]
    def end(a, lo): return 2*a[0]-a[1] if lo else 2*a[-1]-a[-2]
    full[0, 0] = .5*(end(walls["bottom"], True)+end(walls["left"], True))
    full[0, -1] = .5*(end(walls["bottom"], False)+end(walls["right"], True))
    full[-1, 0] = .5*(end(walls["top"], True)+end(walls["left"], False))
    full[-1, -1] = .5*(end(walls["top"], False)+end(walls["right"], False))
    return full, np.linspace(0, 1, nx+2), np.linspace(0, 1, ny+2), report


def vec_metrics(pred, ref, mask):
    p, r = pred[mask], ref[mask]
    mp, mr = np.linalg.norm(p, axis=1), np.linalg.norm(r, axis=1)
    ok = (mp > 1e-14) & (mr > 1e-14)
    ang = np.degrees(np.arccos(np.clip(np.sum(p[ok]*r[ok], axis=1)/(mp[ok]*mr[ok]), -1, 1)))
    w = mp[ok]*mr[ok]
    return {"E": float(np.linalg.norm(p-r)/np.linalg.norm(r)),
            "C": float(np.sum(p*r)/(np.linalg.norm(p)*np.linalg.norm(r))),
            "G": float(np.linalg.norm(p)/np.linalg.norm(r)),
            "angle_w_deg": float(np.average(ang, weights=w))}


def af(state, x, y, eligible, smooth=7, threshold=.05):
    T, qx, qy = [uniform_filter(state[..., k], smooth, mode="nearest") for k in (3, 4, 5)]
    dTdy, dTdx = np.gradient(T, y, x, edge_order=2)
    qm, gm = np.hypot(qx, qy), np.hypot(dTdx, dTdy)
    active = eligible & (qm > threshold*qm[eligible].max()) & (gm > threshold*gm[eligible].max())
    I = np.full(T.shape, np.nan)
    I[active] = (qx[active]*dTdx[active]+qy[active]*dTdy[active])/(qm[active]*gm[active])
    return T, qx, qy, active, active & (I > 0), I


def panel_label(ax, label):
    ax.text(.02, .98, label, transform=ax.transAxes, va="top", ha="left",
            fontsize=15, fontweight="bold", color="black",
            bbox=dict(facecolor="white", alpha=.82, edgecolor="none", pad=2))


def save(fig, out, stem):
    fig.savefig(out/f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out/f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    d, x, y, meta, c0 = load_dsmc(a.data/"dsmc_k05"/"run")
    r13, x13, y13, rep13 = load_r13(a.data/"r13_k05", a.data/"r13_source")
    r26, x26, y26 = load_r26(a.data/"r26_k05", c0)
    r13 = interp(r13, x13, y13, x, y); r26 = interp(r26, x26, y26, x, y)
    methods = [("DSMC", d), ("R13", r13), ("R26", r26)]
    xx, yy = np.meshgrid(x, y)
    eligible = ~(((xx < .05) | (xx > .95)) & (yy > .95))
    sm = {n: np.stack([uniform_filter(s[..., k], 7, mode="nearest") for k in range(6)], -1)
          for n, s in methods}
    fields = [(0, r"$\rho/\rho_0$", "viridis"), (3, r"$T/T_w$", "inferno"),
              (None, r"$|\mathbf{u}|/c_0$", "viridis"), (None, r"$|\mathbf{q}|/(\rho_0c_0^3)$", "cividis")]
    values = []
    for _, s in methods:
        values.append([s[..., 0], s[..., 3], np.hypot(s[..., 1], s[..., 2]), np.hypot(s[..., 4], s[..., 5])])
    fig, axs = plt.subplots(3, 4, figsize=(16.8, 12.0), constrained_layout=True)
    for j, (_, lab, cmap) in enumerate(fields):
        lo = min(v[j][eligible].min() for v in values); hi = max(v[j][eligible].max() for v in values)
        for i, (name, s) in enumerate(methods):
            im = axs[i,j].pcolormesh(x, y, values[i][j], shading="auto", cmap=cmap, vmin=lo, vmax=hi)
            if j == 2: axs[i,j].quiver(xx[::10,::10], yy[::10,::10], s[::10,::10,1], s[::10,::10,2], color="white", scale=2.6, width=.004)
            if j == 3: axs[i,j].quiver(xx[::10,::10], yy[::10,::10], s[::10,::10,4], s[::10,::10,5], color="white", scale=.13, width=.004)
            axs[i,j].set_aspect("equal"); axs[i,j].tick_params(labelsize=12)
            panel_label(axs[i,j], f"({chr(97+i*4+j)}) {name}")
            if i == 2: axs[i,j].set_xlabel(r"$x/L$", fontsize=14)
            if j == 0: axs[i,j].set_ylabel(r"$y/L$", fontsize=14)
        cb=fig.colorbar(im, ax=axs[:,j], shrink=.88, pad=.012); cb.set_label(lab, fontsize=14); cb.ax.tick_params(labelsize=11)
    save(fig, a.out, "fig_kn005_primary_fields")

    afd = af(d,x,y,eligible); afr13=af(r13,x,y,eligible); afr26=af(r26,x,y,eligible)
    afs={"DSMC":afd,"R13":afr13,"R26":afr26}
    fig, axs = plt.subplots(2,3,figsize=(16.8,9.1),constrained_layout=True)
    qmax=max(np.nanpercentile(np.hypot(v[1],v[2])[eligible],99.5) for v in afs.values())
    for j,(name,s) in enumerate(methods):
        T,qx,qy,active,mask,I=afs[name]
        im=axs[0,j].pcolormesh(x,y,np.hypot(qx,qy),shading="auto",cmap="cividis",vmin=0,vmax=qmax)
        axs[0,j].quiver(xx[::9,::9],yy[::9,::9],qx[::9,::9],qy[::9,::9],color="white",scale=.12,width=.004)
        im2=axs[1,j].pcolormesh(x,y,I,shading="auto",cmap="coolwarm",norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1))
        axs[1,j].contour(x,y,mask,levels=[.5],colors="#111111",linewidths=1.8)
        for i in range(2):
            axs[i,j].set_aspect("equal"); axs[i,j].tick_params(labelsize=12)
            panel_label(axs[i,j],f"({chr(97+i*3+j)}) {name}")
        axs[1,j].set_xlabel(r"$x/L$",fontsize=14)
    axs[0,0].set_ylabel(r"$y/L$",fontsize=14); axs[1,0].set_ylabel(r"$y/L$",fontsize=14)
    cb=fig.colorbar(im,ax=axs[0,:],shrink=.92,pad=.012); cb.set_label(r"$|\mathbf{q}|/(\rho_0c_0^3)$",fontsize=14)
    cb=fig.colorbar(im2,ax=axs[1,:],shrink=.92,pad=.012); cb.set_label(r"$I_{AF}=\widehat{\mathbf{q}}\cdot\widehat{\nabla T}$",fontsize=14)
    save(fig,a.out,"fig_kn005_antifourier")

    fig,axs=plt.subplots(2,2,figsize=(15.8,9.4),constrained_layout=True)
    profiles=[(1,"u",x,lambda s:s[len(y)//2,:,1]),(2,"v",y,lambda s:s[:,len(x)//2,2]),
              (3,"T",x,lambda s:s[len(y)//2,:,3]),(4,"q_x",y,lambda s:s[:,len(x)//2,4])]
    colors={"DSMC":"black","R13":"#d55e00","R26":"#0072b2"}
    for p,(ax,(k,lab,coord,fn)) in enumerate(zip(axs.ravel(),profiles)):
        for name,s in methods:
            ax.plot(coord,fn(sm[name]),lw=2.6 if name!="DSMC" else 2.2,label=name,color=colors[name],alpha=.92)
        ax.grid(alpha=.25); ax.tick_params(labelsize=13); ax.set_xlabel(r"$x/L$" if len(coord)==len(x) else r"$y/L$",fontsize=15)
        ax.set_ylabel(rf"${lab}$",fontsize=15); panel_label(ax,f"({chr(97+p)})")
    axs[0,0].legend(frameon=False,fontsize=14,ncol=3)
    save(fig,a.out,"fig_kn005_centerlines")

    metrics={"contract":{"kn_gu":meta["kn_gu"],"molecular_model":meta["molecular_model"],"nx":meta["nx"],"ppc":meta["particles_per_cell"]},
             "r13_status":{"publication_grade":rep13.get("publication_grade"),"external_validation_status":rep13.get("external_validation_status"),"fixed_point_residual":rep13["solver"]["fixed_point_relative_residual"]},
             "vector":{},"anti_fourier":{}}
    for name,s,aft in (("R13",r13,afr13),("R26",r26,afr26)):
        metrics["vector"][name]={"velocity":vec_metrics(s[...,1:3],d[...,1:3],eligible),"heat_flux":vec_metrics(np.stack(aft[1:3],-1),np.stack(afd[1:3],-1),afd[3])}
        inter=(aft[4]&afd[4]&afd[3]).sum(); union=((aft[4]|afd[4])&afd[3]).sum()
        metrics["anti_fourier"][name]={"Jaccard":float(inter/max(union,1)),"Dice":float(2*inter/max((aft[4]&afd[3]).sum()+afd[4].sum(),1))}
    (a.out/"matched_kn005_metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")


if __name__ == "__main__": main()
