"""Mapas de probabilidade do ensemble FCN3 sobre o RS, no visual da keynote.

uso: python plot_rs.py --event-start 2026-11-19T18 --event-end 2026-11-20T12 fcn3_rs_2026-11-16T00_gfs.nc ...
"""
import argparse
import os

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

BG, FG, GREEN, GREY = "#0B100D", "#E8ECE6", "#76B900", "#8A938C"
for fdir in [os.path.expanduser("~/fonts"), "C:/Windows/Fonts", os.path.expanduser("~/AppData/Local/Microsoft/Windows/Fonts")]:
    if os.path.isdir(fdir):
        for f in os.listdir(fdir):
            if f.lower().startswith(("inter", "spacegrotesk", "space")) and f.lower().endswith((".ttf", ".otf")):
                try:
                    font_manager.fontManager.addfont(os.path.join(fdir, f))
                except Exception:
                    pass
avail = {f.name for f in font_manager.fontManager.ttflist}
BODY = "Inter" if "Inter" in avail else "DejaVu Sans"
TITLE = "Space Grotesk" if "Space Grotesk" in avail else BODY
plt.rcParams.update({"font.family": BODY, "text.color": FG, "axes.edgecolor": GREY})

CMAP = LinearSegmentedColormap.from_list("prob", [(0, (0, 0, 0, 0)), (0.1, "#2b3a1a"), (0.4, "#76B900"), (0.7, "#F2C230"), (1.0, "#E8433A")])
CITIES = {"Tupanciretã": (-29.08, -53.84), "Passo Fundo": (-28.26, -52.41), "Porto Alegre": (-30.03, -51.23)}
EXT = [-58.5, -48.5, -34.5, -26.5]

EVENT = ['', '']
PRODUCTS = [
    ("fg10m", 12.5, "Rajada > 45 km/h"),
    ("tp06", 20.0, "Chuva > 20 mm em 6 h"),
    ("kindex", 32.0, "Índice K > 32 (instabilidade)"),
]


def window(ds):
    t0 = np.datetime64(ds.attrs["init"])
    valid = t0 + ds.lead_h.values.astype("timedelta64[h]")
    sel = (valid >= np.datetime64(EVENT[0])) & (valid <= np.datetime64(EVENT[1]))
    return ds.isel(lead_h=np.where(sel)[0]), t0


def prob(ds, var, thr):
    return (ds[var].max("lead_h") > thr).mean("member")


def decorate(ax):
    ax.set_extent(EXT, ccrs.PlateCarree())
    ax.add_feature(cfeature.STATES.with_scale("10m"), edgecolor=GREY, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), edgecolor=FG, linewidth=0.8)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor=FG, linewidth=0.8)
    for n, (la, lo) in CITIES.items():
        ax.plot(lo, la, "o", ms=5, color=FG, mec=BG, transform=ccrs.PlateCarree())
        ax.text(lo + 0.15, la + 0.12, n, fontsize=10, color=FG, transform=ccrs.PlateCarree())
    ax.set_facecolor(BG)


def fmt_init(t0):
    s = str(t0)
    return f"{s[8:10]}/{s[5:7]} {s[11:13]}h UTC"




def main():
    global EVENT
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--event-start", required=True)
    ap.add_argument("--event-end", required=True)
    ap.add_argument("--outdir", default="figs")
    ap.add_argument("--title", default="Temporal de 20/11/2026")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    EVENT[:] = [a.event_start, a.event_end]
    dss = [xr.open_dataset(f) for f in a.files]
    dss = sorted(dss, key=lambda d: d.attrs["init"])

    # Figura 1: uma linha por inicialização, colunas = produtos
    for var, thr, label in PRODUCTS:
        fig, axs = plt.subplots(1, len(dss), figsize=(6 * len(dss), 6.2), subplot_kw={"projection": ccrs.PlateCarree()}, facecolor=BG)
        axs = np.atleast_1d(axs)
        for ax, ds in zip(axs, dss):
            w, t0 = window(ds)
            if var not in w or np.isnan(w[var].values).all():
                continue
            pr = prob(w, var, thr)
            im = ax.pcolormesh(w.lon, w.lat, pr * 100, cmap=CMAP, vmin=0, vmax=100, transform=ccrs.PlateCarree(), shading="auto")
            decorate(ax)
            lead_d = (np.datetime64(a.event_start) - t0) / np.timedelta64(1, "D")
            ax.set_title(f"rodada de {fmt_init(t0)}  ·  {lead_d:.1f} dias antes", color=FG, fontsize=13, fontfamily=TITLE)
        cb = fig.colorbar(im, ax=axs, orientation="horizontal", fraction=0.05, pad=0.04, aspect=50)
        cb.set_label(f"Probabilidade (% de {len(dss[0].member)} membros)  ·  {label}", color=FG, fontsize=12)
        cb.ax.tick_params(colors=FG)
        fig.suptitle(f"{a.title}  ·  {label}", color=GREEN, fontsize=20, fontfamily=TITLE, y=0.98)
        fig.text(0.01, 0.01, "FourCastNet 3 (NVIDIA Earth-2) · ensemble · janela "
                 f"{a.event_start.replace('T', ' ')}–{a.event_end.replace('T', ' ')} UTC", color=GREY, fontsize=9)
        fig.savefig(os.path.join(a.outdir, f"prob_{var}.png"), dpi=150, facecolor=BG, bbox_inches="tight")
        plt.close(fig)
        print("ok", var)

    # Figura 2: painel com os 3 produtos da rodada mais antiga que ainda mostra sinal (a primeira)
    ds = dss[0]
    w, t0 = window(ds)
    fig, axs = plt.subplots(1, 3, figsize=(18, 6.2), subplot_kw={"projection": ccrs.PlateCarree()}, facecolor=BG)
    for ax, (var, thr, label) in zip(axs, PRODUCTS):
        if var in w and not np.isnan(w[var].values).all():
            im = ax.pcolormesh(w.lon, w.lat, prob(w, var, thr) * 100, cmap=CMAP, vmin=0, vmax=100, transform=ccrs.PlateCarree(), shading="auto")
        decorate(ax)
        ax.set_title(label, color=FG, fontsize=14, fontfamily=TITLE)
    cb = fig.colorbar(im, ax=axs, orientation="horizontal", fraction=0.05, pad=0.04, aspect=50)
    cb.set_label("Probabilidade (%)", color=FG)
    cb.ax.tick_params(colors=FG)
    fig.suptitle(f"{a.title}  ·  previsão feita em {fmt_init(t0)}", color=GREEN, fontsize=20, fontfamily=TITLE, y=0.98)
    fig.savefig(os.path.join(a.outdir, "painel_3produtos.png"), dpi=150, facecolor=BG, bbox_inches="tight")
    print("ok painel")


if __name__ == '__main__':
    main()
