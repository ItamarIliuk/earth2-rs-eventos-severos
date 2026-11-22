"""Figuras extras: meteograma de Tupanciretã, caso de maio/2024 e quadros da animação.

uso:
  python plot_extra.py meteo  fcn3_rs_2026-11-1*_gfs.nc
  python plot_extra.py maio   fcn3_rs_2024-04-26T00_arco.nc
  python plot_extra.py anim   fcn3_rs_2026-11-17T00_gfs.nc
"""
import os
import sys

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.argv, MODE, FILES = sys.argv[:1], sys.argv[1], sys.argv[2:]
import plot_rs as P  # reaproveita estilo (precisa de argv mínimo)

BG, FG, GREEN, GREY, CMAP = P.BG, P.FG, P.GREEN, P.GREY, P.CMAP
os.makedirs("figs", exist_ok=True)
TUP = (-29.08, -53.84)


def valid(ds):
    return np.datetime64(ds.attrs["init"]) + ds.lead_h.values.astype("timedelta64[h]")


def near_max(ds, var, la, lo, r=0.5):
    sub = ds[var].sel(lat=slice(la + r, la - r), lon=slice(lo - r, lo + r))
    return sub.max(("lat", "lon"))


if MODE == "meteo":
    dss = sorted([xr.open_dataset(f) for f in FILES], key=lambda d: d.attrs["init"])
    cols = ["#3E5A2A", "#5E8A1E", GREEN, "#C8F05A"]
    fig, axs = plt.subplots(3, 1, figsize=(13, 9), sharex=True, facecolor=BG)
    specs = [("fg10m", 12.5, "Rajada > 45 km/h"), ("tp06", 10.0, "Chuva > 10 mm em 6 h"), ("kindex", 32.0, "Índice K > 32")]
    for ax, (var, thr, lab) in zip(axs, specs):
        ax.set_facecolor(BG)
        for c, ds in zip(cols, dss):
            pr = (near_max(ds, var, *TUP) > thr).mean("member") * 100
            ax.plot(valid(ds), pr, "-", lw=2.4, color=c, label=f"rodada de {P.fmt_init(np.datetime64(ds.attrs['init']))}")
        ax.axvspan(np.datetime64("2026-11-20T03"), np.datetime64("2026-11-20T12"), color="#E8433A", alpha=0.18, lw=0)
        ax.set_ylim(0, 100)
        ax.set_ylabel("%", color=FG)
        ax.set_title(lab, color=FG, loc="left", fontsize=13, fontfamily=P.TITLE)
        ax.tick_params(colors=FG)
        ax.grid(color="#1f2a22", lw=0.6)
    axs[0].legend(facecolor=BG, edgecolor=GREY, labelcolor=FG, fontsize=10, loc="upper left")
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))
    axs[-1].set_xlim(np.datetime64("2026-11-18T00"), np.datetime64("2026-11-21T00"))
    fig.suptitle("Tupanciretã · probabilidade no ensemble FourCastNet 3", color=GREEN, fontsize=19, fontfamily=P.TITLE)
    fig.text(0.01, 0.005, "faixa vermelha: madrugada/manhã de 20/11 (temporal) · máximo num raio de 0,5° · horários UTC",
             color=GREY, fontsize=9)
    fig.savefig("figs/meteograma_tupancireta.png", dpi=150, facecolor=BG, bbox_inches="tight")
    print("ok meteograma")

elif MODE == "maio":
    ds = xr.open_dataset(FILES[0])
    t0, t1 = np.datetime64("2024-04-29T00"), np.datetime64("2024-05-03T00")
    v = valid(ds)
    sel = (v > t0) & (v <= t1)
    acc = ds.tp06.isel(lead_h=np.where(sel)[0]).sum("lead_h")  # mm por membro
    # ERA5 observado (ARCO, tp horário em m)
    from earth2studio.data import ARCO
    times = np.arange(t0 + np.timedelta64(1, "h"), t1 + np.timedelta64(1, "h"), np.timedelta64(1, "h"))
    era = ARCO()(times, ["tp"])
    era = era.sel(lat=slice(ds.lat.max(), ds.lat.min()), lon=slice(ds.lon.min() + 360, ds.lon.max() + 360))
    obs = era.sum("time").squeeze() * 1000.0
    obs = obs.assign_coords(lon=obs.lon - 360)
    lead_d = (t0 - np.datetime64(ds.attrs["init"])) / np.timedelta64(1, "D")
    panels = [
        (obs, "ERA5 (o que aconteceu)", "acc"),
        (acc.mean("member"), f"Média do ensemble · previsão feita {lead_d:.0f} dias antes", "acc"),
        ((acc > 150).mean("member") * 100, "Probabilidade de > 150 mm", "prob"),
    ]
    from matplotlib.colors import LinearSegmentedColormap
    RAIN = LinearSegmentedColormap.from_list("rain", [(0, (0, 0, 0, 0)), (0.08, "#1d3b2a"), (0.3, "#2f7d5b"), (0.55, GREEN), (0.8, "#F2C230"), (1, "#E8433A")])
    fig, axs = plt.subplots(1, 3, figsize=(18, 6.4), subplot_kw={"projection": ccrs.PlateCarree()}, facecolor=BG)
    for ax, (f, title, kind) in zip(axs, panels):
        if kind == "acc":
            im1 = ax.pcolormesh(f.lon, f.lat, f, cmap=RAIN, vmin=0, vmax=400, transform=ccrs.PlateCarree(), shading="auto")
        else:
            im2 = ax.pcolormesh(f.lon, f.lat, f, cmap=CMAP, vmin=0, vmax=100, transform=ccrs.PlateCarree(), shading="auto")
        P.decorate(ax)
        ax.set_title(title, color=FG, fontsize=13, fontfamily=P.TITLE)
    c1 = fig.colorbar(im1, ax=axs[:2], orientation="horizontal", fraction=0.05, pad=0.04, aspect=40)
    c1.set_label("Chuva acumulada 29/04–02/05/2024 (mm)", color=FG); c1.ax.tick_params(colors=FG)
    c2 = fig.colorbar(im2, ax=axs[2], orientation="horizontal", fraction=0.05, pad=0.04, aspect=20)
    c2.set_label("% dos membros", color=FG); c2.ax.tick_params(colors=FG)
    fig.suptitle(f"Enchente de maio de 2024 · previsão iniciada em {P.fmt_init(np.datetime64(ds.attrs['init']))}",
                 color=GREEN, fontsize=20, fontfamily=P.TITLE, y=0.99)
    fig.text(0.01, 0.01, f"FourCastNet 3 + PrecipitationAFNO · {len(ds.member)} membros · condição inicial ERA5", color=GREY, fontsize=9)
    fig.savefig("figs/maio2024_chuva.png", dpi=150, facecolor=BG, bbox_inches="tight")
    np.savez("figs/maio2024_numeros.npz", obs_max=float(obs.sel(lat=slice(-27, -33), lon=slice(-57, -49)).max()),
             ens_mean_max=float(acc.mean("member").sel(lat=slice(-27, -33), lon=slice(-57, -49)).max()))
    print("ok maio", float(obs.max()), float(acc.mean('member').max()))

elif MODE == "anim":
    ds = xr.open_dataset(FILES[0])
    os.makedirs("figs/anim", exist_ok=True)
    v = valid(ds)
    mean = ds.mean("member")
    pg = (ds.fg10m > 12.5).mean("member") * 100
    from matplotlib.colors import LinearSegmentedColormap
    WET = LinearSegmentedColormap.from_list("wet", ["#0B100D", "#16301f", "#2f7d5b", "#76B900", "#E9F7B0"])
    st = 6
    for k in range(len(ds.lead_h)):
        fig = plt.figure(figsize=(12.8, 7.2), facecolor=BG)
        ax = fig.add_axes([0.02, 0.06, 0.70, 0.86], projection=ccrs.PlateCarree())
        ax.set_extent([-75, -40, -42, -12], ccrs.PlateCarree())
        ax.pcolormesh(ds.lon, ds.lat, mean.tcwv.isel(lead_h=k), cmap=WET, vmin=10, vmax=65, transform=ccrs.PlateCarree(), shading="auto")
        ax.quiver(ds.lon[::st], ds.lat[::st], mean.u850.isel(lead_h=k)[::st, ::st], mean.v850.isel(lead_h=k)[::st, ::st],
                  color=FG, alpha=0.7, scale=500, width=0.0018, transform=ccrs.PlateCarree())
        cs = ax.contour(ds.lon, ds.lat, pg.isel(lead_h=k), levels=[30, 60, 90], colors=["#F2C230", "#F08A30", "#E8433A"],
                        linewidths=1.6, transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.STATES.with_scale("50m"), edgecolor=GREY, linewidth=0.4)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor=FG, linewidth=0.7)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor=FG, linewidth=0.7)
        ax.plot(TUP[1], TUP[0], "o", ms=6, color="#E8433A", mec=FG, transform=ccrs.PlateCarree())
        ax.text(TUP[1] + 0.4, TUP[0] - 0.2, "Tupanciretã", color=FG, fontsize=11, transform=ccrs.PlateCarree())
        s = str(v[k])
        fig.text(0.74, 0.86, "FourCastNet 3", color=GREEN, fontsize=24, fontfamily=P.TITLE)
        fig.text(0.74, 0.81, f"rodada de {P.fmt_init(np.datetime64(ds.attrs['init']))}", color=FG, fontsize=13)
        fig.text(0.74, 0.70, f"{s[8:10]}/{s[5:7]}  {s[11:13]}h UTC", color=FG, fontsize=30, fontfamily=P.TITLE)
        fig.text(0.74, 0.65, f"+{int(ds.lead_h[k])} h de previsão", color=GREY, fontsize=13)
        fig.text(0.74, 0.50, "fundo: vapor d'água na coluna\n(média de %d membros)" % len(ds.member), color=FG, fontsize=11)
        fig.text(0.74, 0.42, "setas: vento em 850 hPa\n(jato de baixos níveis)", color=FG, fontsize=11)
        fig.text(0.74, 0.32, "contornos: probabilidade de\nrajada > 45 km/h (30/60/90 %)", color="#F2C230", fontsize=11)
        fig.savefig(f"figs/anim/q{k:03d}.png", dpi=150, facecolor=BG)
        plt.close(fig)
    print("ok anim", len(ds.lead_h), "quadros")

elif MODE == "animk":
    # Versão para o deck: recorte no Sul e contornos de probabilidade de Índice K > 32.
    # Os trópicos têm K alto o tempo todo, então os contornos só aparecem ao sul de 24°S.
    ds = xr.open_dataset(FILES[0])
    out = "figs/anim_k"
    os.makedirs(out, exist_ok=True)
    v = valid(ds)
    mean = ds.mean("member")
    pk = (ds.kindex > 32).mean("member") * 100
    pk = pk.where(ds.lat < -24)
    from matplotlib.colors import LinearSegmentedColormap
    WET = LinearSegmentedColormap.from_list("wet", ["#0B100D", "#16301f", "#2f7d5b", "#76B900", "#E9F7B0"])
    ev0, ev1 = np.datetime64("2026-11-20T03"), np.datetime64("2026-11-20T12")
    st = 3
    for k in range(len(ds.lead_h)):
        fig = plt.figure(figsize=(12.8, 7.2), facecolor=BG)
        ax = fig.add_axes([0.02, 0.05, 0.70, 0.90], projection=ccrs.PlateCarree())
        ax.set_extent([-62, -45.5, -36, -24], ccrs.PlateCarree())
        ax.pcolormesh(ds.lon, ds.lat, mean.tcwv.isel(lead_h=k), cmap=WET, vmin=10, vmax=65, transform=ccrs.PlateCarree(), shading="auto")
        ax.quiver(ds.lon[::st], ds.lat[::st], mean.u850.isel(lead_h=k)[::st, ::st], mean.v850.isel(lead_h=k)[::st, ::st],
                  color=FG, alpha=0.65, scale=420, width=0.0020, transform=ccrs.PlateCarree())
        ax.contourf(ds.lon, ds.lat, pk.isel(lead_h=k), levels=[60, 90, 101], colors=["#F08A30", "#E8433A"], alpha=0.35,
                    transform=ccrs.PlateCarree())
        ax.contour(ds.lon, ds.lat, pk.isel(lead_h=k), levels=[60, 90], colors=["#F08A30", "#E8433A"],
                   linewidths=1.8, transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.STATES.with_scale("50m"), edgecolor=GREY, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor=FG, linewidth=0.8)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor=FG, linewidth=0.8)
        for nome, (la, lo) in {"Tupanciretã": TUP, "Passo Fundo": (-28.26, -52.41), "Porto Alegre": (-30.03, -51.23)}.items():
            ax.plot(lo, la, "o", ms=6, color=FG if nome != "Tupanciretã" else "#E8433A", mec=FG, transform=ccrs.PlateCarree())
            ax.text(lo + 0.3, la + 0.15, nome, color=FG, fontsize=11, transform=ccrs.PlateCarree())
        s = str(v[k])
        fig.text(0.74, 0.86, "FourCastNet 3", color=GREEN, fontsize=24, fontfamily=P.TITLE)
        fig.text(0.74, 0.81, f"previsão feita em {P.fmt_init(np.datetime64(ds.attrs['init']))}", color=FG, fontsize=13)
        fig.text(0.74, 0.70, f"{s[8:10]}/{s[5:7]}  {s[11:13]}h UTC", color=FG, fontsize=30, fontfamily=P.TITLE)
        fig.text(0.74, 0.65, f"+{int(ds.lead_h[k])} h de previsão", color=GREY, fontsize=13)
        if ev0 <= v[k] <= ev1:
            fig.text(0.74, 0.595, "madrugada do temporal", color="#E8433A", fontsize=15, fontfamily=P.TITLE)
        fig.text(0.74, 0.47, "fundo: vapor d'água na coluna\n(média de %d membros)" % len(ds.member), color=FG, fontsize=11)
        fig.text(0.74, 0.39, "setas: vento em 850 hPa\n(jato de baixos níveis)", color=FG, fontsize=11)
        fig.text(0.74, 0.29, "manchas: chance de atmosfera\ninstável, Índice K > 32\nlaranja > 60 %  ·  vermelho > 90 %", color="#F08A30", fontsize=11)
        fig.savefig(f"{out}/q{k:03d}.png", dpi=150, facecolor=BG)
        plt.close(fig)
    print("ok animk", len(ds.lead_h), "quadros")
