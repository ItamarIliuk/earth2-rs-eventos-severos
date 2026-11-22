"""Ensemble FCN3 para eventos severos no RS (Summit Passo Fundo, 27/11/2026).

Para cada membro: FCN3 (global 0,25°, passo 6 h, estocástico) + WindgustAFNO (rajada) +
PrecipitationAFNO v1 (chuva em 6 h, com sp/r500/r850 derivados) + índice K (instabilidade).
Salva só a caixa da América do Sul num NetCDF.

uso: uv run python run_fcn3_rs.py --init 2026-11-17T00 --end 2026-11-21T00 --members 16 --source gfs
"""
import argparse
import time as _time
from collections import OrderedDict

import numpy as np
import torch
import xarray as xr

from earth2studio.data import GFS, ARCO
from earth2studio.data.utils import fetch_data
from earth2studio.models.px import FCN3
from earth2studio.models.dx import WindgustAFNO, PrecipitationAFNO
from earth2studio.utils.coords import map_coords

LAT_N, LAT_S, LON_W, LON_E = 5.0, -45.0, 280.0, 330.0  # América do Sul (lon 0-360)
G, RD = 9.80665, 287.05

p = argparse.ArgumentParser()
p.add_argument("--init", required=True)
p.add_argument("--end", required=True)
p.add_argument("--members", type=int, default=16)
p.add_argument("--source", choices=["gfs", "arco"], default="gfs")
p.add_argument("--out", default=None)
p.add_argument("--no-precip", action="store_true")
a = p.parse_args()

dev = torch.device("cuda")
t0 = np.datetime64(a.init)
nsteps = int((np.datetime64(a.end) - t0) / np.timedelta64(6, "h"))
src = GFS() if a.source == "gfs" else ARCO()
print(f"init {t0}  passos {nsteps}  membros {a.members}  fonte {a.source}", flush=True)

model = FCN3.load_model(FCN3.load_default_package()).to(dev)
wg = WindgustAFNO.load_model(WindgustAFNO.load_default_package()).to(dev)
pa = None if a.no_precip else PrecipitationAFNO.load_model(PrecipitationAFNO.load_default_package()).to(dev)

ic = model.input_coords()
VARS = list(ic["variable"])
x0, c0 = fetch_data(src, time=[t0], variable=VARS, lead_time=ic["lead_time"], device=dev)
x0, c0 = map_coords(x0, c0, ic)
assert not torch.isnan(x0).any(), "condição inicial com NaN"

# Altura do terreno a partir de sp/msl/t2m da condição inicial (para derivar sp nos passos)
xs, cs = fetch_data(src, time=[t0], variable=["sp", "msl", "t2m"], lead_time=ic["lead_time"], device=dev)
sp0, msl0, t2m0 = xs[0, 0, 0], xs[0, 0, 1], xs[0, 0, 2]
h = RD * t2m0 / G * torch.log(msl0 / sp0)
for _ in range(2):
    h = RD * (t2m0 + 0.0065 * h / 2) / G * torch.log(msl0 / sp0)
lat_full = np.asarray(c0["lat"])
lon_full = np.asarray(c0["lon"])
if len(cs["lat"]) != len(lat_full):
    raise SystemExit("grade de sp diferente da do FCN3")

ilat = np.where((lat_full <= LAT_N) & (lat_full >= LAT_S))[0]
ilon = np.where((lon_full >= LON_W) & (lon_full <= LON_E))[0]
box = lambda f: f[..., ilat[0]:ilat[-1] + 1, ilon[0]:ilon[-1] + 1]


def es_hpa(tk):
    tc = tk - 273.15
    return 6.112 * torch.exp(17.67 * tc / (tc + 243.5))


def e_hpa(q, p_hpa):
    return q * p_hpa / (0.622 + 0.378 * q)


def td_c(q, p_hpa):
    lne = torch.log(torch.clamp(e_hpa(q, p_hpa), min=1e-3) / 6.112)
    return 243.5 * lne / (17.67 - lne)


def fields(y, yc):
    """y: (var, lat, lon) global; devolve dicionário de campos derivados."""
    vi = {v: i for i, v in enumerate(yc["variable"])}
    g = lambda n: y[vi[n]]
    out = {
        "tcwv": g("tcwv"), "msl": g("msl") / 100.0, "t2m": g("t2m") - 273.15,
        "u850": g("u850"), "v850": g("v850"),
        "ws10": torch.hypot(g("u10m"), g("v10m")),
    }
    out["ws850"] = torch.hypot(g("u850"), g("v850"))
    k = (g("t850") - g("t500")) + td_c(g("q850"), 850.0) - ((g("t700") - 273.15) - td_c(g("q700"), 700.0))
    out["kindex"] = k
    out["shear_850_500"] = torch.hypot(g("u500") - g("u850"), g("v500") - g("v850"))
    return out, g


def diag_wind(y, yc):
    x = y[None, None]
    c = OrderedDict(time=yc["time"], lead_time=yc["lead_time"], variable=yc["variable"], lat=yc["lat"], lon=yc["lon"])
    xw, cw = map_coords(x, c, wg.input_coords())
    o, oc = wg(xw, cw)
    return o[0, 0, 0]  # (720, 1440)


def diag_precip(g, yc):
    T = lambda f: f[:720]
    msl, t2m = g("msl"), g("t2m")
    sp = msl * torch.exp(-G * h / (RD * (t2m + 0.0065 * h / 2)))
    r500 = torch.clamp(100 * e_hpa(g("q500"), 500.0) / es_hpa(g("t500")), 0, 100)
    r850 = torch.clamp(100 * e_hpa(g("q850"), 850.0) / es_hpa(g("t850")), 0, 100)
    der = {"sp": sp, "r500": r500, "r850": r850}
    pic = pa.input_coords()
    vi = {v: i for i, v in enumerate(yc["variable"])}
    stack = [T(der[v]) if v in der else T(g(v)) for v in pic["variable"]]
    x = torch.stack(stack)[None, None]
    c = OrderedDict(time=yc["time"], lead_time=yc["lead_time"], variable=np.array(pic["variable"]),
                    lat=pic["lat"], lon=pic["lon"])
    o, oc = pa(x, c)
    return o[0, 0, 0] * 1000.0  # m -> mm


names = ["tcwv", "msl", "t2m", "u850", "v850", "ws10", "ws850", "kindex", "shear_850_500", "fg10m", "tp06"]
nlead = nsteps + 1
store = {n: np.full((a.members, nlead, len(ilat), len(ilon)), np.nan, np.float32) for n in names}
leads = np.arange(nlead) * 6

tic = _time.time()
for m in range(a.members):
    model.set_rng(seed=1000 + m)
    it = model.create_iterator(x0.clone(), c0.copy())
    k = 0
    for y, yc in it:
        lt = int(np.asarray(yc["lead_time"])[0] / np.timedelta64(1, "h"))
        k = lt // 6
        if k > nsteps:
            break
        yy = y[0, 0]
        with torch.no_grad():
            f, g = fields(yy, yc)
            f["fg10m"] = torch.nn.functional.pad(diag_wind(yy, yc), (0, 0, 0, 1), value=float("nan"))
            if pa is not None and k > 0:
                f["tp06"] = torch.nn.functional.pad(diag_precip(g, yc), (0, 0, 0, 1), value=float("nan"))
        for n, v in f.items():
            store[n][m, k] = box(v).float().cpu().numpy()
        if k == nsteps:
            break
    print(f"membro {m + 1}/{a.members}  ({_time.time() - tic:.0f} s)", flush=True)

ds = xr.Dataset(
    {n: (("member", "lead_h", "lat", "lon"), store[n]) for n in names},
    coords={"member": np.arange(a.members), "lead_h": leads,
            "lat": lat_full[ilat], "lon": lon_full[ilon] - 360.0},
    attrs={"init": str(t0), "model": "FCN3 + WindgustAFNO + PrecipitationAFNO v1 (Earth2Studio 0.20.0)",
           "source": a.source,
           "units": "tcwv kg/m2; msl hPa; t2m C; vento m/s; kindex C; fg10m m/s (rajada máx 1 h); tp06 mm/6 h"},
)
out = a.out or f"fcn3_rs_{str(t0)[:13].replace(':', '')}_{a.source}.nc"
ds.to_netcdf(out)
print("salvo", out, f"total {_time.time() - tic:.0f} s", flush=True)
