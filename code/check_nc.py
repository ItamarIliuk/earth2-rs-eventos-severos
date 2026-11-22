import sys
import numpy as np
import xarray as xr

ds = xr.open_dataset(sys.argv[1])
print(ds.attrs.get("init"), dict(ds.sizes))
rs = ds.sel(lat=slice(-27, -34), lon=slice(-58, -49))
for v in ds.data_vars:
    a = ds[v].values
    r = rs[v].values
    print(f"{v:14s} nan%={np.isnan(a).mean()*100:5.1f}  global-box min={np.nanmin(a):8.2f} max={np.nanmax(a):8.2f}  RS mean={np.nanmean(r):8.2f} max={np.nanmax(r):8.2f}")
if len(ds.member) > 1:
    d = float(np.nanmean(np.abs(ds.tcwv.isel(member=0, lead_h=-1) - ds.tcwv.isel(member=1, lead_h=-1))))
    print("diferença média tcwv entre membros no último passo:", round(d, 3))
