import xarray as xr
import numpy as np
import json

surface = xr.open_dataset("data/surface_025_2022.nc")

so = np.nan_to_num(surface["so"].isel(time=0).values.squeeze(), nan=0.0)
uo = np.nan_to_num(surface["uo"].isel(time=0).values.squeeze(), nan=0.0)
vo = np.nan_to_num(surface["vo"].isel(time=0).values.squeeze(), nan=0.0)
zos = np.nan_to_num(surface["zos"].isel(time=0).values.squeeze(), nan=0.0)

payload = {
    "so": so.tolist(),
    "uo": uo.tolist(),
    "vo": vo.tolist(),
    "zos": zos.tolist(),
}

with open("test_payload.json", "w") as f:
    json.dump(payload, f)

print("Saved test_payload.json")
