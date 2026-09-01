import argparse
import xarray as xr
import numpy as np
import requests
import pandas as pd

API_BASE = "http://localhost:8001"
SURFACE_DATA_PATH = "data/surface_025_2022.nc"

def get_day_arrays(date_str):
    surface = xr.open_dataset(SURFACE_DATA_PATH)
    all_dates = pd.date_range("2022-01-01", periods=surface.dims["time"], freq="D")
    idx = all_dates.get_loc(date_str)

    so = np.nan_to_num(surface["so"].isel(time=idx).values.squeeze(), nan=0.0)
    uo = np.nan_to_num(surface["uo"].isel(time=idx).values.squeeze(), nan=0.0)
    vo = np.nan_to_num(surface["vo"].isel(time=idx).values.squeeze(), nan=0.0)
    zos = np.nan_to_num(surface["zos"].isel(time=idx).values.squeeze(), nan=0.0)
    return so, uo, vo, zos

def cmd_reconstruct(args):
    so, uo, vo, zos = get_day_arrays(args.date)
    payload = {"so": so.tolist(), "uo": uo.tolist(), "vo": vo.tolist(), "zos": zos.tolist()}
    resp = requests.post(f"{API_BASE}/predict", json=payload)
    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text}")
        return
    data = resp.json()
    depth_idx = data["depths"].index(args.depth)

    if args.lat is not None and args.lon is not None:
        lat_idx = min(range(len(data["lat"])), key=lambda i: abs(data["lat"][i] - args.lat))
        lon_idx = min(range(len(data["lon"])), key=lambda i: abs(data["lon"][i] - args.lon))
        temp = data["temperature"][lat_idx][lon_idx][depth_idx]
        unc = data["uncertainty"][lat_idx][lon_idx][depth_idx]
        print(f"OceanEmbed | {args.date} | {args.depth}m | ({args.lat}, {args.lon})")
        print(f"  Temperature: {temp:.2f} °C")
        print(f"  Uncertainty: ±{unc:.2f} °C")
    else:
        flat_vals = [row[depth_idx] for r in data["temperature"] for row in r]
        print(f"OceanEmbed | {args.date} | {args.depth}m | full grid ({len(data['lat'])}x{len(data['lon'])})")
        print(f"  Mean temperature: {np.mean(flat_vals):.2f} °C")

def cmd_status(args):
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        print(f"Model service: {resp.json()}")
    except Exception as e:
        print(f"Model service unreachable: {e}")

def main():
    parser = argparse.ArgumentParser(prog="oceanembed", description="OceanEmbed CLI — VARUNA-Net ocean temperature reconstruction")
    sub = parser.add_subparsers(dest="command", required=True)

    p_recon = sub.add_parser("reconstruct", help="Get reconstructed temperature + uncertainty")
    p_recon.add_argument("--date", required=True, help="YYYY-MM-DD")
    p_recon.add_argument("--depth", required=True, type=float, help="Depth in meters")
    p_recon.add_argument("--lat", type=float, default=None, help="Optional: specific latitude")
    p_recon.add_argument("--lon", type=float, default=None, help="Optional: specific longitude")
    p_recon.set_defaults(func=cmd_reconstruct)

    p_status = sub.add_parser("status", help="Check model service health")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()