#!/usr/bin/env python3
"""
Load forcing variables from a NetCDF file and populate arrays:
  ta_min, ta_max, pr, sw, lai, soilm, doy, photo

Behavior:
- Tries xarray first, falls back to netCDF4 if xarray isn't available.
- For each requested variable, tries several common candidate variable names.
- If a variable is multi-dimensional with a time dimension, collapses non-time dims by mean to produce a 1-D time series.
- If `doy` isn't present but `time` is, computes day-of-year from the time coordinate.
- Saves each array to an .npy file next to the input NetCDF by default and prints a summary.

Usage:
  python src/load_forcing.py /glade/u/home/linnia/MLphenology/US-MMS_forcing_1999-2014.nc

"""
import sys
import os
import argparse
import warnings
import numpy as np

try:
    import xarray as xr
    USE_XARRAY = True
except Exception:
    USE_XARRAY = False
    import netCDF4 as nc

from datetime import datetime

CANDIDATES = {
    'ta_min': ['ta_min', 'tmin', 'tasmin', 'tmin_inst', 'tmin_day', 'tas_min'],
    'ta_max': ['ta_max', 'tmax', 'tasmax', 'tmax_inst', 'tmax_day', 'tas_max'],
    'pr':     ['pr', 'precipitation', 'precip', 'ppt', 'prcp', 'rain'],
    'sw':     ['sw', 'srad', 'shortwave', 'swrad', 'radiation', 'sw_down', 'sw_in'],
    'lai':    ['lai', 'LAI'],
    'soilm':  ['soilm', 'swvl1', 'soil_moisture', 'soil_moist', 'sm1', 'swc'],
    'doy':    ['doy', 'dayofyear'],
    'photo':  ['photo', 'photoperiod', 'daylength']
}


def find_var_name(ds_vars, candidates):
    """Return the first candidate that exists in ds_vars (case-insensitive where useful)."""
    low_vars = {v.lower(): v for v in ds_vars}
    for c in candidates:
        # exact match first
        if c in ds_vars:
            return c
        # case-insensitive
        if c.lower() in low_vars:
            return low_vars[c.lower()]
    return None


def collapse_time_series(arr, dims, time_dim_name='time'):
    """Ensure arr is a 1-D numpy array over time.
    If arr has a time dimension, move it to axis 0 and mean over other axes.
    If no time dim is present, return flattened array.
    dims: sequence of dimension names in the same order as arr.shape
    """
    a = np.asarray(arr)
    if time_dim_name in dims:
        t_axis = dims.index(time_dim_name)
        if a.ndim == 1:
            return a
        # Move time axis to front
        if t_axis != 0:
            a = np.moveaxis(a, t_axis, 0)
        if a.ndim > 1:
            # mean over non-time axes
            a = a.mean(axis=tuple(range(1, a.ndim)))
        return a
    else:
        # No time dim - just flatten
        return a.ravel()


def load_with_xarray(path):
    ds = xr.open_dataset(path)
    results = {}

    # helper to get numpy array and dims
    def get_arr_and_dims(varname):
        var = ds[varname]
        arr = var.values
        dims = list(var.dims)
        return arr, dims

    ds_vars = list(ds.data_vars.keys()) + list(ds.coords.keys())

    for key, cand_list in CANDIDATES.items():
        found = find_var_name(ds_vars, cand_list)
        if found is not None:
            arr, dims = get_arr_and_dims(found)
            ts = collapse_time_series(arr, dims, time_dim_name='time')
            results[key] = (ts, found)
        else:
            results[key] = (None, None)

    # special compute: doy from time if missing
    if results['doy'][0] is None and 'time' in ds.coords:
        try:
            time_index = ds['time'].to_index()
            doy = np.array([t.dayofyear for t in time_index])
            results['doy'] = (doy, 'computed_from_time')
        except Exception:
            pass

    ds.close()
    return results


def load_with_netcdf4(path):
    ds = nc.Dataset(path)
    ds_vars = list(ds.variables.keys())
    results = {}

    for key, cand_list in CANDIDATES.items():
        found = find_var_name(ds_vars, cand_list)
        if found is not None:
            var = ds.variables[found]
            arr = var[:]
            dims = var.dimensions
            ts = collapse_time_series(arr, dims, time_dim_name='time')
            results[key] = (ts, found)
        else:
            results[key] = (None, None)

    # compute doy from time if missing
    if results['doy'][0] is None and 'time' in ds.variables:
        try:
            time_var = ds.variables['time']
            # Attempt to decode using netCDF4 num2date if units exist
            from netCDF4 import num2date
            time_vals = num2date(time_var[:], units=getattr(time_var, 'units', None), calendar=getattr(time_var, 'calendar', 'standard'))
            doy = np.array([t.timetuple().tm_yday for t in time_vals])
            results['doy'] = (doy, 'computed_from_time')
        except Exception:
            pass

    ds.close()
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ncfile', nargs='?', default='/glade/u/home/linnia/MLphenology/US-MMS_forcing_1999-2014.nc')
    p.add_argument('--outdir', '-o', default=None, help='Directory to save .npy outputs (default: same dir as ncfile)')
    args = p.parse_args()

    path = args.ncfile
    if not os.path.exists(path):
        print(f"ERROR: file not found: {path}")
        sys.exit(2)

    print(f"Loading: {path}  (using xarray={USE_XARRAY})")

    try:
        if USE_XARRAY:
            results = load_with_xarray(path)
        else:
            results = load_with_netcdf4(path)
    except Exception as e:
        print(f"Failed to read NetCDF: {e}")
        raise

    outdir = args.outdir or os.path.dirname(os.path.abspath(path))
    os.makedirs(outdir, exist_ok=True)

    summary = []
    for key in ['ta_min','ta_max','pr','sw','lai','soilm','doy','photo']:
        arr, src = results.get(key, (None, None))
        if arr is None:
            warnings.warn(f"Variable {key} not found in {path} (candidates: {CANDIDATES[key]})")
            continue
        arr = np.asarray(arr)
        # ensure 1-D
        if arr.ndim > 1:
            arr = arr.ravel()
        fname = os.path.join(outdir, f"{key}.npy")
        np.save(fname, arr)
        summary.append((key, fname, arr.shape, src))

    print("\nSaved variables:")
    for k, f, s, src in summary:
        print(f" - {k}: {f}  shape={s}  (source: {src})")

    if not summary:
        print("No variables were saved. Check variable names in the NetCDF file.")


if __name__ == '__main__':
    main()
