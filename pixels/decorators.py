# annotations not evaluated at runtime
from __future__ import annotations

import shutil
from pathlib import Path
from functools import wraps
from typing import Any

import numpy as np
import pandas as pd
from tables import HDF5ExtError
try:
    import zarr
    from numcodecs import Blosc, VLenUTF8
except Exception:
    zarr = None
    Blosc = None
    VLenUTF8 = None

try:
    import xarray as xr
except Exception:
    xr = None

from pixels.configs import *
from pixels import ioutils
from pixels.error import PixelsError
from pixels.units import SelectedUnits


def _safe_key(s: str) -> str:
    return str(s).replace("/", "_").replace(".", "_")


def _make_default_compressor() -> Any:
    if Blosc is None:
        return None
    return Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)

# ---------------------------
# xarray <-> DataFrame helpers
# ---------------------------

def _df_to_zarr_via_xarray(
    df: pd.DataFrame,
    *,
    path: Path | None = None,
    store: "zarr.storage.Store" | None = None,
    group_name: str | None = None,
    compressor=None,
    mode: str = "w",
) -> None:
    """
    Write DataFrame (supports MultiIndex) to a Zarr store/group via xarray.
    If df.index is MultiIndex, we reset it into coordinate variables and record attrs
    so we can reconstruct on read.

    Provide either path (DirectoryStore path) or (store, group).
    """
    if xr is None or zarr is None:
        raise ImportError(
            "xarray/zarr not installed. pip install xarray zarr numcodecs"
        )

    row_prefix = "row"
    col_prefix = "col"

    # Ensure all index/column level names are defined
    if isinstance(df.index, pd.MultiIndex):
        row_names = _default_names(list(df.index.names), row_prefix)
    else:
        row_names = [df.index.name or f"{row_prefix}0"]

    if isinstance(df.columns, pd.MultiIndex):
        col_names = _default_names(list(df.columns.names), col_prefix)
    else:
        col_names = [df.columns.name or f"{col_prefix}0"]

    # Stack ALL column levels to move them into the row index; result index levels = row_names + col_names
    series = df.stack(col_names, future_stack=True)  # Series with MultiIndex index

    # Build DataArray (dims are level names of the Series index, in order)
    da = xr.DataArray.from_series(series).rename("values")
    ds = da.to_dataset()

    # Mark attrs for round-trip (which dims belong to rows vs columns)
    ds.attrs["__via"] = "pd_df_any_mi"
    ds.attrs["__row_dims__"] = row_names
    ds.attrs["__col_dims__"] = col_names

    # check size to determine chunking
    chunking = {}
    for name, size in ds.sizes.items():
        if size > BIG_CHUNKS:
            chunking[name] = BIG_CHUNKS
        else:
            chunking[name] = SMALL_CHUNKS

    ds = ds.chunk(chunking)

    if compressor is None:
        compressor = _make_default_compressor()
    # compressor & object codec
    encoding = {
        "values": {
            "compressor": compressor,
            "chunks": tuple(chunking.values()),
        }
    }
    if ds["values"].dtype == object and VLenUTF8 is not None:
        encoding["values"]["object_codec"] = VLenUTF8()

    # Ensure coords are writable (handle object/string coords)
    # If VLenUTF8 is available, set encoding for object coords; otherwise cast
    # to str
    for cname, coord in ds.coords.items():
        if coord.dtype == object:
            if VLenUTF8 is not None:
                encoding[cname] = {
                    "object_codec": VLenUTF8(),
                    "compressor": compressor,
                }
            else:
                ds = ds.assign_coords({cname: coord.astype(str)})

    # Write
    if path is not None:
        ds.to_zarr(
            str(path),
            mode=mode,
            encoding=encoding,
        )
        try:
            zarr.consolidate_metadata(str(path))
        except Exception:
            pass
    else:
        assert store is not None
        ds.to_zarr(
            store=store,
            group=group_name or "",
            mode=mode,
            encoding=encoding,
        )
        # consolidate requires a path; skipping here since we're inside a shared
        # store


def _df_from_zarr_via_xarray(
    *,
    path: Path | None = None,
    store: "zarr.storage.Store" | None = None,
    group_name: str | None = None,
) -> pd.DataFrame:
    """
    Read a DataFrame written by _df_to_zarr_via_xarray and reconstruct
    MultiIndex if attrs exist.
    Provide either path or (store, group).
    """
    if xr is None or zarr is None:
        raise ImportError(
            "xarray/zarr not installed. pip install xarray zarr numcodecs"
        )

    if path is not None:
        ds = xr.open_zarr(
            str(path),
            consolidated=True,
            chunks="auto",
        )
    else:
        ds = xr.open_zarr(
            store=store,
            group=group_name or "",
            consolidated=False,
            chunks="auto",
        )

    da = ds["values"]
    row_dim = list(ds.attrs.get("__row_dims__") or [])
    col_dim = list(ds.attrs.get("__col_dims__") or [])

    # Series with MultiIndex index (row_dim, *col_dim)
    series = da.to_series()

    # If there are column dims, unstack them back to columns
    if col_dim:
        df = series.unstack(col_dim)
    else:
        # No column dims -> a single column DataFrame
        df = series.to_frame(name="values")

    col_name = [df.columns.name]
    if not (row_dim == df.index.names):
        df.index.set_names(row_dim, inplace=True)
    if not (col_dim == col_name):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns.set_names(col_dim, inplace=True)
        else:
            df.columns.name = col_dim[0]

    return df


# -----------------------------------
# Zarr read/write for arrays and dicts
# -----------------------------------

def _normalise_1d_chunks(chunks: Any, n: int) -> Any:
    if chunks is None:
        return None
    if isinstance(chunks, int):
        return (min(chunks, n),)
    if isinstance(chunks, (tuple, list)):
        if len(chunks) == 0:
            return None
        return (min(int(chunks[0]), n),)
    return None


def _write_arrays_dicts_to_zarr(
    root_path: Path,
    obj: Any,
    *,
    chunks: Any = None,
    compressor: Any = None,
    overwrite: bool = False,
) -> None:
    """
    Write ndarray or dict/nested-dict of ndarrays/DataFrames into a Zarr
    directory.
    DataFrames inside dicts are written via xarray into corresponding groups.
    Top-level pure DataFrame should be written with _df_to_zarr_via_xarray
    instead.
    """
    if zarr is None:
        raise ImportError(
            "zarr/numcodecs not installed. pip install zarr numcodecs"
        )

    store = zarr.DirectoryStore(str(root_path))
    if overwrite and root_path.exists():
        shutil.rmtree(root_path)
    root = zarr.group(
        store=store,
        overwrite=overwrite or (not root_path.exists())
    )

    def write_into(prefix: str, value: Any):
        # prefix: group path relative to root ("" for root)
        if isinstance(value, np.ndarray):
            g = zarr.open_group(store=store, path=prefix, mode="a")
            name = "array" if prefix == "" else prefix.split("/")[-1]
            # In groups, datasets live as siblings; for arrays we use the
            # current group's name
            # Better: use a fixed name for standalone arrays in a group
            # Here we store as "values" for groups, or "array" at root if
            # top-level ndarray
            ds_name = "array" if prefix == "" else "values"
            if ds_name in g:
                del g[ds_name]
            ds_chunks = chunks
            if isinstance(chunks, int) and value.ndim == 1:
                ds_chunks = _normalise_1d_chunks(chunks, len(value))
            g.create_dataset(
                name=ds_name,
                data=value,
                chunks=ds_chunks,
                compressor=compressor,
            )

        elif isinstance(value, pd.DataFrame):
            # Write DF via xarray under this group path
            _df_to_zarr_via_xarray(
                value,
                store=store,
                group_name=prefix or "",
                compressor=compressor,
                mode="w",
            )

        elif isinstance(value, dict):
            # Recurse for each item
            for k, v in value.items():
                key = _safe_key(k)
                next_prefix = f"{prefix}/{key}" if prefix else key
                # Ensure group exists
                zarr.open_group(store=store, path=prefix, mode="a")
                write_into(next_prefix, v)

        else:
            raise TypeError(
                "Zarr backend supports ndarray, DataFrame, or dicts of them. "
                f"Got: {type(value)} at group '{prefix or '/'}'"
            )

    if isinstance(obj, dict):
        for k, v in obj.items():
            write_into(_safe_key(k), v)
    elif isinstance(obj, np.ndarray):
        write_into("", obj)
    else:
        raise TypeError(
            "Top-level object must be ndarray or dict for this writer."
        )

    # Optional consolidation (best when writing via path)
    try:
        zarr.consolidate_metadata(str(root_path))
    except Exception:
        pass


def _read_zarr_generic(root_path: Path) -> Any:
    """
    Read back what _write_arrays_dicts_to_zarr wrote and also detect DataFrame
    groups written via xarray (by checking group attrs['__via']).

    Returns:
      - DataFrame (if top-level was DF written via xarray)
      - zarr.Array (if top-level ndarray)
      - dict tree mixing DataFrames and zarr.Arrays
    """
    if zarr is None:
        raise ImportError(
            "zarr/numcodecs not installed. pip install zarr numcodecs"
        )

    store = zarr.DirectoryStore(str(root_path))
    if not root_path.exists():
        return None
    root = zarr.open_group(store=store, mode="r")

    # If top-level was written via xarray as a DataFrame
    if root.attrs.get("__via") == "pd_df_any_mi" and xr is not None:
        return _df_from_zarr_via_xarray(store=store, group_name="")

    # If top-level is a single array written at root
    if "array" in root and isinstance(root["array"], zarr.Array):
        return root["array"]

    def read_from_group(prefix: str) -> Any:
        g = zarr.open_group(store=store, path=prefix, mode="r")
        # DataFrame group?
        if g.attrs.get("__via") == "pd_df_any_mi" and xr is not None:
            return _df_from_zarr_via_xarray(
                store=store,
                group_name=prefix or "",
            )

        out: dict[str, Any] = {}
        for name, node in g.items():
            full = f"{prefix}/{name}" if prefix else name
            if isinstance(node, zarr.Array):
                # Arrays inside groups are stored as "values"
                if name == "values" and prefix:
                    out[prefix.split("/")[-1]] = node
                else:
                    out[name] = node
            elif isinstance(node, zarr.hierarchy.Group):
                # Recurse
                res = read_from_group(full)
                # If res is a DF and 'name' was only a container, store under
                # that key
                out[name] = res
        return out

    return read_from_group("")


# -----------------------
# Decorator with Zarr
# -----------------------

def cacheable(
    _func=None,
    *,
    cache_format: str | None = None,
    zarr_chunks: Any = None,
    zarr_compressor: Any = None,
    zarr_dim_name: str | None = None,
):
    """
    Decorator factory for caching.

    Usage:
      @cacheable  # default HDF5
      def f(...): ...

      @cacheable(cache_format='zarr')  # Zarr default for this method
      def g(...): ...

    Backend precedence:
      per-call kwarg cache_format > per-method (decorator args)
      > instance default self._cache_format > 'hdf5'

    Zarr options:
      - zarr_chunks: int (rows per chunk for DataFrame; 1D arrays) or tuple/dict for arrays/xarray
      - zarr_compressor: a numcodecs compressor (e.g., Blosc(...)); default zstd+bitshuffle
      - zarr_dim_name: optional row-dimension name when writing DataFrame via xarray
    """
    default_backend = cache_format or "hdf5"

    def decorator(method):
        @wraps(method)
        def wrapper(*args, **kwargs):
            name = kwargs.pop("name", None)

            # Per-call overrides
            per_call_backend = kwargs.pop("cache_format", None)
            per_call_chunks = kwargs.pop("zarr_chunks", zarr_chunks)
            per_call_compressor = kwargs.pop("zarr_compressor", zarr_compressor)
            per_call_dim_name = kwargs.pop("zarr_dim_name", zarr_dim_name)

            inst = args[0]

            # Units gating (unchanged)
            if "units" in kwargs:
                units = kwargs["units"]
                if not isinstance(units, SelectedUnits) or not hasattr(units, "name"):
                    return method(*args, **kwargs)

            if not getattr(inst, "_use_cache", True):
                return method(*args, **kwargs)

            # Build key parts (unchanged)
            self_, *as_list = list(args) + list(kwargs.values())
            arrays = [i for i, arg in enumerate(as_list) if isinstance(arg, np.ndarray)]
            if arrays:
                if name is None:
                    raise PixelsError(
                        "Cacheing methods when passing arrays requires also "
                        "passing name='something'"
                    )
                for i in arrays:
                    as_list[i] = name

            key_parts = [method.__name__] + [
                str(i.name) if hasattr(i, "name") else str(i) for i in as_list
            ]
            base = inst.cache / ("_".join(key_parts) + f"_{inst.stream_id}")

            backend = per_call_backend\
                    or getattr(inst, "_cache_format", None)\
                    or default_backend

            # HDF5 backend
            if backend == "hdf5":
                cache_path = base.with_suffix(".h5")
                if cache_path.exists() and inst._use_cache != "overwrite":
                    try:
                        df = ioutils.read_hdf5(cache_path)
                        logging.info(f"\n> Cache loaded from {cache_path}.")
                    except HDF5ExtError:
                        df = None
                        logging.info("\n> df is None, cache does not exist.")
                    except (KeyError, ValueError):
                        df = {}
                        with pd.HDFStore(cache_path, "r") as store:
                            for key in store.keys():
                                parts = key.lstrip("/").split("/")
                                if len(parts) == 1:
                                    df[parts[0]] = store[key]
                                elif len(parts) == 2:
                                    stream, nm = parts[0], "/".join(parts[1:])
                                    df.setdefault(stream, {})[nm] = store[key]
                        logging.info(f"\n> Cache loaded from {cache_path}.")
                else:
                    df = method(*args, **kwargs)
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    if df is None:
                        cache_path.touch()
                        logging.info("\n> df is None, cache will exist but be empty.")
                    else:
                        if isinstance(df, dict):
                            if ioutils.is_nested_dict(df):
                                for probe_id, nested_dict in df.items():
                                    for nm, values in nested_dict.items():
                                        ioutils.write_hdf5(
                                            path=cache_path,
                                            df=values,
                                            key=f"/{probe_id}/{nm}",
                                            mode="a",
                                        )
                            else:
                                for nm, values in df.items():
                                    ioutils.write_hdf5(
                                        path=cache_path, df=values, key=nm, mode="a"
                                    )
                        else:
                            ioutils.write_hdf5(cache_path, df)
                return df

            # Zarr backend (with DataFrame via xarray, MultiIndex supported)
            if backend == "zarr":
                if zarr is None:
                    raise ImportError(
                        "cache_format='zarr' requires zarr. pip install zarr numcodecs xarray"
                    )
                zarr_path = base.with_suffix(".zarr")
                can_read = zarr_path.exists() and inst._use_cache != "overwrite"

                if can_read:
                    try:
                        obj = _read_zarr_generic(zarr_path)
                        logging.info(f"\n> Zarr cache loaded from {zarr_path}.")
                        return obj
                    except Exception as e:
                        logging.info(f"\n> Failed to read Zarr cache ({e}); recomputing.")

                # inject reserved kwargs so the method can write directly to
                # store
                kwargs["_zarr_out"] = zarr_path

                # Compute fresh
                result = method(*args, **kwargs)
                if result is None:
                    # Method handled writing itself; read and return
                    obj = _read_zarr_generic(zarr_path)
                    logging.info(f"\n> Zarr cache written to {zarr_path}.")
                    return obj

                # Overwrite
                if inst._use_cache == "overwrite" and zarr_path.exists():
                    shutil.rmtree(zarr_path)

                compressor = per_call_compressor or _make_default_compressor()

                # DataFrame via xarray (works for MultiIndex)
                if isinstance(result, pd.DataFrame):
                    _df_to_zarr_via_xarray(
                        result,
                        path=zarr_path,
                        compressor=compressor,
                        mode="w",
                    )
                    logging.info(
                        f"\n> Zarr cache (DataFrame via xarray) written to {zarr_path}."
                    )
                    return result

                # Dict/nested-dict of arrays or DataFrames
                if isinstance(result, dict) or isinstance(result, np.ndarray):
                    _write_arrays_dicts_to_zarr(
                        zarr_path,
                        result,
                        chunks=per_call_chunks,
                        compressor=compressor,
                        overwrite=True,
                    )
                    logging.info(f"\n> Zarr cache written to {zarr_path}.")
                    return result

                # Fallback for unsupported types: write HDF5 like before
                logging.warning(
                    "cache_format='zarr' requested but result type "
                    "not supported for Zarr; falling back to HDF5."
                )
                h5_fallback = base.with_suffix(".h5")
                if isinstance(result, dict):
                    if ioutils.is_nested_dict(result):
                        for probe_id, nested_dict in result.items():
                            for nm, values in nested_dict.items():
                                ioutils.write_hdf5(
                                    path=h5_fallback,
                                    df=values,
                                    key=f"/{probe_id}/{nm}",
                                    mode="a",
                                )
                    else:
                        for nm, values in result.items():
                            ioutils.write_hdf5(
                                path=h5_fallback,
                                df=values,
                                key=str(nm),
                                mode="a",
                            )
                else:
                    ioutils.write_hdf5(h5_fallback, result)
                logging.info(f"\n> Cache written to {h5_fallback} (fallback).")
                return result

            raise ValueError(f"Unknown cache_format/backend: {backend}")

        return wrapper

    if _func is None:
        return decorator
    else:
        return decorator(_func)

'''
import numpy as np
import pandas as pd
from tables import HDF5ExtError

from pixels.configs import *
from pixels import ioutils
from pixels.error import PixelsError
from pixels.units import SelectedUnits

def cacheable(method):
    """
    Methods with this decorator will have their output cached to disk so that
    future calls with the same set of arguments will simply load the result from
    disk. However, from pixels.error import PixelsError if the key word argument
    list contains `units` and it is not either `None` or an instance of
    `SelectedUnits` then this is disabled.
    """
    def wrapper(*args, **kwargs):
        name = kwargs.pop("name", None)

        if "units" in kwargs:
            units = kwargs["units"]
            if not isinstance(units, SelectedUnits) or not hasattr(units, "name"):
                return method(*args, **kwargs)

        self, *as_list = list(args) + list(kwargs.values())
        if not self._use_cache:
            return method(*args, **kwargs)

        arrays = [i for i, arg in enumerate(as_list) if isinstance(arg, np.ndarray)]
        if arrays:
            if name is None:
                raise PixelsError(
                    "Cacheing methods when passing arrays requires also "
                    "passing name='something'"
                )
            for i in arrays:
                as_list[i] = name

        # build a key: method name + all args
        key_parts = [method.__name__] + [str(i.name) if hasattr(i, "name")
                                         else str(i) for i in as_list]
        cache_path = self.cache /\
                ("_".join(key_parts) + f"_{self.stream_id}.h5")

        if cache_path.exists() and self._use_cache != "overwrite":
            # load cache
            try:
                df = ioutils.read_hdf5(cache_path)
                logging.info(f"\n> Cache loaded from {cache_path}.")
            except HDF5ExtError:
                df = None
                logging.info("\n> df is None, cache does not exist.")
            except (KeyError, ValueError):
                # if key="df" is not found, then use HDFStore to list and read
                # all dfs
                # create df as a dictionary to hold all dfs
                df = {}
                with pd.HDFStore(cache_path, "r") as store:
                    # list all keys
                    for key in store.keys():
                        # remove "/" in key and split
                        parts = key.lstrip("/").split("/")
                        if len(parts) == 1:
                            # use the only key name as dict key
                            df[parts[0]] = store[key]
                        elif len(parts) == 2:
                            # stream id is the first, data name is the second
                            stream, name = parts[0], "/".join(parts[1:])
                            df.setdefault(stream, {})[name] = store[key]
                logging.info(f"\n> Cache loaded from {cache_path}.")
        else:
            df = method(*args, **kwargs)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if df is None:
                cache_path.touch()
                logging.info("\n> df is None, cache will exist but be empty.")
            else:
                # allows to save multiple dfs in a dict in one hdf5 file
                if isinstance(df, dict):
                    if ioutils.is_nested_dict(df):
                        for probe_id, nested_dict in df.items():
                            # NOTE: we remove `.ap` in stream id cuz having `.`in
                            # the key name get problems
                            for name, values in nested_dict.items():
                                ioutils.write_hdf5(
                                    path=cache_path,
                                    df=values,
                                    key=f"/{probe_id}/{name}",
                                    mode="a",
                                )
                    else:
                        for name, values in df.items():
                            ioutils.write_hdf5(
                                path=cache_path,
                                df=values,
                                key=name,
                                mode="a",
                            )
                else:
                    ioutils.write_hdf5(cache_path, df)
        return df
    return wrapper
'''
