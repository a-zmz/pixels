# annotations not evaluated at runtime
from __future__ import annotations

import shutil
from pathlib import Path
from functools import wraps
from typing import Any
import inspect

import numpy as np
import pandas as pd
from tables import HDF5ExtError
try:
    import zarr
    from numcodecs import Blosc
except Exception:
    zarr = None
    Blosc = None

try:
    import xarray as xr
except Exception:
    xr = None

from pixels.configs import *
from pixels.constants import *
from pixels import ioutils
from pixels.error import PixelsError
from pixels.units import SelectedUnits

def _ensure_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _resolve_cache_dirs(spec, inst):
    """
    Resolve cache_dir spec into a non-empty list of Paths.

    spec may be:
      - None
      - Path | str
      - list[Path | str]
      - callable(self) -> Path | str | list[Path | str]
    """
    if spec is None:
        return []

    value = spec(inst) if callable(spec) else spec
    values = _ensure_list(value)

    return [Path(v) for v in values if v is not None]


def _safe_key(s: str) -> str:
    return str(s).replace("/", "_").replace(".", "_")


def _make_default_compressor() -> Any:
    if Blosc is None:
        return None
    return Blosc(cname="zstd", clevel=9, shuffle=Blosc.SHUFFLE)

# ---------------------------
# xarray <-> DataFrame helpers
# ---------------------------

def _default_names(names: list[str | None], prefix: str) -> list[str]:
    # Replace None level names with defaults: f"{prefix}{i}"
    return [n if n is not None else f"{prefix}{i}" for i, n in enumerate(names)]

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
    If df.index is MultiIndex, we reset it into coordinate variables and record
    attrs so we can reconstruct on read.

    Provide either path (DirectoryStore path) or (store, group).
    """
    if xr is None or zarr is None:
        raise ImportError(
            "xarray/zarr not installed. pip install xarray zarr numcodecs"
        )

    row_prefix = "row"
    col_prefix = "col"

    # ensure all index/column level names are defined
    if isinstance(df.index, pd.MultiIndex):
        row_names = _default_names(list(df.index.names), row_prefix)
        row_is_multi = True
    else:
        row_names = [df.index.name or f"{row_prefix}0"]
        row_is_multi = False

    if isinstance(df.columns, pd.MultiIndex):
        col_names = _default_names(list(df.columns.names), col_prefix)
        col_is_multi = True
    else:
        col_names = [df.columns.name or f"{col_prefix}0"]
        col_is_multi = False

    ds = xr.Dataset(
        data_vars={
            "values": (("__row__", "__col__"), df.to_numpy(copy=False))
        },
        coords={
            "__row__": np.arange(df.shape[0]),
            "__col__": np.arange(df.shape[1]),
        },
    )

    row_level_vars = []
    col_level_vars = []
    # store row index levels
    if row_is_multi:
        for i, name in enumerate(row_names):
            arr = df.index.get_level_values(i).to_numpy()
            if arr.dtype == object:
                arr = arr.astype(str)

            var = f"__row_{_safe_key(str(name))}__"
            ds[var] = ("__row__", arr)
            row_level_vars.append(var)
    else:
        arr = df.index.to_numpy()
        if arr.dtype == object:
            arr = arr.astype(str)

        name = row_names[0] if row_names else "index"
        var = f"__row_{_safe_key(str(name))}__"
        ds[var] = ("__row__", arr)
        row_level_vars.append(var)

    # store column index levels
    if col_is_multi:
        for i, name in enumerate(col_names):
            arr = df.columns.get_level_values(i).to_numpy()
            if arr.dtype == object:
                arr = arr.astype(str)

            var = f"__col_{_safe_key(str(name))}__"
            ds[var] = ("__col__", arr)
            col_level_vars.append(var)
    else:
        arr = df.columns.to_numpy()
        if arr.dtype == object:
            arr = arr.astype(str)

        name = col_names[0] if col_names else "columns"
        var = f"__col_{_safe_key(str(name))}__"
        ds[var] = ("__col__", arr)
        col_level_vars.append(var)

    # mark attrs for round-trip
    ds.attrs["__via"] = "pd_df_any_mi_2d"
    ds.attrs["__row_dims__"] = row_names
    ds.attrs["__col_dims__"] = col_names
    ds.attrs["__row_is_multiindex__"] = row_is_multi
    ds.attrs["__col_is_multiindex__"] = col_is_multi
    ds.attrs["__row_level_vars__"] = row_level_vars
    ds.attrs["__col_level_vars__"] = col_level_vars

    # safe chunking for 2D values
    itemsize = np.dtype(df.to_numpy(copy=False).dtype).itemsize
    target_bytes = 128 * 1024**2

    nrow, ncol = df.shape

    row_chunk = min(nrow, BIG_CHUNKS*2)
    col_chunk = min(ncol, max(1, target_bytes // max(1, row_chunk * itemsize)))

    # if still too large due to large dtype/object, reduce row_chunk
    while row_chunk * col_chunk * itemsize > target_bytes:
        row_chunk = max(1, row_chunk // 2)

    ds = ds.chunk({
        "__row__": row_chunk,
        "__col__": col_chunk,
    })

    if compressor is None:
        compressor = _make_default_compressor()

    encoding = {
        "values": {
            "compressor": compressor,
            "chunks": (row_chunk, col_chunk),
        }
    }

    # index-level arrays are 1D; chunk them safely
    for name in ds.data_vars:
        if name != "values":
            size = ds[name].sizes[ds[name].dims[0]]
            encoding[name] = {
                "compressor": compressor,
            }

    if path is not None:
        ds.to_zarr(str(path), mode=mode, encoding=encoding)
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
            chunks=None,
        )
    else:
        ds = xr.open_zarr(
            store=store,
            group=group_name or "",
            consolidated=False,
            chunks=None,
        )

    via = ds.attrs.get("__via")

    if via == "pd_df_any_mi_2d":
        return _df_from_zarr_via_xarray_2d_layout(ds)

    # backward-compatible old reader
    return _df_from_zarr_via_xarray_old_layout(ds)

def _df_from_zarr_via_xarray_old_layout(ds: xr.Dataset) -> pd.DataFrame:
    """
    Read a DataFrame written by _df_to_zarr_via_xarray and reconstruct
    MultiIndex if attrs exist, using the old layout.
    """
    da = ds["values"]
    row_dim = list(ds.attrs.get("__row_dims__") or [])
    col_dim = list(ds.attrs.get("__col_dims__") or [])

    # >>> gpt version >>>
    # fast path: no column dims -> DataFrame with single column "values"
    if not col_dim:
        # bring data in the correct dim order
        da2 = da.transpose(*row_dim)
        data = da2.data
        # compute to numpy (uses on-disk chunking; no large rechunk)
        values = data.compute() if hasattr(data, "compute")\
                                else np.asarray(data)
        # build index
        row_levels = [ds.coords[n].values for n in row_dim]
        if len(row_levels) == 1:
            index = pd.Index(row_levels[0], name=row_dim[0])
        else:
            index = pd.MultiIndex.from_product(row_levels, names=row_dim)

        df = pd.DataFrame({"values": values.reshape((-1,))}, index=index)
        df.columns.name = (col_dim[0] if col_dim else None)

        return df

    # general path: row dims + col dims -> 2D dataframe
    # ensure data is ordered as [row_dims..., col_dims...]
    da2 = da.transpose(*(row_dim + col_dim))
    data = da2.data
    values = data.compute() if hasattr(data, "compute") else np.asarray(data)

    # reshape to 2D: (nrows, ncols)
    nrows = int(np.prod([da2.sizes[d] for d in row_dim]))
    ncols = int(np.prod([da2.sizes[d] for d in col_dim]))
    values2d = values.reshape((nrows, ncols))

    # build row index from row level coords
    row_levels = [ds.coords[n].values for n in row_dim]
    if len(row_levels) == 1:
        row_index = pd.Index(row_levels[0], name=row_dim[0])
    else:
        row_index = pd.MultiIndex.from_product(row_levels, names=row_dim)

    # build column index from column level coords
    col_levels = [ds.coords[n].values for n in col_dim]
    if len(col_levels) == 1:
        col_index = pd.Index(col_levels[0], name=col_dim[0])
    else:
        col_index = pd.MultiIndex.from_product(col_levels, names=col_dim)

    df = pd.DataFrame(values2d, index=row_index, columns=col_index)
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)

    return df

def _df_from_zarr_via_xarray_2d_layout(ds: xr.Dataset) -> pd.DataFrame:
    da = ds["values"]

    # ensure expected order
    da = da.transpose("__row__", "__col__")

    data = da.data
    values = data.compute() if hasattr(data, "compute") else np.asarray(data)

    row_names = list(ds.attrs.get("__row_dims__") or [])
    col_names = list(ds.attrs.get("__col_dims__") or [])

    row_is_multi = bool(ds.attrs.get("__row_is_multiindex__", False))
    col_is_multi = bool(ds.attrs.get("__col_is_multiindex__", False))

    # reconstruct row index
    if row_is_multi:
        row_arrays = [
            ds[var].values
            for var in list(ds.attrs.get("__row_level_vars__") or [])
        ]
        row_index = pd.MultiIndex.from_arrays(row_arrays, names=row_names)
    else:
        row_index = pd.Index(ds[row_level_vars[0]].values, name=row_names[0])

    # reconstruct column index
    if col_is_multi:
        col_arrays = [
            ds[var].values
            for var in list(ds.attrs.get("__col_level_vars__") or [])
        ]
        col_index = pd.MultiIndex.from_arrays(col_arrays, names=col_names)
    else:
        col_index = pd.Index(ds[col_level_vars[0]].values, name=col_names[0])

    return pd.DataFrame(values, index=row_index, columns=col_index)

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


def _filter_reserved_kwargs(fn, reserved: dict[str, Any]) -> dict[str, Any]:
    """
    Return subset of `reserved` that `fn` will accept (has a parameter by that
    name, or has **kwargs).
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return {}
    # if method has **kwargs, pass everything
    if any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    ):
        return reserved

    # otherwise pass only the names it declares
    return {k: v for k, v in reserved.items() if k in sig.parameters}

# -----------------------
# Decorator with Zarr
# -----------------------

def cacheable(
    _func=None,
    *,
    cache_format: str | None = None,
    cache_dir: Path | str | None = None,
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
      - zarr_chunks: int (rows per chunk for DataFrame; 1D arrays) or tuple/dict
        for arrays/xarray
      - zarr_compressor: a numcodecs compressor (e.g., Blosc(...)); default
        zstd+bitshuffle
      - zarr_dim_name: optional row-dimension name when writing DataFrame via
        xarray
    """
    deco_backend = cache_format

    def decorator(method):
        @wraps(method)
        def wrapper(*args, **kwargs):
            name = kwargs.pop("name", None)

            # Per-call overrides
            per_call_backend = kwargs.pop("cache_format", None)
            per_call_cache_dir = kwargs.pop("cache_dir", None)
            per_call_chunks = kwargs.pop("zarr_chunks", zarr_chunks)
            per_call_compressor = kwargs.pop("zarr_compressor", zarr_compressor)
            per_call_dim_name = kwargs.pop("zarr_dim_name", zarr_dim_name)

            inst = args[0]

            # Units gating (unchanged)
            if "units" in kwargs:
                units = kwargs["units"]
                if not isinstance(units, SelectedUnits)\
                    or not hasattr(units, "name"):
                    return method(*args, **kwargs)

            if not getattr(inst, "_use_cache", True):
                return method(*args, **kwargs)

            # Build key parts (unchanged)
            self_, *as_list = list(args) + list(kwargs.values())
            arrays = [
                i for i, arg in enumerate(as_list)
                if isinstance(arg, np.ndarray)
            ]
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

            # Resolve cache dirs with precedence:
            # per-call cache_dir > decorator cache_dir > instance _cache_dir >
            # instance cache
            cache_dirs = (
                _resolve_cache_dirs(per_call_cache_dir, inst)
                or _resolve_cache_dirs(cache_dir, inst)
                or _resolve_cache_dirs(getattr(inst, "_cache_dir", None), inst)
                or _resolve_cache_dirs(getattr(inst, "cache", None), inst)
            )

            if not cache_dirs:
                raise PixelsError(
                    "No cache directory available. "
                    "Provide cache_dir or set inst.cache."
                )

            cache_key = "_".join(key_parts) + f"_{inst.stream_id}"
            bases = [d / cache_key for d in cache_dirs]
            write_base = bases[0]

            """
            def resolve_dir(spec):
                if spec is None:
                    return None
                return spec(inst) if callable(spec) else spec

            # set base dir
            base_dir = (
                resolve_dir(per_call_cache_dir)
                or resolve_dir(cache_dir)
                or getattr(inst, "_cache_dir", None)
                or getattr(inst, "cache", None)
            )
            if base_dir is None:
                raise PixelsError(
                    "No cache directory available. "
                    "Provide cache_dir or set inst.cache."
                )
            base_dir = Path(base_dir)
            base = base_dir / ("_".join(key_parts) + f"_{inst.stream_id}")
            """

            backend = (per_call_backend
                or deco_backend
                or getattr(inst, "_cache_format", None)
                or "hdf5"
            )

            # HDF5 backend
            if backend == "hdf5":
                cache_paths = [b.with_name(b.name + ".h5") for b in bases]

                cache_path = None
                if inst._use_cache != "overwrite":
                    for p in cache_paths:
                        if p.exists():
                            cache_path = p
                            break

                if cache_path is not None:
                    try:
                        df = ioutils.read_hdf5(cache_path)
                        logging.info(f"\n\t> Cache loaded from {cache_path}.")
                    except HDF5ExtError:
                        df = None
                        logging.info("\n\t> df is None, cache does not exist.")
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
                        logging.info(f"\n\t> Cache loaded from {cache_path}.")
                else:
                    df = method(*args, **kwargs)
                    write_path = write_base.with_name(write_base.name + ".h5")
                    write_path.parent.mkdir(parents=True, exist_ok=True)

                    if df is None:
                        write_path.touch()
                        logging.info(
                            "\n\t> df is None, cache will exist but be empty."
                        )
                    else:
                        if isinstance(df, dict):
                            if ioutils.is_nested_dict(df):
                                for probe_id, nested_dict in df.items():
                                    for nm, values in nested_dict.items():
                                        ioutils.write_hdf5(
                                            path=write_path,
                                            df=values,
                                            key=f"/{probe_id}/{nm}",
                                            mode="a",
                                        )
                            else:
                                for nm, values in df.items():
                                    ioutils.write_hdf5(
                                        path=write_path,
                                        df=values,
                                        key=nm,
                                        mode="a",
                                    )
                        else:
                            ioutils.write_hdf5(write_path, df)
                    cache_path = write_path
                return df

            # Zarr backend (with DataFrame via xarray, MultiIndex supported)
            if backend == "zarr":
                if zarr is None:
                    raise ImportError(
                        "cache_format='zarr' requires zarr. "
                        "pip install zarr numcodecs xarray"
                    )
                zarr_paths = [b.with_name(b.name + ".zarr") for b in bases]

                read_path = None
                if inst._use_cache != "overwrite":
                    for p in zarr_paths:
                        if p.exists():
                            read_path = p
                            break

                if read_path is not None:
                    try:
                        obj = _read_zarr_generic(read_path)
                        logging.info(f"\n\t> Zarr cache loaded from {read_path}.")
                        return obj
                    except Exception as e:
                        logging.info(
                            f"\n\t> Failed to read Zarr cache from {read_path} ({e}); "
                            "recomputing."
                        )

                zarr_path = write_base.with_name(write_base.name + ".zarr")

                # inject reserved kwargs so the method can write directly to
                # store, if the method accepts
                reserved = {"_zarr_out": zarr_path}
                kwargs.update(_filter_reserved_kwargs(method, reserved))

                # Compute fresh
                result = method(*args, **kwargs)
                if result is None:
                    # Method handled writing itself; read and return
                    obj = _read_zarr_generic(zarr_path)
                    logging.info(f"\n\t> Zarr cache written to {zarr_path}.")
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
                        "\n\t> Zarr cache (DataFrame via xarray) written to "
                        f"{zarr_path}."
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
                    logging.info(f"\n\t> Zarr cache written to {zarr_path}.")
                    return result

                # Fallback for unsupported types: write HDF5 like before
                logging.warning(
                    "\tcache_format='zarr' requested but result type "
                    "not supported for Zarr; falling back to HDF5."
                )
                h5_fallback = write_base.with_suffix(".h5")
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
                logging.info(
                    f"\n\t> Cache written to {h5_fallback} (fallback)."
                )
                return result

            raise ValueError(f"Unknown cache_format/backend: {backend}")

        return wrapper

    if _func is None:
        return decorator
    else:
        return decorator(_func)
