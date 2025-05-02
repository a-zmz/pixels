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
        key_parts = [method.__name__] + [str(i) for i in as_list]
        cache_path = self.cache /\
                ("_".join(key_parts) + f"_{self.stream_id}.h5")

        if cache_path.exists() and self._use_cache != "overwrite":
            # load cache
            try:
                df = ioutils.read_hdf5(cache_path)
                logging.info(f"\n> Cache loaded from {cache_path}.")
            except HDF5ExtError:
                df = None
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
                elif isinstance(df, dict):
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
