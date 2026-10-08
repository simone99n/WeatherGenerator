# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from pathlib import Path
from typing import override

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    check_reader_data,
)
from weathergen.train.utils import Stage

_logger = logging.getLogger(__name__)


class DataReaderGREP(DataReaderTimestep):
    """
    Wrapper data reader for gridded Zarr datasets, in zarr format 2 or 3.

    Two grid layouts are supported:
      * regular 1-D lat/lon, e.g. E-OBS: ``latitude(lat)``, ``longitude(lon)``
      * curvilinear 2-D, e.g. C-GLORS / NEMO ORCA: ``nav_lat(y, x)``, ``nav_lon(y, x)``

    The time dimension is detected from the data variables rather than assumed, so ``time``,
    ``time_centered``, ``time_counter`` and anything else all work. Converts the gridded data
    to ReaderData format.
    """

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
        stage: Stage | None = None,
    ) -> None:
        """
        Construct data reader for Zarr GREP dataset

        Parameters
        ----------
        filename :
            filename (and path) of dataset
        stream_info :
            information about stream
        stage :
            training stage; accepted for interface compatibility, not used

        Returns
        -------
        None
        """

        # Store configuration but DO NOT open files here (fork-safety for multiprocessing workers)
        self._filename = filename
        self._tw_handler = tw_handler
        self._stream_info = stream_info
        self._initialized = False

        # Call super() with placeholder period; will be overwritten after lazy init sets the real
        # data_start_time / data_end_time / period on this instance directly.
        super().__init__(tw_handler, stream_info)

        # Grid properties (populated during lazy init)
        self.latitudes: NDArray | None = None
        self.longitudes: NDArray | None = None
        self.n_lat: int = 0
        self.n_lon: int = 0
        self.n_points: int = 0

        # Time axis, set once by _lazy_init; see _detect_time_dim for why both are needed.
        self._time_dim: str | None = None
        self._time_coord: str | None = None

        # Opt-in, because E-OBS is a land-only grid whose NaN sea points are part of its
        # distribution. C-GLORS needs it: its land mask is a coordinate sentinel that would
        # otherwise collapse 288k points into one healpix cell.
        self._drop_all_nan = bool(stream_info.get("drop_all_nan_rows", False))

        # debug
        self.log_debug = False

        # Set empty defaults so the object is always in a valid state
        self.init_empty()
        self._lazy_init()

    def _lazy_init(self) -> None:
        """
        Open the dataset and populate all metadata.  Called once per worker.
        """
        if self._initialized:
            return
        self._initialized = True

        # No zarr_format: zarr auto-detects format 2 and 3 stores. Deliberately not wrapped in
        # try/except - a store that cannot be opened has to abort the run, because
        # MultiStreamDataSampler silently substitutes spoofed mean-valued data for an empty
        # stream and the run would then train on the remaining streams looking healthy.
        ds: xr.Dataset = xr.open_zarr(self._filename, consolidated=True, chunks=None)

        # ---- Time axis -------------------------------------------------------
        self._time_dim, self._time_coord = _detect_time_dim(ds, self._stream_info["name"])
        time_coord: NDArray = ds.coords[self._time_coord].values

        data_start_time = np.datetime64(time_coord[0])
        data_end_time = np.datetime64(time_coord[-1])

        if self._tw_handler.t_start >= data_end_time or self._tw_handler.t_end <= data_start_time:
            name = self._stream_info["name"]
            _logger.warning(f"{name} is not supported over data loader window. Stream is skipped.")
            return  # leave in empty state

        if len(time_coord) > 1:
            period = np.timedelta64(time_coord[1] - time_coord[0])
        else:
            period = np.timedelta64(1, "D")

        if "frequency" in self._stream_info:
            # Reuse the same helper the base module uses (timedelta_to_str inverse).
            # The base module exposes no str_to_timedelta, so we parse manually.
            period = _str_to_timedelta(self._stream_info["frequency"])

        # Patch the instance attributes that DataReaderTimestep._get_dataset_idxs reads.
        # This is equivalent to what super().__init__(..., data_start_time, data_end_time, period)
        # would set, but without the side-effects of calling __init__ a second time.
        self.data_start_time = data_start_time
        self.data_end_time = data_end_time
        self.period = period

        # ---- Spatial grid ----------------------------------------------------
        if "latitude" in ds.coords and "longitude" in ds.coords:
            # Regular 1-D grid (e.g. E-OBS): latitude(lat,) longitude(lon,)
            self._curvilinear = False
            self.latitudes = ds.coords["latitude"].values.astype(np.float32)
            self.longitudes = ds.coords["longitude"].values.astype(np.float32)

            if np.any(self.latitudes < -90) or np.any(self.latitudes > 90):
                _logger.warning(
                    f"Latitude values outside [-90, 90] in '{self._stream_info['name']}'; clipping."
                )
                self.latitudes = np.clip(self.latitudes, -90.0, 90.0)

            if np.any(self.longitudes < -180) or np.any(self.longitudes > 180):
                _logger.warning(
                    f"Longitude values outside [-180, 180] in '{self._stream_info['name']}'; "
                    "converting from [0, 360]."
                )
                self.longitudes = ((self.longitudes + 180.0) % 360.0 - 180.0).astype(np.float32)

            self.n_lat = len(self.latitudes)
            self.n_lon = len(self.longitudes)
            self.n_points = self.n_lat * self.n_lon

        elif "nav_lat" in ds.coords and "nav_lon" in ds.coords:
            # Curvilinear 2-D grid (e.g. C-GLORS): nav_lat(y, x), nav_lon(y, x)
            _logger.info(
                f"Dataset '{self._stream_info['name']}' uses curvilinear grid (nav_lat/nav_lon)."
            )
            self._curvilinear = True
            nav_lat = ds.coords["nav_lat"].values.astype(np.float32)  # (y, x)
            nav_lon = ds.coords["nav_lon"].values.astype(np.float32)  # (y, x)

            nav_lat = np.clip(nav_lat, -90.0, 90.0)
            nav_lon = ((nav_lon + 180.0) % 360.0 - 180.0).astype(np.float32)

            # Store flat point lists — used directly to build coords in _get()
            self._nav_lat_flat = nav_lat.flatten()  # (n_points,)
            self._nav_lon_flat = nav_lon.flatten()  # (n_points,)

            self.n_lat, self.n_lon = nav_lat.shape
            self.n_points = self.n_lat * self.n_lon

            # Provide sorted unique 1-D views for callers that inspect .latitudes/.longitudes
            self.latitudes = np.unique(self._nav_lat_flat)
            self.longitudes = np.unique(self._nav_lon_flat)

        else:
            raise ValueError(
                f"Dataset '{self._stream_info['name']}' has neither "
                "'latitude'/'longitude' nor 'nav_lat'/'nav_lon' coordinates."
            )

        # ---- Available variables (non-stat, time-varying) --------------------
        available_vars: list[str] = [
            var
            for var in ds.data_vars
            if not var.endswith("_mean")
            and not var.endswith("_std")
            and self._time_dim in ds[var].dims
        ]

        # ---- Channel selection -----------------------------------------------
        # source_idx / target_idx are indices into available_vars (like Anemoi uses ds.variables).
        source_channels_filter = self._stream_info.get("source")
        source_exclude = self._stream_info.get("source_exclude", [])
        self.source_channels, self.source_idx = self._select_channels(
            available_vars, source_channels_filter, source_exclude
        )
        self.source_idx = list(self.source_idx)  # keep as list, consistent with base class
        _logger.info(
            f"{self._stream_info['name']} selected source channels: "
            f"{self.source_channels} (indices: {self.source_idx})"
        )

        target_channels_filter = self._stream_info.get("target")
        target_exclude = self._stream_info.get("target_exclude", [])
        self.target_channels, self.target_idx = self._select_channels(
            available_vars, target_channels_filter, target_exclude
        )
        self.target_idx = list(self.target_idx)
        _logger.info(
            f"{self._stream_info['name']} selected target channels: "
            f"{self.target_channels} (indices: {self.target_idx})"
        )

        # A stream may legitimately have no channels on one side, but having none on either means
        # the filters do not match this store - a silent no-op stream otherwise.
        if available_vars and not self.source_idx and not self.target_idx:
            raise ValueError(
                f"Stream '{self._stream_info['name']}' selected no source and no target channels "
                f"from {len(available_vars)} available variables. Check the 'source' / 'target' / "
                f"'*_exclude' filters. Available: {available_vars[:20]}"
            )

        self.geoinfo_channels = []
        self.geoinfo_idx = np.array([], dtype=np.int64)
        self.mean_geoinfo = np.zeros(0, dtype=np.float32)
        self.stdev_geoinfo = np.ones(0, dtype=np.float32)

        self.target_channel_weights = self.parse_target_channel_weights()

        # ---- Statistics ------------------------------------------------------
        # mean/stdev must be arrays of length == len(available_vars) so that
        # the base-class _normalize/_denormalize can index them with source_idx / target_idx.
        # Only the selected slots are ever read, and a store may hold hundreds of variables, so
        # this needs the channel selection above to have happened already.
        self.mean, self.stdev = self._load_statistics(
            available_vars, ds, set(self.source_idx) | set(self.target_idx)
        )

        # ---- Length (timesteps inside the window) ----------------------------
        time_mask = (time_coord >= self._tw_handler.t_start) & (time_coord < self._tw_handler.t_end)
        self.len = int(np.sum(time_mask))

        # ---- Keep reference to open dataset ----------------------------------
        self.ds = ds
        self.available_vars = available_vars

        ds_name = self._stream_info["name"]
        _logger.info(f"{ds_name}: source channels: {self.source_channels}")
        _logger.info(f"{ds_name}: target channels: {self.target_channels}")
        _logger.info(f"{ds_name}: grid shape: {self.n_lat} x {self.n_lon}")

        self.properties = {"stream_id": self._stream_info.get("stream_id", 0)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _select_channels(
        self,
        available_vars: list[str],
        include_filters: list[str] | None,
        exclude_filters: list[str] | None = None,
    ) -> tuple[list[str], NDArray[np.int64]]:
        """Return (channel_names, indices_into_available_vars)."""
        if exclude_filters is None:
            exclude_filters = []

        selected_names: list[str] = []
        selected_idxs: list[int] = []

        for i, var in enumerate(available_vars):
            if include_filters is not None:
                if not any(f in var or f == var for f in include_filters):
                    continue
            if any(f in var for f in exclude_filters):
                continue
            selected_names.append(var)
            selected_idxs.append(i)

        return selected_names, np.array(selected_idxs, dtype=np.int64)

    def _load_statistics(
        self,
        available_vars: list[str],
        ds: xr.Dataset,
        needed_idx: set[int] | None = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Return mean and stdev arrays aligned with *available_vars* (not just selected channels).
        This matches the layout expected by DataReaderBase._normalize / _denormalize which index
        the arrays with source_idx / target_idx.

        *needed_idx* restricts which slots are actually read from the store. Unread slots keep
        their neutral 0.0 / 1.0 values and are never indexed. Each statistic is a separate scalar
        array in the store, so on a several-hundred-variable store reading them all costs seconds
        per worker for values nothing uses.
        """
        means = np.zeros(len(available_vars), dtype=np.float32)
        stds = np.ones(len(available_vars), dtype=np.float32)

        for i, ch in enumerate(available_vars):
            if needed_idx is not None and i not in needed_idx:
                continue

            mean_var = f"{ch}_mean"
            std_var = f"{ch}_std"

            if mean_var in ds.data_vars:
                means[i] = float(ds[mean_var].values)
            else:
                _logger.warning(f"No pre-computed mean for {ch}, using 0.0.")

            if std_var in ds.data_vars:
                stds[i] = float(ds[std_var].values)
            else:
                _logger.warning(f"No pre-computed std for {ch}, using 1.0.")

        # Avoid division by zero
        stds[stds <= 1e-5] = 1.0
        return means, stds

    # ------------------------------------------------------------------
    # DataReaderBase / DataReaderTimestep overrides
    # ------------------------------------------------------------------

    @override
    def init_empty(self) -> None:
        super().init_empty()
        self.ds = None
        self.len = 0
        self.n_points = 0
        self.available_vars = []

    @override
    def length(self) -> int:
        return self.len

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        """
        Get data for a time window.

        Parameters
        ----------
        idx : TIndex
            Index of temporal window
        channels_idx : list[int]
            Indices of channels to return, expressed as indices into *available_vars*
            (i.e. what base class stores as source_idx / target_idx).
        """
        self._lazy_init()

        if not channels_idx:
            return ReaderData.empty(
                num_data_fields=0,
                num_geo_fields=0,
            )

        (t_idxs, dtr) = self._get_dataset_idxs(idx)

        if self.ds is None or self.len == 0 or len(t_idxs) == 0:
            _logger.info(
                f"No valid time indices found for idx={idx}; returning empty data. "
                "(if self.ds is None or self.len == 0 or len(t_idxs) == 0:)"
            )
            return ReaderData.empty(
                num_data_fields=len(channels_idx),
                num_geo_fields=0,
            )

        # Map channel indices → variable names
        selected_channels = [self.available_vars[i] for i in channels_idx]

        time_values = self.ds.coords[self._time_coord].values
        n_time = len(time_values)

        if self.log_debug:
            _logger.info(
                f"Available vars: {self.available_vars}, requested channels: {selected_channels}"
                f"\n Fetching data for idx={idx} (dataset time range: "
                f"{time_values[0]} to {time_values[-1]}), "
                f"\n time dim: {self._time_dim}, time coord: {self._time_coord}"
                f"\n Selected time indices: {t_idxs} -- len={len(t_idxs)}"
            )

        # Full-grid coordinates, shared by every timestep before any row dropping.
        if self._curvilinear:
            # nav_lat/nav_lon are already flattened (n_points,)
            coords_grid = np.stack([self._nav_lat_flat, self._nav_lon_flat], axis=1).astype(
                np.float32
            )
            # Mask the fill points where nav_lat == 0 AND nav_lon == 0
            masked = (self._nav_lat_flat == 0.0) & (self._nav_lon_flat == 0.0)
            coords_grid[masked] = np.nan
        else:
            lon_grid, lat_grid = np.meshgrid(self.longitudes, self.latitudes)
            coords_grid = np.stack([lat_grid.flatten(), lon_grid.flatten()], axis=1).astype(
                np.float32
            )

        data_arrays: list[NDArray] = []
        coords_arrays: list[NDArray] = []
        datetimes_list: list[np.datetime64] = []

        for t_idx in t_idxs:
            if t_idx < 0 or t_idx >= n_time:
                continue

            # (n_points, n_channels)
            timestep_data = np.stack(
                [
                    self.ds[ch]
                    .isel({self._time_dim: int(t_idx)})
                    .values.astype(np.float32)
                    .flatten()
                    for ch in selected_channels
                ],
                axis=1,
            )
            timestep_coords = coords_grid

            # Drop grid points carrying no data in any selected channel, e.g. the ORCA land
            # mask. Recomputed per timestep from the loaded values rather than cached, because
            # the mask depends on selected_channels: a surface field and a deep level do not
            # share one. Dropping here rather than downstream also keeps max_num_targets
            # meaning what it says, since ReaderData.shuffle runs before the NaN-coord filter.
            if self._drop_all_nan:
                keep = ~np.isnan(timestep_data).all(axis=1)
                if not keep.all():
                    timestep_data = timestep_data[keep]
                    timestep_coords = coords_grid[keep]

            data_arrays.append(timestep_data)
            coords_arrays.append(timestep_coords)
            dt = np.datetime64(time_values[t_idx])
            # rows per timestep is not self.n_points once all-NaN rows are dropped
            datetimes_list.extend([dt] * timestep_data.shape[0])

        if not data_arrays:
            _logger.info(f"No valid time indices found for idx={idx}; returning empty data.")
            return ReaderData.empty(
                num_data_fields=len(channels_idx),
                num_geo_fields=0,
            )

        # (sum of per-timestep rows, n_channels)
        data = np.vstack(data_arrays)
        coords = np.vstack(coords_arrays)

        geoinfos = np.zeros((len(data), 0), dtype=np.float32)
        datetimes = np.array(datetimes_list, dtype="datetime64[s]")

        rd = ReaderData(
            coords=coords,
            geoinfos=geoinfos,
            data=data,
            datetimes=datetimes,
        )
        if self.log_debug:
            _logger.info(
                f"Constructed ReaderData with coords shape {coords.shape}, "
                f"geoinfos shape {geoinfos.shape}, data shape {data.shape}, "
                f"datetimes shape {datetimes.shape}"
            )
            _logger.info(
                f"  Sample coords: {coords[:5]}, sample data: {data[:5]}, "
                f"geoinfos: {geoinfos[:5]}, sample datetimes: {datetimes[:5]}"
            )
            _logger.info(f"  Channels in data: {selected_channels}")
            _logger.info(
                f"  data type: {data.dtype}, coords type: {coords.dtype}, "
                f"datetimes type: {datetimes.dtype}"
            )
            self.log_debug = False  # only log once per worker to avoid spamming

        check_reader_data(rd, dtr)

        return rd


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _detect_time_dim(ds: xr.Dataset, name: str) -> tuple[str, str]:
    """
    Identify the time dimension and the coordinate holding its values.

    The time axis is found structurally rather than by name, for two reasons. The naming is not
    stable across datasets - E-OBS uses 'time', C-GLORS v7 'time_centered', C-GLORS v8
    'time_counter' - and the dimension name and the coordinate name can differ: stock NEMO output
    has a 'time_counter' dimension carrying auxiliary 'time_centered' and 'time_instant'
    coordinates, so indexing by coordinate name would fail.

    A time axis is taken to be a dimension that has a 1-D datetime64 coordinate and is shared by
    the data variables. Candidates are ranked by how many data variables carry them, preferring a
    dimension that also has a same-named index coordinate.

    Parameters
    ----------
    ds :
        open dataset
    name :
        stream name, for error messages

    Returns
    -------
    (time dimension name, time coordinate name)
    """
    dt_coords: dict[str, list[str]] = {}
    for coord_name, coord in ds.coords.items():
        if coord.ndim == 1 and np.issubdtype(coord.dtype, np.datetime64):
            dt_coords.setdefault(str(coord.dims[0]), []).append(str(coord_name))

    if not dt_coords:
        raise ValueError(
            f"Dataset '{name}' has no 1-D datetime coordinate, so its time axis cannot be "
            f"identified. Coordinates: {list(ds.coords)}"
        )

    def _rank(dim: str) -> tuple[int, bool]:
        n_vars = sum(1 for var in ds.data_vars if dim in ds[var].dims)
        return (n_vars, dim in dt_coords[dim])

    time_dim = max(dt_coords, key=_rank)
    coord_names = dt_coords[time_dim]
    time_coord = time_dim if time_dim in coord_names else coord_names[0]

    _logger.info(f"{name}: time dimension '{time_dim}', time coordinate '{time_coord}'.")
    return time_dim, time_coord


def _str_to_timedelta(s: str) -> np.timedelta64:
    """
    Parse simple frequency strings such as '6h', '1D', '30min' into np.timedelta64.
    Supported suffixes: h, H, D, min, T, s, S.
    """
    import re

    m = re.fullmatch(r"(\d+)\s*(h|H|D|min|T|s|S)", s.strip())
    if m is None:
        raise ValueError(f"Cannot parse frequency string: {s!r}")
    value = int(m.group(1))
    unit_map = {"h": "h", "H": "h", "D": "D", "min": "m", "T": "m", "s": "s", "S": "s"}
    return np.timedelta64(value, unit_map[m.group(2)])
