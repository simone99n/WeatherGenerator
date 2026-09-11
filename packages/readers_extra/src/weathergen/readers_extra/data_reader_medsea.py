# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Data reader for the Mediterranean Sea reanalysis (NEMO, 1/24 degree).

Nine ocean variables as daily means, 18 levels down to 971 m, one Zarr store per
year from 1987 to 2021. There is no land-sea mask in the file: land is NaN.
"""

import json
import logging
from pathlib import Path
from typing import override

import numpy as np
import xarray as xr

from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    check_reader_data,
)
from weathergen.train.utils import Stage
from weathergen.utils.distributed import is_root

_logger = logging.getLogger(__name__)

# The grid is regular, but the nav_lat / nav_lon arrays stored in the file are
# zero on 41% of the cells (all of them on land), so anything that takes their
# min or max gets nonsense. We rebuild the coordinates from these four numbers
# instead. Lesson: always look at the min and max of your coordinates.
NY, NX = 380, 1016
DXY = 1.0 / 24.0
LAT0, LON0 = 30.1875, -6.0

# The 18 depths in metres. They are not evenly spaced.
DEPTH_M = [
    1.0182366, 3.1657474, 5.4649634, 7.9203773, 10.536604, 19.398211,
    29.885643, 51.379860, 72.623688, 97.928726, 153.43285, 203.17044,
    249.91585, 303.56131, 398.54471, 556.40887, 756.19604, 971.07788,
]  # fmt: skip

# Variables that exist only at the surface: sea surface height, net heat flux,
# net water flux, and the two components of the wind stress.
SURFACE_VARS = ["sossheig", "sohefldo", "sowaflup", "sozotaux", "sometauy"]

# Variables that have all 18 levels: temperature, salinity, and the two
# components of the current.
#
# Careful with the currents. The model stores them half a cell away from the
# temperature (about 1.9 km), and so it does for the wind stress above. Stacking
# them with temperature as if they were in the same place puts a small error
# into every gradient the model sees. Which way to shift them back is not yet
# confirmed for this dataset, so the example config uses only the variables that
# are already in the right place: votemper, vosaline and sossheig.
VOLUME_VARS = ["votemper", "vosaline", "vozocrtx", "vomecrty"]

# The file has three time axes. time_counter is the middle of the day that each
# average covers, which is the right label for a daily mean. time_instant is the
# same thing plus 12 hours, the end of the day. We use time_counter.
TIME_DIM = "time_counter"

# Sea cells at the surface, out of the 380 * 1016 = 386080 cells of the box.
N_SEA_POINTS = 144990

# One channel per surface variable, plus one per level for the deep ones. The
# name carries the depth in metres, rounded: votemper_1m ... votemper_971m.
CHANNELS: list[str] = []
CHANNEL_VAR: list[str] = []
CHANNEL_LEVEL: list[int] = []
for _var in SURFACE_VARS:
    CHANNELS.append(_var)
    CHANNEL_VAR.append(_var)
    CHANNEL_LEVEL.append(0)
for _var in VOLUME_VARS:
    for _level, _depth in enumerate(DEPTH_M):
        CHANNELS.append(f"{_var}_{round(_depth)}m")
        CHANNEL_VAR.append(_var)
        CHANNEL_LEVEL.append(_level)


class DataReaderMedSea(DataReaderTimestep):
    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
        stage: Stage,
    ) -> None:
        # open_zarr reads the metadata only. No data is loaded here, and one
        # reader handles one year: the framework makes one per file in the list.
        self.ds = xr.open_zarr(filename, consolidated=True, chunks=None, zarr_format=2)
        self.times = self.ds.coords[TIME_DIM].values

        data_start_time = self.times[0]
        data_end_time = self.times[-1]
        period = self.times[1] - self.times[0]

        if tw_handler.t_start >= data_end_time or tw_handler.t_end <= data_start_time:
            name = stream_info["name"]
            _logger.warning(f"{name} is not supported over data loader window. Stream is skipped.")
            super().__init__(tw_handler, stream_info)
            self.init_empty()
            return

        super().__init__(tw_handler, stream_info, data_start_time, data_end_time, period)
        self.len = len(self.times)

        lat = LAT0 + np.arange(NY, dtype=np.float32) * DXY
        lon = LON0 + np.arange(NX, dtype=np.float32) * DXY
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")

        # The land-sea mask is nowhere in the file: it is only the pattern of
        # NaN in the data. It never changes with time, so we read it once, here,
        # instead of looking for NaN in every sample.
        wet = np.isfinite(self.ds["votemper"].isel({TIME_DIM: 0}).values)
        wet = wet.reshape(len(DEPTH_M), NY * NX)

        # We return one point per sea cell. Land is simply not in the list, so
        # the model never sees it: 386080 cells become 144990 points.
        self.sea_points = np.flatnonzero(wet[0])
        self.latitudes = lat_grid.reshape(-1)[self.sea_points]
        self.longitudes = lon_grid.reshape(-1)[self.sea_points]
        self.n_points = len(self.sea_points)

        # A cell that is sea at one level is sea at every level above it, so
        # counting the sea levels of a column gives the depth of its floor. This
        # is how the model is told where the deep channels have no data.
        n_levels = wet.sum(axis=0)[self.sea_points]
        self.floor_depth = np.array(DEPTH_M, dtype=np.float32)[n_levels - 1]

        self.source_channels = stream_info.get("source", CHANNELS)
        self.source_idx = [CHANNELS.index(c) for c in self.source_channels]

        self.target_channels = stream_info.get("target", CHANNELS)
        self.target_idx = [CHANNELS.index(c) for c in self.target_channels]

        self.geoinfo_channels = ["sea_floor_depth"]
        self.geoinfo_idx = [0]

        self.target_channel_weights = self.parse_target_channel_weights()

        # Mean and standard deviation per channel, so per variable AND per
        # level: in the Mediterranean the temperature goes from 15-28 C at the
        # surface to about 13.5 C below 500 m, and one mean per variable would
        # flatten all of that. They are made once by make_medsea_norm_stats.py
        # and never computed here: the archive is 1.38 TiB.
        stats_path = Path(stream_info["norm_stats"])
        assert stats_path.exists(), (
            f"No normalisation statistics at {stats_path}. "
            f"Make them once with make_medsea_norm_stats.py."
        )
        stats = json.loads(stats_path.read_text())
        self.mean = np.array([stats[c]["mean"] for c in CHANNELS], dtype=np.float32)
        self.stdev = np.array([stats[c]["std"] for c in CHANNELS], dtype=np.float32)

        self.mean_geoinfo = np.array([self.floor_depth.mean()], dtype=np.float32)
        self.stdev_geoinfo = np.array([self.floor_depth.std()], dtype=np.float32)

        if is_root():
            name = stream_info["name"]
            if self.n_points != N_SEA_POINTS:
                _logger.warning(f"{name}: {self.n_points} sea points, expected {N_SEA_POINTS}.")
            _logger.info(f"{name}: source channels: {self.source_channels}")
            _logger.info(f"{name}: target channels: {self.target_channels}")

    @override
    def init_empty(self) -> None:
        super().init_empty()
        self.len = 0

    @override
    def length(self) -> int:
        return self.len

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        (t_idxs, dtr) = self._get_dataset_idxs(idx)

        if self.len == 0 or len(t_idxs) == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        blocks = []
        for t_idx in t_idxs:
            # Read each variable once for this day, then take the levels we
            # want out of it. One chunk on disk holds five levels, so reading
            # one level at a time would fetch the same chunk five times over.
            fields = {
                var: self.ds[var].isel({TIME_DIM: int(t_idx)}).values.reshape(-1, NY * NX)
                for var in {CHANNEL_VAR[c] for c in channels_idx}
            }
            columns = [
                fields[CHANNEL_VAR[c]][CHANNEL_LEVEL[c]][self.sea_points] for c in channels_idx
            ]
            blocks.append(np.stack(columns, axis=-1))

        data = np.vstack(blocks).astype(np.float32)

        # Under the sea floor there is no value. We write the mean of the
        # channel, which the framework turns into exactly 0 when it normalises.
        # Writing a plain 0 here instead would mean a sea temperature of 0 C.
        missing = ~np.isfinite(data)
        data[missing] = np.broadcast_to(self.mean[channels_idx], data.shape)[missing]

        latlon = np.stack([self.latitudes, self.longitudes], axis=-1)
        coords = np.vstack((latlon,) * len(t_idxs))

        geoinfos = np.vstack((self.floor_depth.reshape(-1, 1),) * len(t_idxs))
        datetimes = np.repeat(self.times[t_idxs], self.n_points)

        rd = ReaderData(coords=coords, geoinfos=geoinfos, data=data, datetimes=datetimes)
        check_reader_data(rd, dtr)

        return rd
