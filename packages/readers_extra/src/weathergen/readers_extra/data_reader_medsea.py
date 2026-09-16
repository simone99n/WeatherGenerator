# (C) Copyright 2026 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Data reader for the Mediterranean Sea reanalysis (NEMO, 1/24 degree).

Nine ocean variables, 18 levels down to 971 m, one Zarr store per year from 1987
to 2021 (must change). The time step is whatever the file says it is,
daily means in this archive, so 3-hourly or multi-day output reads the same way.
There is no land-sea mask in the file: land is NaN.
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

# The grid is regular
# nav_lat and nav_lon can be contain the sea-over-land mask
NY, NX = 380, 1016
DXY = 1.0 / 24.0
LAT0, LON0 = 30.1875, -6.0

DEPTH_M = [
    1.0182366, 3.1657474, 5.4649634, 7.9203773, 10.536604, 19.398211,
    29.885643, 51.379860, 72.623688, 97.928726, 153.43285, 203.17044,
    249.91585, 303.56131, 398.54471, 556.40887, 756.19604, 971.07788,
]  # fmt: skip

SURFACE_VARS = ["sossheig", "sohefldo", "sowaflup", "sozotaux", "sometauy"]

# Careful with the currents. The model stores them half a cell away from the
# temperature (about 1.9 km), and so it does for the wind stress above. Stacking
# them with temperature as if they were in the same place puts a small error
# into every gradient the model sees.
VOLUME_VARS = ["votemper", "vosaline", "vozocrtx", "vomecrty"]

# time_counter is the centre of the interval that each average covers,
#   which is the right label for a mean over that interval.
# time_instant is the end of the same interval, half a time step later: 12 hours
#   for the daily means of this archive.
TIME_DIM = "time_counter"

# Sea cells at the surface, out of the 380 * 1016 = 386080 cells of the box.
N_SEA_POINTS = 144990

# Every line below that changes the shape of an array says what the new shape
# is, using these names:
#
#   rows, cols   the grid, NY by NX
#   cells        the grid flattened, rows * cols, sea and land together
#   levels       the 18 depths
#   points       the sea cells only, which is all this reader returns
#   steps        the time steps of the window being asked for
#   channels     the channels being asked for
#   geoinfos     the geoinfo channels, one column each

# The depth as it is written in a channel name, one per level.
DEPTH_NAMES = [round(depth) for depth in DEPTH_M]

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
    for _level, _depth in enumerate(DEPTH_NAMES):
        CHANNELS.append(f"{_var}_{_depth}m")
        CHANNEL_VAR.append(_var)
        CHANNEL_LEVEL.append(_level)


def _channel_idxs(names: list[str], field: str, stream_name: str) -> list[int]:
    """
    Turn channel names from the stream config into indices into CHANNELS.

    CHANNELS is the same for every store of the archive, so a name that is not
    in it is a typo in the config rather than a store that happens to lack the
    field. Say so here, and say which names do exist, instead of leaving
    list.index to raise "'x' is not in list" with no stream and no list.
    """
    unknown = [c for c in names if c not in CHANNELS]
    assert not unknown, (
        f"{stream_name}: {field} lists channel(s) {unknown} that this dataset does not have. "
        f"Available: {', '.join(SURFACE_VARS)}, and <var>_<depth>m for each of "
        f"{', '.join(VOLUME_VARS)} with <depth> one of "
        f"{', '.join(str(depth) for depth in DEPTH_NAMES)}."
    )

    return [CHANNELS.index(c) for c in names]


class DataReaderMedSea(DataReaderTimestep):
    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
        stage: Stage,
    ) -> None:
        # *** STEP 1: OPEN DATASET METADATA *** #
        # open_zarr reads the metadata only. No data is loaded here
        self.ds = xr.open_zarr(filename, consolidated=True, chunks=None, zarr_format=2)

        # *** STEP 2: TIME COORDINATE HANDLING *** #
        self.times = self.ds.coords[TIME_DIM].values  # (steps in this store,)
        data_start_time = self.times[0]
        data_end_time = self.times[-1]
        period = self.times[1] - self.times[0]  # time step in the dataset

        if tw_handler.t_start >= data_end_time or tw_handler.t_end <= data_start_time:
            name = stream_info["name"]
            _logger.warning(f"{name} is not supported over data loader window. Stream is skipped.")
            super().__init__(tw_handler, stream_info)
            self.init_empty()
            return

        super().__init__(tw_handler, stream_info, data_start_time, data_end_time, period)
        self.len = len(self.times)

        # *** STEP 3: LAT-LON GRID *** #
        lat = LAT0 + np.arange(NY, dtype=np.float32) * DXY  # (rows,)
        lon = LON0 + np.arange(NX, dtype=np.float32) * DXY  # (cols,)
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")  # both (rows, cols)

        # *** STEP 4: LAND MASK *** #
        # It never changes with time, so we read it once, here, instead of
        # looking for NaN in every sample.
        # NaN --> Land mask
        wet = np.isfinite(self.ds["votemper"].isel({TIME_DIM: 0}).values)  # (levels, rows, cols)
        wet = wet.reshape(len(DEPTH_M), NY * NX)  # (levels, cells)

        # *** STEP 5: SEA POINTS AND MASKED COORDINATES *** #
        # We return one point per sea cell. Land is simply not in the list, so
        # the model never sees it: 386080 cells become 144990 points.
        self.sea_points = np.flatnonzero(wet[0])  # (points,), an index into cells
        self.latitudes = lat_grid.reshape(-1)[self.sea_points]  # (points,)
        self.longitudes = lon_grid.reshape(-1)[self.sea_points]  # (points,)
        self.n_points = len(self.sea_points)

        # *** STEP 6: CHANNEL and GEOINFOS INDEXING *** #
        name = stream_info["name"]

        self.source_channels = stream_info.get("source", CHANNELS)
        self.target_channels = stream_info.get("target", CHANNELS)
        self.geoinfo_channels = stream_info.get("geoinfo_channels", [])

        self.source_idx = _channel_idxs(self.source_channels, "source", name)
        self.target_idx = _channel_idxs(self.target_channels, "target", name)
        self.geoinfo_idx = _channel_idxs(self.geoinfo_channels, "geoinfo_channels", name)

        # *** STEP 7: CHANNEL WEIGHTS *** #
        self.target_channel_weights = self.parse_target_channel_weights()

        # *** STEP 8: NORMALISATION STATISTICS *** #
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
        # Both (all channels,): the framework indexes them with source_idx and
        # target_idx, which point into the full channel list.
        self.mean = np.array([stats[c]["mean"] for c in CHANNELS], dtype=np.float32)
        self.stdev = np.array([stats[c]["std"] for c in CHANNELS], dtype=np.float32)

        # Both (geoinfos,), already the subset the config asked for:
        # normalize_geoinfos indexes these by position, while mean and stdev
        # above are indexed by channel.
        self.mean_geoinfo = self.mean[self.geoinfo_idx]
        self.stdev_geoinfo = self.stdev[self.geoinfo_idx]

        # *** STEP 9: LOGGER OUTPUT *** #
        if is_root():
            if self.n_points != N_SEA_POINTS:
                _logger.warning(f"{name}: {self.n_points} sea points, expected {N_SEA_POINTS}.")
            _logger.info(f"{name}: source channels: {self.source_channels}")
            _logger.info(f"{name}: target channels: {self.target_channels}")
            _logger.info(f"{name}: geoinfo channels: {self.geoinfo_channels}")

    @override
    def init_empty(self) -> None:
        super().init_empty()
        self.len = 0

    @override
    def length(self) -> int:
        return self.len

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        # *** STEP 11: GET DATASET INDEXES FOR THIS TIME WINDOW *** #
        (t_idxs, dtr) = self._get_dataset_idxs(idx)

        if self.len == 0 or len(t_idxs) == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=len(self.geoinfo_idx)
            )

        # ** STEP 12: READ DATA FROM DISK *** #
        blocks = []
        geoinfo_blocks = []
        for t_idx in t_idxs:
            # Read each variable once for this time step, then take the levels
            # we want out of it. One chunk on disk holds five levels, so reading
            # one level at a time would fetch the same chunk five times over.
            # Each value is (levels, cells), or (1, cells) for a surface one.
            # The geoinfo channels are read here too, so that a variable needed
            # by both is fetched once, but they never share a column with the
            # data: they are stacked on their own, below.
            fields = {
                var: self.ds[var].isel({TIME_DIM: int(t_idx)}).values.reshape(-1, NY * NX)
                for var in {CHANNEL_VAR[c] for c in [*channels_idx, *self.geoinfo_idx]}
            }
            # Pick the variable, then the level, then keep the sea cells only.
            columns = [  # a list of (points,), one per channel
                fields[CHANNEL_VAR[c]][CHANNEL_LEVEL[c]][self.sea_points] for c in channels_idx
            ]
            blocks.append(np.stack(columns, axis=-1))  # (points, channels)

            if self.geoinfo_idx:
                geoinfo_columns = [  # a list of (points,), one per geoinfo
                    fields[CHANNEL_VAR[c]][CHANNEL_LEVEL[c]][self.sea_points]
                    for c in self.geoinfo_idx
                ]
                geoinfo_blocks.append(np.stack(geoinfo_columns, axis=-1))  # (points, geoinfos)

        data = np.vstack(blocks).astype(np.float32)  # (steps * points, channels)

        # (steps * points, geoinfos), with no column at all when the config asked
        # for no geoinfo, which is what most of the other readers return.
        geoinfos = (
            np.vstack(geoinfo_blocks).astype(np.float32)
            if geoinfo_blocks
            else np.zeros((len(data), 0), dtype=np.float32)
        )

        # Under the sea floor there is no value. We write the mean of the
        # channel, which the framework turns into exactly 0 when it normalises.
        # Writing a plain 0 here instead would mean a sea temperature of 0 C.
        missing = ~np.isfinite(data)  # (steps * points, channels)
        data[missing] = np.broadcast_to(self.mean[channels_idx], data.shape)[missing]

        # The same for the geoinfos, where it also keeps the point: the
        # framework drops every point whose geoinfos are not all finite, so a
        # deep channel used as a geoinfo would delete the sea floor in silence.
        missing = ~np.isfinite(geoinfos)  # (steps * points, geoinfos)
        geoinfos[missing] = np.broadcast_to(self.mean[self.geoinfo_idx], geoinfos.shape)[missing]
        # NOTE sto rimpiendo con la media ma magari voglio scartare il punto
        # (remove_nan_coords_and_geoinfos)

        # The same points come back at every time step, so the coordinates are
        # simply repeated once per step.
        latlon = np.stack([self.latitudes, self.longitudes], axis=-1)  # (points, 2)
        coords = np.vstack((latlon,) * len(t_idxs))  # (steps * points, 2)

        datetimes = np.repeat(self.times[t_idxs], self.n_points)  # (steps * points,)

        rd = ReaderData(coords=coords, geoinfos=geoinfos, data=data, datetimes=datetimes)
        check_reader_data(rd, dtr)

        return rd
