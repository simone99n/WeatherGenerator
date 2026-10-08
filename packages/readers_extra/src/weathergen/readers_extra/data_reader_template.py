# (C) Copyright 2026 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Template for a data reader. Copy it to data_reader_<yourdata>.py and fill in every TODO.

It reads the Zarr store written by prepare_data.py (its name must end in .zarr), which has
- a regular grid, with 1-D latitude and longitude axes;
- a fixed time step, with standard dates;
- data variables with the axes (time, lat, lon), in any order;
- optionally, geoinfo variables that do not change in time, with the axes (lat, lon);
- the mean and standard deviation of every variable, in the groups "mean" and "std".

data_reader_medsea.py next to this file is a complete reader built in the same way.
"""

import numpy as np
import xarray as xr

from weathergen.datasets.data_reader_base import DataReaderTimestep, ReaderData, check_reader_data

# TODO: the names of the time, latitude and longitude axes in your file.
TIME, LAT, LON = "time", "lat", "lon"

# TODO: the variables of your file that can be channels, i.e. that have the axes (time, lat, lon).
CHANNELS = ["t2m", "sst"]


class DataReaderTemplate(DataReaderTimestep):  # TODO: rename it, e.g. DataReaderMyData
    # The four argument names are fixed: the framework passes them by name.
    def __init__(self, tw_handler, filename, stream_info, stage):
        # *** STEP 1: OPEN THE FILE *** #
        # This reads the description of the file only. The data are read later, in _get.
        self.ds = xr.open_dataset(filename)

        # *** STEP 2: THE TIMES IN THE FILE *** #
        self.times = self.ds[TIME].values  # (times,)
        period = self.times[1] - self.times[0]  # the time step of the data
        assert np.issubdtype(self.times.dtype, np.datetime64), f"{TIME}: not standard dates"
        assert (np.diff(self.times) == period).all(), f"{TIME}: the time step is not constant"
        # "+ period" so that the last time of the file can be read too.
        super().__init__(tw_handler, stream_info, self.times[0], self.times[-1] + period, period)

        # *** STEP 3: ONE POINT PER GRID CELL *** #
        lat, lon = np.meshgrid(self.ds[LAT].values, self.ds[LON].values, indexing="ij")
        lon = (lon + 180) % 360 - 180  # longitudes 0..360 become -180..180
        latlon = np.stack([lat.reshape(-1), lon.reshape(-1)], axis=1)  # (points, 2)
        self.latlon = latlon.astype(np.float32)

        # *** STEP 4: THE CHANNELS ASKED FOR IN THE STREAM CONFIG (DEFAULT: ALL) *** #
        self.source_channels = stream_info.get("source", CHANNELS)
        self.target_channels = stream_info.get("target", CHANNELS)
        # Their positions in CHANNELS. A name that is not in CHANNELS stops here.
        self.source_idx = [CHANNELS.index(c) for c in self.source_channels]
        self.target_idx = [CHANNELS.index(c) for c in self.target_channels]
        self.target_channel_weights = self.parse_target_channel_weights()

        # *** STEP 5: NORMALISATION STATISTICS, READ FROM THE FILE *** #
        mean = xr.open_dataset(filename, group="mean")
        std = xr.open_dataset(filename, group="std")
        # One value per channel, for ALL the channels and in the order of CHANNELS.
        self.mean = np.array([float(mean[c]) for c in CHANNELS], dtype=np.float32)
        self.stdev = np.array([float(std[c]) for c in CHANNELS], dtype=np.float32)
        ok = np.isfinite(self.stdev) & (self.stdev > 0)
        assert ok.all(), f"std must be finite and > 0: {CHANNELS} {self.stdev}"

        # *** STEP 6: GEOINFOS, FIELDS THAT DO NOT CHANGE IN TIME (OPTIONAL) *** #
        self.geoinfo_channels = stream_info.get("geoinfo_channels", [])  # e.g. ["orography"]
        self.geoinfo_idx = list(range(len(self.geoinfo_channels)))  # one number per geoinfo
        self.mean_geoinfo = np.array([float(mean[g]) for g in self.geoinfo_channels])
        self.stdev_geoinfo = np.array([float(std[g]) for g in self.geoinfo_channels])
        # Read once: (points, geoinfos). A missing value becomes the mean: with a missing
        # geoinfo, the framework would drop the whole point.
        self.geoinfos = np.zeros((len(self.latlon), len(self.geoinfo_channels)), dtype=np.float32)
        for i, g in enumerate(self.geoinfo_channels):
            field = self.ds[g].transpose(LAT, LON).values.reshape(-1)  # (points,)
            self.geoinfos[:, i] = np.where(np.isfinite(field), field, self.mean_geoinfo[i])

    def length(self):  # required by the base class
        return len(self.times)

    def _get(self, idx, channels_idx):
        # *** STEP 7: THE TIMES OF THE FILE INSIDE TIME WINDOW idx *** #
        t_idxs, window = self._get_dataset_idxs(idx)
        if len(t_idxs) == 0:
            return ReaderData.empty(len(channels_idx), len(self.geoinfo_idx))

        # *** STEP 8: READ THE DATA, ONE COLUMN PER CHANNEL *** #
        n_rows = len(t_idxs) * len(self.latlon)  # times * points
        data = np.zeros((n_rows, len(channels_idx)), dtype=np.float32)
        for i, c in enumerate(channels_idx):
            field = self.ds[CHANNELS[c]].isel({TIME: t_idxs}).transpose(TIME, LAT, LON)
            field = field.values.reshape(-1)  # (times * points,)
            # A missing value (NaN, or inf) stays NaN: the framework gives the model 0 there
            # and leaves that point out of the loss.
            data[:, i] = np.where(np.isfinite(field), field, np.nan)

        # *** STEP 9: THE SAME POINTS AND GEOINFOS AT EVERY TIME OF THE WINDOW *** #
        coords = np.tile(self.latlon, (len(t_idxs), 1))  # (times * points, 2)
        geoinfos = np.tile(self.geoinfos, (len(t_idxs), 1))  # (times * points, geoinfos)
        datetimes = np.repeat(self.times[t_idxs], len(self.latlon))  # (times * points,)

        # *** STEP 10: PACK AND CHECK *** #
        rd = ReaderData(coords=coords, geoinfos=geoinfos, data=data, datetimes=datetimes)
        check_reader_data(rd, window)  # stops with a message if something does not fit
        return rd
