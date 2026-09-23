# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""DataReaderGREP against the store layouts it has to support.

The reader had no test coverage, which is how a hardcoded ``zarr_format=2`` and a
``time``/``time_centered`` guess survived in a reader that is also pointed at a zarr v3 store
whose time dimension is ``time_counter``. The three store fixtures below are the three layouts
actually in use:

  regular      zarr v2, ``time`` / ``latitude`` / ``longitude``          e.g. E-OBS
  curvilinear  zarr v3, ``time_counter`` / ``nav_lat`` / ``nav_lon``     e.g. C-GLORS v8
  nemo         zarr v2, ``time_counter`` dimension carrying auxiliary
               ``time_centered`` and ``time_instant`` coordinates       stock NEMO output

The third is why the time axis is resolved to a (dimension, coordinate) pair rather than one
name: ``time_centered`` is not a dimension there, so ``isel(time_centered=...)`` is invalid.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from weathergen.datasets.data_reader_base import TimeWindowHandler
from weathergen.readers_extra.data_reader_grep import DataReaderGREP, _detect_time_dim

N_TIME, N_Y, N_X = 4, 3, 3
N_POINTS = N_Y * N_X
CHANNELS = ["aaa", "bbb"]

T0 = np.datetime64("2020-01-01T00:00")
DAY = np.timedelta64(1, "D")


def _values(n_time=N_TIME, fill=None):
    data = np.arange(n_time * N_POINTS, dtype=np.float32).reshape(n_time, N_Y, N_X)
    if fill is not None:
        # one grid point is missing at every timestep, like an ocean land mask
        data[:, 0, 0] = fill
    return data


def _write(ds: xr.Dataset, path, zarr_format: int):
    ds.to_zarr(path, consolidated=True, zarr_format=zarr_format)
    return path


@pytest.fixture
def regular_store(tmp_path):
    """zarr v2, 1-D latitude/longitude, time dimension named 'time'."""
    times = T0 + np.arange(N_TIME) * DAY
    ds = xr.Dataset(
        {name: (("time", "latitude", "longitude"), _values()) for name in CHANNELS}
        | {f"{name}_mean": ((), np.float32(1.0)) for name in CHANNELS}
        | {f"{name}_std": ((), np.float32(2.0)) for name in CHANNELS},
        coords={
            "time": times,
            "latitude": np.linspace(40.0, 42.0, N_Y, dtype=np.float32),
            "longitude": np.linspace(8.0, 10.0, N_X, dtype=np.float32),
        },
    )
    return _write(ds, tmp_path / "regular.zarr", zarr_format=2)


@pytest.fixture
def curvilinear_store(tmp_path):
    """zarr v3, 2-D nav_lat/nav_lon, time dimension named 'time_counter'."""
    return _write(_curvilinear_dataset(fill=None), tmp_path / "curvilinear.zarr", zarr_format=3)


@pytest.fixture
def curvilinear_store_with_gaps(tmp_path):
    """As above, but one grid point is NaN in every channel at every timestep."""
    return _write(_curvilinear_dataset(fill=np.nan), tmp_path / "gaps.zarr", zarr_format=3)


def _curvilinear_dataset(fill):
    times = T0 + np.arange(N_TIME) * DAY
    lat, lon = np.meshgrid(
        np.linspace(-40.0, 40.0, N_X, dtype=np.float32),
        np.linspace(-10.0, 10.0, N_Y, dtype=np.float32),
    )
    return xr.Dataset(
        {name: (("time_counter", "y", "x"), _values(fill=fill)) for name in CHANNELS}
        | {f"{name}_mean": ((), np.float32(1.0)) for name in CHANNELS}
        | {f"{name}_std": ((), np.float32(2.0)) for name in CHANNELS},
        coords={"time_counter": times, "nav_lat": (("y", "x"), lat), "nav_lon": (("y", "x"), lon)},
    )


@pytest.fixture
def nemo_store(tmp_path):
    """Stock NEMO: the dimension is time_counter, time_centered only rides on it."""
    times = T0 + np.arange(N_TIME) * DAY
    lat, lon = np.meshgrid(
        np.linspace(-40.0, 40.0, N_X, dtype=np.float32),
        np.linspace(-10.0, 10.0, N_Y, dtype=np.float32),
    )
    ds = xr.Dataset(
        {name: (("time_counter", "y", "x"), _values()) for name in CHANNELS},
        coords={
            "time_counter": times,
            "time_centered": ("time_counter", times - np.timedelta64(12, "h")),
            "time_instant": ("time_counter", times),
            "nav_lat": (("y", "x"), lat),
            "nav_lon": (("y", "x"), lon),
        },
    )
    return _write(ds, tmp_path / "nemo.zarr", zarr_format=2)


def _handler(window_len=DAY):
    return TimeWindowHandler(T0, T0 + N_TIME * DAY, window_len, DAY)


def _reader(path, **stream_info):
    info = {"name": "TEST", "stream_id": 7, "source": CHANNELS, "target": CHANNELS}
    info.update(stream_info)
    return DataReaderGREP(tw_handler=_handler(), filename=path, stream_info=info, stage="train")


# --------------------------------------------------------------------- time axis detection


@pytest.mark.parametrize(
    ("fixture_name", "expected"),
    [
        ("regular_store", ("time", "time")),
        ("curvilinear_store", ("time_counter", "time_counter")),
        # the dimension, not the authoritative-looking auxiliary coordinate
        ("nemo_store", ("time_counter", "time_counter")),
    ],
)
def test_time_axis_is_detected_structurally(request, fixture_name, expected):
    path = request.getfixturevalue(fixture_name)
    with xr.open_zarr(path, consolidated=True, chunks=None) as ds:
        assert _detect_time_dim(ds, "TEST") == expected


def test_a_store_without_a_datetime_coordinate_is_rejected(tmp_path):
    ds = xr.Dataset(
        {"aaa": (("step", "y"), np.zeros((2, 2), dtype=np.float32))},
        coords={"step": np.arange(2), "nav_lat": ("y", np.zeros(2, dtype=np.float32))},
    )
    ds.to_zarr(tmp_path / "notime.zarr", consolidated=True, zarr_format=3)

    with xr.open_zarr(tmp_path / "notime.zarr", consolidated=True, chunks=None) as opened:
        with pytest.raises(ValueError, match="no 1-D datetime coordinate"):
            _detect_time_dim(opened, "TEST")


# ----------------------------------------------------------------------------- both layouts


@pytest.mark.parametrize(
    ("fixture_name", "curvilinear"),
    [("regular_store", False), ("curvilinear_store", True), ("nemo_store", True)],
)
def test_both_grid_layouts_read_a_full_window(request, fixture_name, curvilinear):
    reader = _reader(request.getfixturevalue(fixture_name))

    assert reader._curvilinear is curvilinear
    assert reader.source_channels == CHANNELS
    assert reader.target_channels == CHANNELS
    assert reader.n_points == N_POINTS
    assert reader.period == np.timedelta64(DAY, "ms")

    rdata = reader.get_source(0)
    assert rdata.data.shape == (N_POINTS, len(CHANNELS))
    assert rdata.coords.shape == (N_POINTS, 2)
    assert rdata.datetimes.shape == (N_POINTS,)
    # check_reader_data already asserted this inside _get; make the contract explicit
    assert (rdata.datetimes == T0.astype("datetime64[s]")).all()


def test_statistics_are_read_for_the_selected_channels(curvilinear_store):
    reader = _reader(curvilinear_store)

    assert list(reader.mean[reader.source_idx]) == [1.0, 1.0]
    assert list(reader.stdev[reader.source_idx]) == [2.0, 2.0]


def test_a_multi_timestep_window_stacks_timesteps(curvilinear_store):
    """Two daily records inside one window: rows and coordinates both tile."""
    info = {"name": "TEST", "stream_id": 7, "source": CHANNELS, "target": CHANNELS}
    reader = DataReaderGREP(
        tw_handler=TimeWindowHandler(T0, T0 + N_TIME * DAY, 2 * DAY, 2 * DAY),
        filename=curvilinear_store,
        stream_info=info,
        stage="train",
    )

    rdata = reader.get_source(0)
    assert rdata.data.shape == (2 * N_POINTS, len(CHANNELS))
    # the per-timestep coordinate blocks are identical, which is what np.tile used to assume
    np.testing.assert_array_equal(rdata.coords[:N_POINTS], rdata.coords[N_POINTS:])
    assert len(np.unique(rdata.datetimes)) == 2


# --------------------------------------------------------------------------- loud failures


def test_a_missing_store_raises_rather_than_reading_nothing(tmp_path):
    """A stream that silently reads nothing is substituted with spoofed data downstream."""
    with pytest.raises(Exception):  # noqa: B017 - zarr/xarray raise several types here
        _reader(tmp_path / "does_not_exist.zarr")


def test_filters_that_match_nothing_are_rejected(curvilinear_store):
    with pytest.raises(ValueError, match="no source and no target channels"):
        _reader(curvilinear_store, source=["nope"], target=["also_nope"])


def test_one_empty_side_is_allowed(curvilinear_store):
    """A source-only or target-only stream is legitimate; both empty is not."""
    reader = _reader(curvilinear_store, source=CHANNELS, target=["nope"])

    assert reader.source_channels == CHANNELS
    assert reader.target_channels == []


# ------------------------------------------------------------------------ all-NaN row drop


def test_all_nan_rows_are_kept_by_default(curvilinear_store_with_gaps):
    reader = _reader(curvilinear_store_with_gaps)
    rdata = reader.get_source(0)

    assert rdata.data.shape == (N_POINTS, len(CHANNELS))
    assert np.isnan(rdata.data).any()


def test_all_nan_rows_are_dropped_when_requested(curvilinear_store_with_gaps):
    reader = _reader(curvilinear_store_with_gaps, drop_all_nan_rows=True)
    rdata = reader.get_source(0)

    assert rdata.data.shape == (N_POINTS - 1, len(CHANNELS))
    assert not np.isnan(rdata.data).any()
    # coords, data and datetimes stay aligned once the row count is no longer n_points
    assert rdata.coords.shape[0] == rdata.data.shape[0] == rdata.datetimes.shape[0]


def test_dropping_rows_keeps_the_surviving_coordinates(curvilinear_store_with_gaps):
    kept = _reader(curvilinear_store_with_gaps, drop_all_nan_rows=True).get_source(0)
    full = _reader(curvilinear_store_with_gaps).get_source(0)

    # the dropped point is the first one, so the rest must match position for position
    np.testing.assert_array_equal(kept.coords, full.coords[1:])
