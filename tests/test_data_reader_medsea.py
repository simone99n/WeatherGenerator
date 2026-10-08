# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Tests for the Mediterranean reanalysis reader.

The real archive is 1.38 TiB and lives on the CMCC machines, so these tests
build a tiny Zarr store with the same structure: the same variable names, three
identical depth coordinates, time_counter at noon, land as NaN with a nested
mask, and the mean and std groups the reader takes its statistics from.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from numpy.typing import NDArray

from weathergen.datasets.data_reader_base import TimeWindowHandler
from weathergen.readers_extra.data_reader_medsea import (
    CHANNEL_LEVEL,
    CHANNEL_VAR,
    CHANNELS,
    DEPTH_M,
    NX,
    NY,
    SURFACE_VARS,
    TIME_DIM,
    VOLUME_VARS,
    DataReaderMedSea,
)
from weathergen.readers_extra.medsea_write_stats import write_stats

N_DAYS = 4
N_LEVELS = len(DEPTH_M)
DEPTH_DIM = {
    "votemper": "deptht",
    "vosaline": "deptht",
    "vozocrtx": "depthu",
    "vomecrty": "depthv",
}


def _make_mask() -> NDArray[np.bool_]:
    """A nested mask: the sea floor gets shallower towards the north."""
    floor_index = np.linspace(N_LEVELS - 1, 0, NY, dtype=int)
    floor_index = np.repeat(floor_index[:, None], NX, axis=1)  # (y, x)
    land = np.zeros((NY, NX), dtype=bool)
    land[:, :10] = True  # a strip of permanent land on the western edge

    levels = np.arange(N_LEVELS)[:, None, None]
    return (levels <= floor_index[None]) & ~land[None]  # (level, y, x)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """Write a synthetic store plus matching statistics, return their paths."""
    path = tmp_path_factory.mktemp("medsea") / "reanalysis-1987.zarr"
    wet = _make_mask()
    rng = np.random.default_rng(0)

    # time_counter sits at noon, like the real daily means.
    times = np.datetime64("1987-01-01T12:00") + np.arange(N_DAYS) * np.timedelta64(1, "D")

    data_vars = {}
    for var in SURFACE_VARS:
        field = rng.normal(size=(N_DAYS, NY, NX)).astype(np.float32)
        field[:, ~wet[0]] = np.nan
        data_vars[var] = ((TIME_DIM, "y", "x"), field)

    for var in VOLUME_VARS:
        field = rng.normal(size=(N_DAYS, N_LEVELS, NY, NX)).astype(np.float32)
        # Offset each level, so that per-level statistics matter.
        field += np.arange(N_LEVELS, dtype=np.float32)[None, :, None, None] * 10.0
        field[:, ~wet] = np.nan
        data_vars[var] = ((TIME_DIM, DEPTH_DIM[var], "y", "x"), field)

    depths = np.array(DEPTH_M, dtype=np.float32)
    ds = xr.Dataset(
        data_vars,
        coords={
            TIME_DIM: times,
            "deptht": depths,
            "depthu": depths,
            "depthv": depths,
            # Zeroed in the real store: the reader must rebuild lat/lon itself.
            "nav_lat": (("y", "x"), np.zeros((NY, NX), dtype=np.float32)),
            "nav_lon": (("y", "x"), np.zeros((NY, NX), dtype=np.float32)),
        },
    )
    ds.to_zarr(path, consolidated=True, zarr_format=2)

    # The statistics of this store are known exactly: every level was offset by
    # 10 * level and the noise has unit deviation. They go in through the real
    # writer, so the test exercises that code and not a copy of it. The mean of
    # a surface variable is exactly 0.0, which is also the regression against
    # fill_value 0 reading a zero statistic back as NaN.
    write_stats(path, _analytic_stats(path), force=True)

    return path, wet


def _analytic_stats(store: Path) -> dict[str, NDArray]:
    """The .npz that medsea_stats.py would write for this synthetic store."""

    stats: dict[str, NDArray] = {}
    for var in [*SURFACE_VARS, *VOLUME_VARS]:
        levels = N_LEVELS if var in VOLUME_VARS else 1
        shape = (levels,) if var in VOLUME_VARS else ()
        stats[f"mean_{var}"] = (10.0 * np.arange(levels)).reshape(shape)
        stats[f"std_{var}"] = np.ones(levels).reshape(shape)
        stats[f"count_{var}"] = np.full(levels, 1000, dtype=np.int64).reshape(shape)
    for dim in ("deptht", "depthu", "depthv"):
        stats[f"depth_{dim}"] = np.array(DEPTH_M, dtype=np.float64)

    return stats | {
        "stores": np.array([str(store)]),
        "time_first": np.array("1987-01-01T12:00:00"),
        "time_last": np.array("1987-01-04T12:00:00"),
        "n_time_steps": np.array(N_DAYS),
    }


@pytest.fixture(scope="module")
def store_without_stats(tmp_path_factory):
    """A store the writer never touched: votemper only, which is all __init__ reads."""

    path = tmp_path_factory.mktemp("medsea_bare") / "reanalysis-1987.zarr"
    times = np.datetime64("1987-01-01T12:00") + np.arange(2) * np.timedelta64(1, "D")
    field = np.zeros((2, N_LEVELS, NY, NX), dtype=np.float32)

    xr.Dataset(
        {"votemper": ((TIME_DIM, "deptht", "y", "x"), field)},
        coords={TIME_DIM: times, "deptht": np.array(DEPTH_M, dtype=np.float32)},
    ).to_zarr(path, consolidated=True, zarr_format=2)

    return path


def _reader(
    store,
    source=None,
    target=None,
    geoinfo_channels=None,
    window_hours=24,
    t_start="1987-01-02T00:00",
):
    path, _wet = store
    stream_info = {
        "name": "MedSeaTest",
        "source": source if source is not None else ["sossheig", "votemper_1m", "votemper_971m"],
        "target": target if target is not None else ["votemper_1m"],
    }
    # Left out of stream_info entirely unless asked for, so that the tests also
    # cover the default: a config that says nothing about geoinfos.
    if geoinfo_channels is not None:
        stream_info["geoinfo_channels"] = geoinfo_channels
    tw = TimeWindowHandler(
        np.datetime64(t_start),
        np.datetime64("1987-01-05T00:00"),
        np.timedelta64(window_hours, "h"),
        np.timedelta64(24, "h"),
    )
    return DataReaderMedSea(tw, path, stream_info, stage="train")


def test_channel_list():
    assert len(CHANNELS) == len(SURFACE_VARS) + len(VOLUME_VARS) * N_LEVELS == 77
    assert len(set(CHANNELS)) == len(CHANNELS), "channel names must be unique"
    assert CHANNELS[:5] == SURFACE_VARS

    # The name of a channel must match the variable and level it points at.
    i = CHANNELS.index("votemper_971m")
    assert CHANNEL_VAR[i] == "votemper"
    assert CHANNEL_LEVEL[i] == N_LEVELS - 1

    j = CHANNELS.index("sossheig")
    assert CHANNEL_VAR[j] == "sossheig"
    assert CHANNEL_LEVEL[j] == 0


def test_coordinates_are_rebuilt_not_read(store):
    """The stored nav_lat/nav_lon are all zeros here, so they must not be used."""
    reader = _reader(store)

    assert reader.latitudes.min() >= 30.1875
    assert reader.latitudes.max() <= 45.98
    assert reader.longitudes.min() >= -6.0
    assert reader.longitudes.max() <= 36.30
    assert (reader.latitudes != 0.0).all()


def test_only_sea_points_are_returned(store):
    _path, wet = store
    reader = _reader(store)

    assert reader.n_points == int(wet[0].sum())
    assert reader.n_points < NY * NX


def test_get_source_shapes_and_time(store):
    reader = _reader(store)

    rdata = reader.get_source(np.int64(0))

    assert rdata.data.shape == (reader.n_points, 3)
    assert rdata.coords.shape == (reader.n_points, 2)
    assert rdata.geoinfos.shape == (reader.n_points, 0)
    # The window starting 1987-01-02T00:00 covers the field labelled at noon.
    assert (rdata.datetimes == np.datetime64("1987-01-02T12:00")).all()


def test_values_match_the_store(store):
    path, wet = store
    reader = _reader(store)

    rdata = reader.get_source(np.int64(0))

    ds = xr.open_zarr(path, consolidated=True, chunks=None, zarr_format=2)
    # Window index 0 is 1987-01-02, i.e. day 1 of the store.
    expected = ds["votemper"].isel({TIME_DIM: 1, "deptht": 0}).values.reshape(-1)
    expected = expected[np.flatnonzero(wet[0])]

    column = reader.source_channels.index("votemper_1m")
    np.testing.assert_allclose(rdata.data[:, column], expected, rtol=1e-6)


def test_below_the_sea_floor_normalises_to_zero(store):
    _path, wet = store
    reader = _reader(store)

    rdata = reader.get_source(np.int64(0))

    # No NaN ever reaches the model.
    assert np.isfinite(rdata.data).all()

    normalized = reader.normalize_source_channels(rdata.data.copy())
    deep = reader.source_channels.index("votemper_971m")
    # Columns whose deepest level is land: the last level has no value there.
    dry = ~wet.reshape(N_LEVELS, -1)[-1][reader.sea_points]

    assert dry.any(), "the synthetic mask must have columns shallower than the last level"
    np.testing.assert_allclose(normalized[dry, deep], 0.0, atol=1e-5)
    assert np.abs(normalized[~dry, deep]).max() > 1e-3, "sea points must keep their signal"


def test_multi_day_window_stacks_days(store):
    reader = _reader(store, source=["sossheig"], target=["sossheig"], window_hours=48)

    rdata = reader.get_source(np.int64(0))

    assert rdata.data.shape == (2 * reader.n_points, 1)
    assert len(np.unique(rdata.datetimes)) == 2
    # The same points come back each day, so coordinates simply repeat.
    np.testing.assert_allclose(rdata.coords[: reader.n_points], rdata.coords[reader.n_points :])


def test_geoinfos_default_to_none(store):
    reader = _reader(store)

    assert reader.geoinfo_channels == []
    assert reader.get_geoinfo_size() == 0
    assert reader.get_source(np.int64(0)).geoinfos.shape == (reader.n_points, 0)


def test_geoinfos_are_read_from_the_dataset(store):
    path, wet = store
    reader = _reader(store, source=["votemper_1m"], geoinfo_channels=["sossheig"])

    rdata = reader.get_source(np.int64(0))

    assert rdata.geoinfos.shape == (reader.n_points, 1)

    ds = xr.open_zarr(path, consolidated=True, chunks=None, zarr_format=2)
    # Window index 0 is 1987-01-02, i.e. day 1 of the store.
    expected = ds["sossheig"].isel({TIME_DIM: 1}).values.reshape(-1)[np.flatnonzero(wet[0])]
    np.testing.assert_allclose(rdata.geoinfos[:, 0], expected, rtol=1e-6)

    # The statistics come from the same file as the ones for the data channels.
    sossheig = CHANNELS.index("sossheig")
    np.testing.assert_allclose(reader.mean_geoinfo, [reader.mean[sossheig]])
    np.testing.assert_allclose(reader.stdev_geoinfo, [reader.stdev[sossheig]])


def test_a_channel_can_be_data_and_geoinfo_at_once(store):
    reader = _reader(store, source=["sossheig", "votemper_1m"], geoinfo_channels=["sossheig"])

    rdata = reader.get_source(np.int64(0))

    column = reader.source_channels.index("sossheig")
    np.testing.assert_allclose(rdata.geoinfos[:, 0], rdata.data[:, column])


def test_a_deep_geoinfo_keeps_every_point(store):
    """A NaN geoinfo makes the framework drop the point, so geoinfos are filled too."""
    reader = _reader(store, geoinfo_channels=["votemper_971m"])

    rdata = reader.get_source(np.int64(0))

    assert np.isfinite(rdata.geoinfos).all()
    assert rdata.remove_nan_coords_and_geoinfos().len() == reader.n_points


def test_unknown_geoinfo_channel_is_refused(store):
    with pytest.raises(AssertionError, match="mixed_layer_depth") as excinfo:
        _reader(store, geoinfo_channels=["mixed_layer_depth"])

    # The message must name the field it came from and the channels that exist.
    assert "geoinfo_channels" in str(excinfo.value)
    assert "votemper" in str(excinfo.value)


def test_window_outside_the_store_is_empty(store):
    path, _wet = store
    stream_info = {
        "name": "MedSeaTest",
        "source": ["sossheig"],
        "target": ["sossheig"],
    }
    tw = TimeWindowHandler(
        np.datetime64("1990-01-01T00:00"),
        np.datetime64("1990-01-05T00:00"),
        np.timedelta64(24, "h"),
        np.timedelta64(24, "h"),
    )
    reader = DataReaderMedSea(tw, path, stream_info, stage="train")

    assert reader.length() == 0
    assert reader.get_source(np.int64(0)).is_empty()


def test_a_store_without_statistics_says_how_to_make_them(store_without_stats):
    stream_info = {"name": "MedSeaTest", "source": ["votemper_1m"], "target": ["votemper_1m"]}
    tw = TimeWindowHandler(
        np.datetime64("1987-01-02T00:00"),
        np.datetime64("1987-01-05T00:00"),
        np.timedelta64(24, "h"),
        np.timedelta64(24, "h"),
    )

    with pytest.raises(AssertionError, match="medsea_write_stats"):
        DataReaderMedSea(tw, store_without_stats, stream_info, stage="train")


def test_statistics_come_from_the_store_groups(store):
    path, _wet = store
    reader = _reader(store)

    means = xr.open_zarr(path, group="mean", consolidated=True, chunks=None, zarr_format=2)

    # A level of a volume variable, and the 0-d statistic of a surface one:
    # np.atleast_1d in the reader is what makes the second one work.
    deep = CHANNELS.index("votemper_971m")
    assert reader.mean[deep] == np.float32(means["votemper"].values[N_LEVELS - 1])
    assert reader.mean[CHANNELS.index("sossheig")] == np.float32(means["sossheig"].values)
    # The synthetic store was built with every level offset by 10 * level.
    assert reader.mean[deep] == np.float32(10.0 * (N_LEVELS - 1))


def test_a_channel_with_no_variation_is_refused(store):
    path, _wet = store
    flat = _analytic_stats(path)
    flat["std_votemper"] = np.zeros(N_LEVELS)
    write_stats(path, flat, force=True)

    try:
        # _normalize divides without a guard, so this must not reach training.
        with pytest.raises(AssertionError, match="standard deviation of 0"):
            _reader(store, source=["votemper_1m"], target=["votemper_1m"])
    finally:
        write_stats(path, _analytic_stats(path), force=True)


def test_registry_exposes_the_reader():
    from weathergen.readers_extra.registry import get_extra_reader

    assert get_extra_reader("medsea") is DataReaderMedSea
