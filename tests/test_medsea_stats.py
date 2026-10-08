# (C) Copyright 2026 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Tests for the medsea statistics scripts, without the 12 GB of the real store.

Two synthetic stores with the shape of the real one, full grid but a couple of
time steps: one whose statistics are known exactly, one filled with noise to be
compared against numpy. The tests run wherever the reader's own tests run.
"""

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from numpy.typing import NDArray

from weathergen.readers_extra.data_reader_medsea import (
    DEPTH_M,
    NX,
    NY,
    SURFACE_VARS,
    TIME_DIM,
    VOLUME_VARS,
)
from weathergen.readers_extra.medsea_stats import DEPTH_DIM, VARS, as_npz, compute_stats, provenance
from weathergen.readers_extra.medsea_write_stats import GROUPS, write_stats

N_LEVELS = len(DEPTH_M)

# The constant every level of a volume variable is filled with, and the one
# value every surface variable takes, in the analytic store.
LEVEL_STEP = 10.0
SURFACE_VALUE = 0.25


def _mask() -> NDArray[np.bool_]:
    """Land on the western strip, and a sea floor that shallows towards the north."""

    floor = np.linspace(N_LEVELS - 1, 0, NY, dtype=int)
    wet = np.arange(N_LEVELS)[:, None, None] <= floor[None, :, None]
    land = np.zeros((NY, NX), dtype=bool)
    land[:, :10] = True

    return wet & ~land[None]


def _write_store(path: Path, n_steps: int, filler: Callable) -> NDArray[np.bool_]:
    """A store shaped like the real one, with land and the sea floor as NaN."""

    wet = _mask()
    data_vars = {}
    for var in SURFACE_VARS:
        field = filler((n_steps, NY, NX))
        field[:, ~wet[0]] = np.nan
        data_vars[var] = ((TIME_DIM, "y", "x"), field)
    for var in VOLUME_VARS:
        field = filler((n_steps, N_LEVELS, NY, NX))
        field[:, ~wet] = np.nan
        data_vars[var] = ((TIME_DIM, DEPTH_DIM[var], "y", "x"), field)

    depths = np.array(DEPTH_M, dtype=np.float32)
    times = np.datetime64("2021-01-01T12:00") + np.arange(n_steps) * np.timedelta64(1, "D")
    xr.Dataset(
        data_vars,
        coords={TIME_DIM: times, "deptht": depths, "depthu": depths, "depthv": depths},
    ).to_zarr(path, consolidated=True, zarr_format=2)

    return wet


def _constant(shape: tuple[int, ...]) -> NDArray[np.float32]:
    """Level L filled with 10 * L, or SURFACE_VALUE for a surface field."""

    if len(shape) == 4:
        levels = LEVEL_STEP * np.arange(N_LEVELS, dtype=np.float32)
        return np.broadcast_to(levels[None, :, None, None], shape).copy()

    return np.full(shape, SURFACE_VALUE, dtype=np.float32)


@pytest.fixture(scope="module")
def analytic_store(tmp_path_factory):
    """Every level constant, so mean and deviation are known exactly."""

    path = tmp_path_factory.mktemp("analytic") / "reanalysis-2021.zarr"
    _write_store(path, 1, _constant)

    return path


@pytest.fixture(scope="module")
def noisy_store(tmp_path_factory):
    """Two steps of noise, to be compared against numpy over the same cells."""

    path = tmp_path_factory.mktemp("noisy") / "reanalysis-2021.zarr"
    rng = np.random.default_rng(0)
    wet = _write_store(path, 2, lambda shape: rng.normal(size=shape).astype(np.float32))

    return path, wet


def test_constant_levels_give_exact_statistics(analytic_store):
    stats = compute_stats([analytic_store])

    # Exact, not approximate: the values are representable and the accumulation
    # is in float64, so there is nothing to round.
    np.testing.assert_array_equal(
        stats["votemper"]["mean"], LEVEL_STEP * np.arange(N_LEVELS, dtype=np.float64)
    )
    np.testing.assert_array_equal(stats["votemper"]["std"], np.zeros(N_LEVELS))

    # A surface variable keeps one statistic, not a column of one.
    assert stats["sossheig"]["mean"].shape == ()
    assert float(stats["sossheig"]["mean"]) == SURFACE_VALUE
    assert float(stats["sossheig"]["std"]) == 0.0


def test_a_channel_that_does_not_vary_is_declared(analytic_store, caplog):
    """The old script hid this behind np.maximum(var, 0). It must be said out loud."""

    with caplog.at_level(logging.WARNING):
        compute_stats([analytic_store])

    declared = [record.message for record in caplog.records if "not positive" in record.message]
    assert len(declared) == len(SURFACE_VARS) + len(VOLUME_VARS) * N_LEVELS
    assert any("votemper level 17" in message for message in declared)


def test_noise_matches_numpy(noisy_store):
    path, _wet = noisy_store
    stats = compute_stats([path])

    ds = xr.open_zarr(path, consolidated=True, chunks=None, zarr_format=2)
    field = ds["votemper"].values  # (steps, levels, y, x)

    for level in (0, 7, N_LEVELS - 1):
        values = field[:, level]
        assert stats["votemper"]["count"][level] == int(np.isfinite(values).sum())
        assert stats["votemper"]["mean"][level] == pytest.approx(
            float(np.nanmean(values, dtype=np.float64)), rel=1e-12
        )
        # nanstd is the population deviation, which is what the script computes.
        assert stats["votemper"]["std"][level] == pytest.approx(
            float(np.nanstd(values, dtype=np.float64)), rel=1e-6
        )

    surface = ds["sossheig"].values
    assert stats["sossheig"]["mean"] == pytest.approx(
        float(np.nanmean(surface, dtype=np.float64)), rel=1e-12
    )


def test_deeper_levels_have_fewer_cells(noisy_store):
    path, wet = noisy_store
    stats = compute_stats([path])

    counts = stats["votemper"]["count"]
    assert counts[0] == 2 * int(wet[0].sum())
    assert counts[N_LEVELS - 1] < counts[0], "the nested mask must lose cells with depth"


def test_max_steps_stops_early(noisy_store):
    path, _wet = noisy_store

    one = compute_stats([path], max_steps=1)
    both = compute_stats([path])

    assert 2 * one["votemper"]["count"][0] == both["votemper"]["count"][0]
    assert int(provenance([path], max_steps=1)["n_time_steps"]) == 1
    assert int(provenance([path])["n_time_steps"]) == 2


def test_writing_does_not_change_the_root(analytic_store):
    """The regression test of the whole design: the reader's dataset must not move."""

    def signature() -> tuple:
        ds = xr.open_zarr(analytic_store, consolidated=True, chunks=None, zarr_format=2)
        return sorted(ds.data_vars), sorted(ds.coords), sorted(ds.attrs)

    before = signature()
    write_stats(analytic_store, as_npz(compute_stats([analytic_store]), [analytic_store]), True)

    assert signature() == before
    assert len(before[0]) == len(VARS)


def test_a_zero_statistic_reads_back_as_zero(analytic_store):
    """With the zarr v2 default fill_value of 0, xarray would mask these as NaN."""

    write_stats(analytic_store, as_npz(compute_stats([analytic_store]), [analytic_store]), True)

    stds = xr.open_zarr(analytic_store, group="std", consolidated=True, chunks=None, zarr_format=2)
    assert np.isfinite(stds["votemper"].values).all()
    assert float(stds["votemper"].values[0]) == 0.0
    assert float(stds["sossheig"].values) == 0.0


def test_the_groups_carry_counts_and_units(analytic_store):
    npz = as_npz(compute_stats([analytic_store]), [analytic_store])
    write_stats(analytic_store, npz, True)

    means = xr.open_zarr(
        analytic_store, group="mean", consolidated=True, chunks=None, zarr_format=2
    )
    assert means["votemper"].attrs["count"] == npz["count_votemper"].tolist()
    assert means["votemper"].attrs["units"] == "degC"
    assert means.attrs["statistic"] == "mean"
    assert means.attrs["n_time_steps"] == 1
    np.testing.assert_allclose(means["deptht"].values, DEPTH_M)


def test_writing_twice_needs_force(analytic_store):
    npz = as_npz(compute_stats([analytic_store]), [analytic_store])
    write_stats(analytic_store, npz, True)

    with pytest.raises(AssertionError, match="--force"):
        write_stats(analytic_store, npz, force=False)

    # And with the flag it goes through, so a climatology can be replaced.
    write_stats(analytic_store, npz, force=True)
    for group in GROUPS:
        ds = xr.open_zarr(analytic_store, group=group, consolidated=True, zarr_format=2)
        assert len(ds.data_vars) == len(VARS)
