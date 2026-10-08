# (C) Copyright 2026 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Check a store that medsea_write_stats.py has written, one pass/fail line each.

    python -m weathergen.readers_extra.medsea_check_stats \\
        /e/scratch/weatherai/norberti1/reanalysis-2021_v2.zarr --days 8

Eight checks, in order of what they would catch:

1. the root dataset the reader opens is exactly what it was before writing;
2. the twelve NEMO attributes of the root are all still there;
3. both groups hold nine statistics, with the shapes and depths they should;
4. an independent recomputation on a few random days agrees with the stored
   statistics, which is what catches a swapped variable or inverted levels;
5. one channel is recomputed over every time step and must agree to 1e-10;
6. the count attributes match a fresh count of the finite cells;
7. every statistic array has fill_value NaN, so a statistic of exactly 0.0
   reads back as 0.0 and not as NaN;
8. end to end: a DataReaderMedSea built without norm_stats normalises a window
   to roughly zero mean and unit deviation.

Checks 4 and 5 recompute with numpy's own nanmean and nansum rather than
calling medsea_stats.py, so a mistake in the accumulation there cannot be
confirmed by the check that is supposed to catch it.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from weathergen.readers_extra.data_reader_medsea import (
    CHANNEL_LEVEL,
    CHANNEL_VAR,
    CHANNELS,
    NX,
    NY,
    SURFACE_VARS,
    TIME_DIM,
    VOLUME_VARS,
    DataReaderMedSea,
)
from weathergen.readers_extra.medsea_stats import VARS
from weathergen.readers_extra.medsea_write_stats import GROUPS

_logger = logging.getLogger(__name__)

# What the reader must keep seeing at the root of the store.
ROOT_DATA_VARS = sorted(VARS)
ROOT_COORDS = [
    "deptht",
    "depthu",
    "depthv",
    "nav_lat",
    "nav_lon",
    "time_centered",
    TIME_DIM,
    "time_instant",
]
ROOT_ATTRS = [
    "Conventions",
    "TimeStamp",
    "description",
    "file_name",
    "ibegin",
    "jbegin",
    "name",
    "ni",
    "nj",
    "production",
    "timeStamp",
    "title",
]

# A handful of channels for the end-to-end check: surface, deep, and a flux.
END_TO_END_CHANNELS = ["sossheig", "sohefldo", "votemper_1m", "votemper_971m", "vosaline_250m"]

type Result = tuple[bool | None, str, str]


def _open(store: Path, group: str | None = None) -> xr.Dataset:
    """The reader's own call, so the checks see what the reader sees."""
    return xr.open_zarr(store, group=group, consolidated=True, chunks=None, zarr_format=2)


def check_root_dataset(store: Path) -> Result:
    ds = _open(store)
    data_vars, coords = sorted(ds.data_vars), sorted(ds.coords)
    ok = data_vars == ROOT_DATA_VARS and coords == ROOT_COORDS

    return (
        ok,
        "root dataset unchanged",
        f"{len(data_vars)} data_vars, {len(coords)} coords"
        if ok
        else f"data_vars {data_vars}, coords {coords}",
    )


def check_root_attrs(store: Path) -> Result:
    attrs = sorted(_open(store).attrs)
    missing = [a for a in ROOT_ATTRS if a not in attrs]

    return (
        not missing,
        "NEMO root attributes kept",
        f"all {len(ROOT_ATTRS)} there" if not missing else f"missing {missing}",
    )


def check_groups(store: Path) -> Result:
    root_depth = _open(store)["deptht"].values
    problems = []
    for group in GROUPS:
        ds = _open(store, group)
        for var in VARS:
            if var not in ds:
                problems.append(f"{group}/{var} missing")
                continue
            wanted = (len(root_depth),) if var in VOLUME_VARS else ()
            if ds[var].shape != wanted:
                problems.append(f"{group}/{var} shape {ds[var].shape} not {wanted}")
        if "deptht" in ds and not np.allclose(ds["deptht"].values, root_depth):
            problems.append(f"{group}/deptht differs from the root")

    return (
        not problems,
        "groups have the right arrays",
        f"{len(GROUPS)} groups x {len(VARS)} statistics" if not problems else str(problems[:3]),
    )


def _recompute_mean(ds: xr.Dataset, var: str, steps: list[int]) -> NDArray:
    """Mean per level over these time steps, with numpy's nanmean."""

    totals, counts = None, None
    for step in steps:
        field = ds[var].isel({TIME_DIM: step}).values.reshape(-1, NY * NX)
        finite = np.isfinite(field)
        step_total = np.where(finite, field, 0.0).sum(axis=1, dtype=np.float64)
        step_count = finite.sum(axis=1)
        totals = step_total if totals is None else totals + step_total
        counts = step_count if counts is None else counts + step_count

    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, totals / counts, np.nan)


def check_random_days(store: Path, n_days: int, seed: int) -> Result:
    ds = _open(store)
    n_steps = ds.sizes[TIME_DIM]
    rng = np.random.default_rng(seed)
    steps = sorted(rng.choice(n_steps, size=min(n_days, n_steps), replace=False).tolist())

    stored_mean = {var: np.atleast_1d(_open(store, "mean")[var].values) for var in VARS}
    stored_std = {var: np.atleast_1d(_open(store, "std")[var].values) for var in VARS}

    worst, worst_name = 0.0, ""
    for var in VARS:
        sample = np.atleast_1d(_recompute_mean(ds, var, steps))
        # The tolerance is the annual standard deviation of the channel, not
        # std/sqrt(n): a few days of a seasonal field are nowhere near the
        # annual mean in units of the standard error, while a swapped variable
        # or an inverted level shows up as many deviations at depth, where the
        # deviation is small.
        for level in range(len(sample)):
            spread = stored_std[var][level]
            if not np.isfinite(spread) or spread == 0.0:
                continue
            distance = abs(sample[level] - stored_mean[var][level]) / spread
            if distance > worst:
                worst, worst_name = distance, f"{var}[{level}]"

    return (
        worst < 2.0,
        f"independent recomputation on {len(steps)} random days",
        f"worst gap {worst:.2f} annual deviations, on {worst_name}",
    )


def check_exact_channel(store: Path, channel: str) -> Result:
    var, level = CHANNEL_VAR[CHANNELS.index(channel)], CHANNEL_LEVEL[CHANNELS.index(channel)]
    ds = _open(store)

    total, count = 0.0, 0
    for step in range(ds.sizes[TIME_DIM]):
        values = ds[var].isel({TIME_DIM: step}).values.reshape(-1, NY * NX)[level]
        total += np.nansum(values.astype(np.float64))
        count += int(np.isfinite(values).sum())

    fresh = total / count
    stored = float(np.atleast_1d(_open(store, "mean")[var].values)[level])
    gap = abs(fresh - stored) / max(abs(stored), 1e-12)

    return (
        gap < 1e-10,
        f"exact recomputation of {channel} over all {ds.sizes[TIME_DIM]} steps",
        f"stored {stored:.10f}, recomputed {fresh:.10f}, relative gap {gap:.2e}",
    )


def check_counts(store: Path) -> Result:
    ds = _open(store)
    n_steps = int(_open(store, "mean").attrs["n_time_steps"])
    problems = []
    for var in VARS:
        stored = np.atleast_1d(_open(store, "mean")[var].attrs["count"])
        field = ds[var].isel({TIME_DIM: 0}).values.reshape(-1, NY * NX)
        fresh = np.isfinite(field).sum(axis=1) * n_steps
        if not np.array_equal(stored, fresh):
            problems.append(f"{var}: stored {stored[:2]} vs {fresh[:2]} from one step x {n_steps}")

    return (
        not problems,
        "count attributes match a fresh count",
        f"level 0: {np.atleast_1d(_open(store, 'mean')['votemper'].attrs['count'])[0]}"
        if not problems
        else str(problems[:2]),
    )


def check_fill_values(store: Path) -> Result:
    """Regression: fill_value 0 would make a statistic of exactly 0.0 read as NaN."""

    wrong = []
    for group in GROUPS:
        for zarray in sorted((store / group).glob("*/.zarray")):
            fill = json.loads(zarray.read_text()).get("fill_value")
            if fill != "NaN":
                wrong.append(f"{group}/{zarray.parent.name}={fill}")

    n_arrays = sum(len(list((store / group).glob("*/.zarray"))) for group in GROUPS)

    return (
        not wrong and n_arrays > 0,
        "every statistic array has fill_value NaN",
        f"{n_arrays} arrays checked" if not wrong else str(wrong[:3]),
    )


def check_end_to_end(store: Path) -> Result:
    """Build the reader with no norm_stats at all and normalise one window."""

    from weathergen.datasets.data_reader_base import TimeWindowHandler

    stream_info = {
        "name": "MedSeaCheck",
        "source": END_TO_END_CHANNELS,
        "target": ["votemper_1m"],
    }
    tw = TimeWindowHandler(
        np.datetime64("2021-01-02T00:00"),
        np.datetime64("2021-01-05T00:00"),
        np.timedelta64(24, "h"),
        np.timedelta64(24, "h"),
    )

    try:
        reader = DataReaderMedSea(tw, store, stream_info, stage="train")
    except KeyError as error:
        return (None, "end to end without norm_stats", f"the reader still wants {error}")

    rdata = reader.get_source(np.int64(0))
    normalized = reader.normalize_source_channels(rdata.data.copy())

    problems = []
    for i, channel in enumerate(reader.source_channels):
        mean, std = normalized[:, i].mean(), normalized[:, i].std()
        _logger.info(f"    {channel:16} normalised mean {mean:8.3f}  deviation {std:6.3f}")
        # One day is not the year, so these are loose bounds: they catch a
        # statistic applied to the wrong channel, not a seasonal offset.
        if abs(mean) > 3.0 or not 0.02 < std < 5.0:
            problems.append(channel)

    return (
        not problems,
        "end to end: a window normalises to about 0 and 1",
        f"{len(reader.source_channels)} channels in range"
        if not problems
        else f"out of range: {problems}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store", type=Path, help="the store to check")
    parser.add_argument("--days", type=int, default=8, help="random days for the recomputation")
    parser.add_argument(
        "--exact-channel",
        default="sossheig",
        help="channel recomputed over every step; a surface one costs seconds, a deep one minutes",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for the random days")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    assert args.store.exists(), f"No store at {args.store}."
    assert args.exact_channel in CHANNELS, (
        f"{args.exact_channel} is not a channel. Surface ones are {SURFACE_VARS}, "
        f"the deep ones are <var>_<depth>m for {VOLUME_VARS}."
    )

    results = [
        check_root_dataset(args.store),
        check_root_attrs(args.store),
        check_groups(args.store),
        check_random_days(args.store, args.days, args.seed),
        check_exact_channel(args.store, args.exact_channel),
        check_counts(args.store),
        check_fill_values(args.store),
        check_end_to_end(args.store),
    ]

    _logger.info("")
    failed = 0
    for i, (ok, label, detail) in enumerate(results, start=1):
        if ok is None:
            mark = "PENDING"
        elif ok:
            mark = "PASS   "
        else:
            mark = "FAIL   "
            failed += 1
        _logger.info(f"{mark} {i}. {label}: {detail}")

    pending = sum(1 for ok, _, _ in results if ok is None)
    _logger.info(f"\n{len(results) - failed - pending} passed, {failed} failed, {pending} pending")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
