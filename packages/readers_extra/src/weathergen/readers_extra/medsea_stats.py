# (C) Copyright 2026 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Compute the normalisation statistics of the Mediterranean Sea reanalysis.

    python -m weathergen.readers_extra.medsea_stats \\
        /e/scratch/weatherai/norberti1/reanalysis-2021_v2.zarr \\
        --out /e/scratch/weatherai/norberti1/medsea_stats_2021.npz

Every time step of every store given is read, one step at a time, and no store
is ever written to: the output is an .npz that medsea_write_stats.py turns into
the mean/ and std/ groups inside the store.

Give every store of the archive in one run. Statistics per year would normalise
the same value differently depending on which year the sampler drew it from.

The statistics are per variable AND per level: in the Mediterranean the
temperature goes from 15-28 C at the surface to about 13.5 C below 500 m, so one
number per variable would flatten the deep signal. Land and the cells below the
sea floor are NaN in the file and are left out, which makes each level's mean an
average over its own sea cells, unweighted by cell area.

The .npz holds five arrays per variable, shaped (18,) for the volume variables
and () for the surface ones:

    mean_<var>  std_<var>  count_<var>  min_<var>  max_<var>

plus depth_deptht / depth_depthu / depth_depthv, and the provenance entries
stores, time_first, time_last and n_time_steps.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
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

_logger = logging.getLogger(__name__)

# The depth coordinate each volume variable is labelled with. The currents live
# on the staggered U and V grids, so they carry their own coordinate even though
# the three are numerically equal in this store.
DEPTH_DIM = {
    "votemper": "deptht",
    "vosaline": "deptht",
    "vozocrtx": "depthu",
    "vomecrty": "depthv",
}

DEPTH_DIMS = ["deptht", "depthu", "depthv"]

# Surface variables first, so the order matches CHANNELS in the reader.
VARS = [*SURFACE_VARS, *VOLUME_VARS]

# What each statistic is called in the .npz and in the store groups.
STATISTICS = ["mean", "std", "count", "min", "max"]


def _n_rows(var: str) -> int:
    """Levels a variable has: one for a surface field, 18 for a volume one."""
    return len(DEPTH_M) if var in VOLUME_VARS else 1


def compute_stats(
    stores: list[Path], max_steps: int | None = None
) -> dict[str, dict[str, NDArray]]:
    """
    Read the stores and return one dict of statistics per variable.

    Parameters
    ----------
    stores :
        Zarr stores to read, all of them in one call.
    max_steps :
        Read at most this many time steps per store, for a quick look. None
        reads every step, which is what the real statistics need.

    Returns
    -------
    {variable: {"mean", "std", "count", "min", "max", "depth"}}, each entry
    shaped (18,) for a volume variable and () for a surface one, except "depth"
    which is (18,) or empty.
    """

    rows = {var: _n_rows(var) for var in VARS}
    count = {var: np.zeros(rows[var], dtype=np.int64) for var in VARS}
    total = {var: np.zeros(rows[var], dtype=np.float64) for var in VARS}
    total_sq = {var: np.zeros(rows[var], dtype=np.float64) for var in VARS}
    lowest = {var: np.full(rows[var], np.inf, dtype=np.float64) for var in VARS}
    highest = {var: np.full(rows[var], -np.inf, dtype=np.float64) for var in VARS}

    depths: dict[str, NDArray] = {}

    for store in stores:
        ds = xr.open_zarr(store, consolidated=True, chunks=None, zarr_format=2)
        times = ds.coords[TIME_DIM].values
        n_steps = len(times) if max_steps is None else min(len(times), max_steps)
        _logger.info(f"{store.name}: reading {n_steps} of {len(times)} time steps")

        for dim in DEPTH_DIMS:
            depths[dim] = ds.coords[dim].values.astype(np.float64)

        for t_idx in range(n_steps):
            for var in VARS:
                # One time step of one variable: (levels, cells), or (1, cells)
                # for a surface field. The store is chunked with one step per
                # chunk, so a step at a time is already aligned with the disk.
                field = ds[var].isel({TIME_DIM: t_idx}).values.reshape(-1, NY * NX)

                for level in range(field.shape[0]):
                    values = field[level]
                    # Land and everything below the sea floor is NaN. Dropping
                    # it here is what makes this a mean over the sea cells.
                    values = values[np.isfinite(values)]
                    if values.size == 0:
                        continue

                    # float64 throughout: summing 5e7 float32 values would lose
                    # the low bits long before the end of the year.
                    count[var][level] += values.size
                    total[var][level] += np.sum(values, dtype=np.float64)
                    total_sq[var][level] += np.sum(np.square(values, dtype=np.float64))
                    lowest[var][level] = min(lowest[var][level], values.min())
                    highest[var][level] = max(highest[var][level], values.max())

            if (t_idx + 1) % 30 == 0 or t_idx + 1 == n_steps:
                _logger.info(f"  {store.name}: {t_idx + 1}/{n_steps}")

    stats: dict[str, dict[str, NDArray]] = {}
    for var in VARS:
        counts = count[var]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(counts > 0, total[var] / counts, np.nan)
            variance = np.where(counts > 0, total_sq[var] / counts - np.square(mean), np.nan)

        for level in range(len(counts)):
            if counts[level] == 0:
                _logger.warning(f"{var} level {level}: no finite value, mean and std are NaN.")
            elif variance[level] <= 0.0:
                # Said out loud rather than hidden by a clip: a channel that
                # does not vary cannot be normalised, and the reader divides by
                # this number without a guard.
                _logger.warning(
                    f"{var} level {level}: variance is {variance[level]:.3e}, not positive. "
                    f"std is 0, so this channel cannot be normalised."
                )

        stat = {
            "mean": mean,
            "std": np.sqrt(np.clip(variance, 0.0, None)),
            "count": counts,
            "min": np.where(counts > 0, lowest[var], np.nan),
            "max": np.where(counts > 0, highest[var], np.nan),
            "depth": depths[DEPTH_DIM[var]] if var in VOLUME_VARS else np.zeros(0),
        }
        if var in SURFACE_VARS:
            # A surface variable has one statistic, not a column of one: 0-d
            # here and 0-d in the store, so the reader can tell them apart.
            stat = {k: v.reshape(()) if k != "depth" else v for k, v in stat.items()}

        stats[var] = stat

    return stats


def provenance(stores: list[Path], max_steps: int | None = None) -> dict[str, NDArray]:
    """Which stores were read and which time range they cover, metadata only."""

    first, last, n_steps = None, None, 0
    for store in stores:
        ds = xr.open_zarr(store, consolidated=True, chunks=None, zarr_format=2)
        times = ds.coords[TIME_DIM].values
        n = len(times) if max_steps is None else min(len(times), max_steps)
        first = times[0] if first is None else min(first, times[0])
        last = times[n - 1] if last is None else max(last, times[n - 1])
        n_steps += n

    return {
        "stores": np.array([str(store) for store in stores]),
        "time_first": np.array(str(first)),
        "time_last": np.array(str(last)),
        "n_time_steps": np.array(n_steps),
    }


def _log_table(stats: dict[str, dict[str, NDArray]]) -> None:
    """One line per variable and level, so the numbers can be eyeballed."""

    _logger.info(
        f"{'variable':10} {'level':>5} {'depth[m]':>9} {'count':>12} "
        f"{'mean':>14} {'std':>12} {'min':>12} {'max':>12}"
    )
    for var, stat in stats.items():
        mean, std = np.atleast_1d(stat["mean"]), np.atleast_1d(stat["std"])
        counts = np.atleast_1d(stat["count"])
        low, high = np.atleast_1d(stat["min"]), np.atleast_1d(stat["max"])
        depth = stat["depth"]

        for level in range(len(mean)):
            depth_str = f"{depth[level]:9.2f}" if len(depth) else f"{'-':>9}"
            level_str = f"{level:5}" if len(depth) else f"{'-':>5}"
            _logger.info(
                f"{var:10} {level_str} {depth_str} {counts[level]:12} "
                f"{mean[level]:14.6f} {std[level]:12.6f} {low[level]:12.4f} {high[level]:12.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stores", type=Path, nargs="+", help="zarr stores to read")
    parser.add_argument("--out", type=Path, default=None, help="write the statistics to this .npz")
    parser.add_argument(
        "--max-steps", type=int, default=None, help="read at most this many steps per store"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    for store in args.stores:
        assert store.exists(), f"No store at {store}."
    if args.max_steps is not None:
        _logger.warning(
            f"--max-steps {args.max_steps}: a quick look, not the statistics of the archive."
        )

    stats = compute_stats(args.stores, max_steps=args.max_steps)
    _log_table(stats)

    if args.out is None:
        _logger.info("No --out given, nothing written.")
        return

    arrays = {f"{key}_{var}": stats[var][key] for var in VARS for key in STATISTICS}
    for dim in DEPTH_DIMS:
        depth = next(stats[var]["depth"] for var in VOLUME_VARS if DEPTH_DIM[var] == dim)
        arrays[f"depth_{dim}"] = depth
    arrays |= provenance(args.stores, max_steps=args.max_steps)

    np.savez(args.out, **arrays)
    _logger.info(f"Wrote {len(arrays)} arrays to {args.out}")


if __name__ == "__main__":
    main()
