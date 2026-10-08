# (C) Copyright 2026 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Write the statistics computed by medsea_stats.py into the store itself.

    python -m weathergen.readers_extra.medsea_write_stats \\
        /e/scratch/weatherai/norberti1/reanalysis-2021_v2.zarr \\
        --from-npz /e/scratch/weatherai/norberti1/medsea_stats_2021.npz --dry-run

Two groups, mean/ and std/, are added next to the nine variables, each holding
one array per variable plus the three depth coordinates. They are subgroups, so
the root dataset the reader opens does not change at all: that invariant is
recorded before writing and checked again afterwards.

Several stores can be given: the same climatology belongs in every store of an
archive, since statistics per year would normalise the same value differently
depending on the year the sampler drew it from.

What is written is tiny, about 1.5 KB of data, and the only existing files
touched are the three root metadata files, which consolidation rewrites. They
are copied first to <store>.metadata-backup-<timestamp>/, next to the store and
never inside it: a foreign file inside a store makes zarr warn about an
unrecognised component of the hierarchy.
"""

import argparse
import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import xarray as xr
import zarr
from numpy.typing import NDArray

from weathergen.readers_extra.data_reader_medsea import VOLUME_VARS
from weathergen.readers_extra.medsea_stats import DEPTH_DIM, DEPTH_DIMS, VARS

_logger = logging.getLogger(__name__)

# The variables carry no units attribute in the store: votemper and vosaline
# have nothing in their .zattrs beyond what xarray consumes. These are declared
# here rather than copied, and the group says so, so that nobody mistakes them
# for something the archive stated.
UNITS = {
    "votemper": "degC",
    "vosaline": "psu",
    "sossheig": "m",
    "sohefldo": "W m-2",
    "sowaflup": "kg m-2 s-1",
    "sozotaux": "N m-2",
    "sometauy": "N m-2",
    "vozocrtx": "m s-1",
    "vomecrty": "m s-1",
}

GROUPS = ["mean", "std"]

# Rewritten by consolidation, so backed up first.
ROOT_METADATA = [".zmetadata", ".zattrs", ".zgroup"]

_MEAN_NOTE = (
    "Unweighted mean over the sea cells of each level: land and the cells below the sea "
    "floor are NaN in the store and were left out. Cell area is not accounted for, so at "
    "1/24 degree a cell at 45.98 N weighs the same as one at 30.19 N while covering about "
    "80% of its area."
)


def _root_signature(store: Path) -> tuple[list[str], list[str], list[str]]:
    """What the reader sees at the root: the invariant this script must not break."""

    ds = xr.open_zarr(store, consolidated=True, chunks=None, zarr_format=2)
    return sorted(ds.data_vars), sorted(ds.coords), sorted(ds.attrs)


def _array_plan(npz: dict[str, NDArray], group: str) -> list[tuple[str, NDArray, list[str], str]]:
    """The arrays of one group: name, values, _ARRAY_DIMENSIONS, units."""

    plan = [(dim, npz[f"depth_{dim}"], [dim], "m") for dim in DEPTH_DIMS]
    for var in VARS:
        dims = [DEPTH_DIM[var]] if var in VOLUME_VARS else []
        plan.append((var, npz[f"{group}_{var}"], dims, UNITS[var]))

    return plan


def _group_attrs(group: str, npz: dict[str, NDArray]) -> dict:
    return {
        "statistic": group,
        "source_stores": [str(s) for s in np.atleast_1d(npz["stores"])],
        "time_range": [str(npz["time_first"]), str(npz["time_last"])],
        "n_time_steps": int(npz["n_time_steps"]),
        "created_by": "weathergen.readers_extra.medsea_write_stats",
        "created_on": datetime.now(UTC).isoformat(timespec="seconds"),
        "note": _MEAN_NOTE,
        "units_note": "units are declared by this script; the store's variables carry none",
    }


def _log_plan(npz: dict[str, NDArray]) -> None:
    """The tree that would be written, with shapes, dtypes and bytes."""

    total, n_arrays = 0, 0
    for group in GROUPS:
        _logger.info(f"{group}/")
        for name, values, dims, units in _array_plan(npz, group):
            total += values.nbytes
            n_arrays += 1
            dim_str = f"[{', '.join(dims)}]" if dims else "[] (0-d)"
            _logger.info(
                f"    {name:10} shape={str(values.shape):6} {values.dtype} "
                f"{dim_str:12} units={units:12} {values.nbytes:5} B"
            )
    _logger.info(f"{total} bytes of data in {n_arrays} arrays")


def _backup_root_metadata(store: Path) -> Path:
    """Copy the three files consolidation rewrites, outside the store."""

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M")
    backup = store.parent / f"{store.name}.metadata-backup-{stamp}"
    backup.mkdir(parents=True, exist_ok=True)
    for name in ROOT_METADATA:
        src = store / name
        if src.exists():
            shutil.copy2(src, backup / name)
    _logger.info(f"Root metadata backed up to {backup}")

    return backup


def write_stats(store: Path, npz: dict[str, NDArray], force: bool = False) -> None:
    """
    Add the mean/ and std/ groups to one store, then check the root is untouched.

    Parameters
    ----------
    store :
        The zarr store to write into.
    npz :
        Arrays as medsea_stats.py wrote them.
    force :
        Overwrite groups that are already there.
    """

    before = _root_signature(store)

    root = zarr.open_group(store, mode="a", zarr_format=2)
    existing = [group for group in GROUPS if group in root]
    assert not existing or force, (
        f"{store} already has {existing}. Pass --force to overwrite, or remove them with "
        f"rm -rf {' '.join(str(store / group) for group in existing)}"
    )

    _backup_root_metadata(store)

    for group in GROUPS:
        handle = root.require_group(group)
        for name, values, dims, units in _array_plan(npz, group):
            # fill_value must be NaN, not the default 0: xarray reads the v2
            # fill_value as _FillValue and masks every value equal to it, so a
            # statistic of exactly 0.0 would come back as NaN. A standard
            # deviation of 0 is a plausible value, so the loss would be silent.
            array = handle.create_array(
                name,
                shape=values.shape,
                dtype="f8",
                fill_value=np.nan,
                overwrite=force,
            )
            array[...] = values
            array.attrs["_ARRAY_DIMENSIONS"] = dims
            array.attrs["units"] = units
            if name in VARS:
                # How many finite values went into each level, so that the
                # statistic stays verifiable without a third group.
                array.attrs["count"] = npz[f"count_{name}"].tolist()

        handle.attrs.update(_group_attrs(group, npz))

    zarr.consolidate_metadata(store, zarr_format=2)

    _verify(store, npz, before)


def _verify(store: Path, npz: dict[str, NDArray], before: tuple) -> None:
    """Re-read the store in this same process and check nothing else moved."""

    after = _root_signature(store)
    assert after == before, (
        f"The root changed: data_vars/coords/attrs were {before} and are now {after}."
    )
    _logger.info(
        f"Root unchanged: {len(before[0])} data_vars, {len(before[1])} coords, "
        f"{len(before[2])} attributes"
    )

    for group in GROUPS:
        ds = xr.open_zarr(store, group=group, consolidated=True, chunks=None, zarr_format=2)
        for var in VARS:
            assert var in ds, f"{group}/{var} is missing after writing."
            values = ds[var].values
            expected = npz[f"{group}_{var}"]
            assert values.shape == expected.shape, (
                f"{group}/{var}: shape {values.shape} instead of {expected.shape}."
            )
            counts = np.atleast_1d(npz[f"count_{var}"])
            finite = np.isfinite(np.atleast_1d(values))
            assert finite[counts > 0].all(), (
                f"{group}/{var}: NaN read back where count > 0. If the statistic is exactly "
                f"0.0, check that the array was created with fill_value=NaN."
            )
        _logger.info(f"{group}/: {len(VARS)} statistics re-read, no NaN where count > 0")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stores", type=Path, nargs="+", help="zarr stores to write into")
    parser.add_argument(
        "--from-npz", type=Path, required=True, help="statistics from medsea_stats.py"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the tree that would be written, write nothing"
    )
    parser.add_argument("--force", action="store_true", help="overwrite groups already there")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    assert args.from_npz.exists(), f"No statistics at {args.from_npz}."
    for store in args.stores:
        assert store.exists(), f"No store at {store}."

    with np.load(args.from_npz, allow_pickle=False) as loaded:
        npz = dict(loaded)

    wanted = [f"{group}_{var}" for group in [*GROUPS, "count"] for var in VARS]
    missing = [key for key in wanted if key not in npz]
    assert not missing, f"{args.from_npz} is missing {missing[:5]}. Re-run medsea_stats.py."

    _logger.info(
        f"Statistics from {npz['n_time_steps']} time steps, "
        f"{npz['time_first']} to {npz['time_last']}"
    )
    _log_plan(npz)

    if args.dry_run:
        _logger.info("--dry-run: nothing written.")
        return

    for store in args.stores:
        _logger.info(f"Writing into {store}")
        write_stats(store, npz, force=args.force)
        _logger.info(f"{store}: done, {json.dumps(sorted(GROUPS))} written and verified")


if __name__ == "__main__":
    main()
