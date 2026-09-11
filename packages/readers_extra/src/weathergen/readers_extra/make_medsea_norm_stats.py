# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Make the normalisation statistics that data_reader_medsea.py reads.

Run once:

    python -m weathergen.readers_extra.make_medsea_norm_stats \\
        /work/cmcc/machine_learning/medformer/data/datasets/reanalysis/zarr \\
        medsea_norm_stats.json

The archive is 1.38 TiB, so we do not read all of it: we take a few days out of
every year and use only the sea points, since land is NaN. The output is one
mean and one standard deviation per channel:

    {"votemper_1m": {"mean": 18.5, "std": 4.2}, ...}
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import xarray as xr

from weathergen.readers_extra.data_reader_medsea import (
    CHANNEL_LEVEL,
    CHANNEL_VAR,
    CHANNELS,
    NX,
    NY,
    TIME_DIM,
)

_logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_dir", type=Path, help="directory holding reanalysis-YYYY.zarr")
    parser.add_argument("output", type=Path, help="JSON file to write")
    parser.add_argument("--days-per-year", type=int, default=4, help="days to sample per store")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    stores = sorted(args.zarr_dir.glob("reanalysis-*.zarr"))
    assert stores, f"No reanalysis-*.zarr store found in {args.zarr_dir}."

    # Count of values, their sum, and the sum of their squares, per channel.
    count = np.zeros(len(CHANNELS), dtype=np.int64)
    total = np.zeros(len(CHANNELS), dtype=np.float64)
    total_sq = np.zeros(len(CHANNELS), dtype=np.float64)

    for store in stores:
        ds = xr.open_zarr(store, consolidated=True, chunks=None, zarr_format=2)
        n_times = ds.sizes[TIME_DIM]
        # Spread the days over the year so that every season is sampled.
        t_idxs = np.linspace(0, n_times - 1, args.days_per_year, dtype=int)
        _logger.info(f"{store.name}: reading {len(t_idxs)} of {n_times} days")

        for t_idx in t_idxs:
            for var in dict.fromkeys(CHANNEL_VAR):
                field = ds[var].isel({TIME_DIM: int(t_idx)}).values
                field = field.astype(np.float64).reshape(-1, NY * NX)

                for c in range(len(CHANNELS)):
                    if CHANNEL_VAR[c] != var:
                        continue
                    values = field[CHANNEL_LEVEL[c]]
                    sea = values[np.isfinite(values)]
                    count[c] += sea.size
                    total[c] += sea.sum()
                    total_sq[c] += np.square(sea).sum()

    assert (count > 0).all(), (
        f"No sea values found for {[CHANNELS[c] for c in np.flatnonzero(count == 0)][:5]}."
    )

    mean = total / count
    # std = sqrt(mean of squares - square of mean), clipped because rounding can
    # make it slightly negative for a channel that barely varies.
    std = np.sqrt(np.maximum(total_sq / count - np.square(mean), 0.0))

    stats = {c: {"mean": float(mean[i]), "std": float(std[i])} for i, c in enumerate(CHANNELS)}
    args.output.write_text(json.dumps(stats, indent=2, sort_keys=True))
    _logger.info(f"Wrote statistics for {len(stats)} channels to {args.output}")


if __name__ == "__main__":
    main()
