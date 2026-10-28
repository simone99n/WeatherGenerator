![WeatherGenerator logo](https://raw.githubusercontent.com/ecmwf/WeatherGenerator/refs/heads/sorcha/dev/1696/assets/wg_logo.svg)

# The `weathergen-readers-extra` package

Data readers for datasets that the two built-in readers do not cover. A reader turns one file of
one dataset into the flat point clouds the model consumes, one per time window.

This page is the tutorial for writing a new one. It follows
[`data_reader_medsea.py`](src/weathergen/readers_extra/data_reader_medsea.py), the Mediterranean Sea
reanalysis reader, because it is the smallest complete example: one file per year, nine variables,
a land-sea mask that exists only as a pattern of NaN.

Paths like `/e/scratch/weatherai/norberti1` are from the Jupiter setup of that dataset. They are
examples, not requirements.

---

## 1. When you need a reader, and when you do not

You do **not** need one if your data is an [anemoi](https://github.com/ecmwf/anemoi-datasets)
dataset: `DataReaderAnemoi` already handles those, and the stream config selects channels,
statistics and geoinfos. Most gridded reanalyses reach the model that way.

You need one when the file layout is the dataset's own: a NEMO output, an ICON run, a table of
observations. The job of a reader is small and precisely bounded:

> Given a time window, return the points inside it — coordinates, values, timestamps — as flat
> arrays, plus the statistics needed to normalise them.

Everything else (tokenisation, masking, batching, the loss) is the framework's.

---

## 2. The mental model: cells, points, channels, steps

This is the part worth getting right before writing code, because the rest follows from it.

The model does not consume grids. It consumes **point clouds**: every row of the returned array is
one point with a latitude, a longitude, a timestamp, its channel values and its geoinfos.

For the Mediterranean reader the vocabulary is:

| Name | What it is | Number in that dataset |
| --- | --- | --- |
| **cell** | one position of the rectangular grid as stored in the file, land included | 380 × 1016 = 386 080 |
| **point** | one **water column**: a cell that is sea, which is all the reader returns | 144 990 (37.6%) |
| **level** | one of the 18 depths | 18 |
| **channel** | one variable, or one variable at one level | 5 + 4 × 18 = 77 |
| **step** | one time step of the window being asked for | 1 per day here |

Two consequences that shape the whole reader:

**A point is a column, not a box.** The 18 levels do not multiply the points, they become
*channels* of the same point: `votemper_1m … votemper_971m`. In three dimensions that dataset has
2 241 828 wet cells, but only 144 990 points. Depth went into the channel axis.

**Land is excluded once, not filtered per sample.** A land cell would give a row of pure NaN, so the
mask is read once in `__init__` and turned into an index:

```python
wet = np.isfinite(self.ds["votemper"].isel({TIME_DIM: 0}).values)  # (levels, rows, cols)
wet = wet.reshape(len(DEPTH_M), NY * NX)                           # (levels, cells)
self.sea_points = np.flatnonzero(wet[0])                           # (points,), an index into cells
```

`sea_points` is the bridge between the two index spaces: its *values* live in cell space, its
*length* is the number of points. It is one-directional — there is no cell-from-point map in the
reader, and if you need one (to write a prediction back onto the grid) you build it.

It is valid **only** on arrays flattened in C order to `(..., cells)`. A stray transpose or a
Fortran-order reshape does not raise: it silently returns values from other places in the basin.

Every line of the reader that changes an array's shape says what the new shape is, using exactly
these names. Copy that habit; it is the cheapest documentation there is.

---

## 3. The contract

Your class derives from `DataReaderTimestep` (data at a fixed period) or from `DataReaderBase`
(irregular data, such as observations), both in
`src/weathergen/datasets/data_reader_base.py`.

**Seven attributes** must exist by the end of `__init__`
([data_reader_base.py:305-312](../../src/weathergen/datasets/data_reader_base.py)):

```python
source_channels, target_channels, geoinfo_channels   # names, in column order
source_idx, target_idx, geoinfo_idx                  # indices into your full channel list
target_channel_weights                               # usually self.parse_target_channel_weights()
```

**Four statistics arrays**, not declared abstract but required:

```python
self.mean, self.stdev              # (all channels,)  indexed by CHANNEL index
self.mean_geoinfo, self.stdev_geoinfo  # (geoinfos,)  indexed by POSITION
```

That asymmetry is real and it is the single easiest thing to get wrong. `_normalize` does
`mean[ch]` with the global channel index, while `normalize_geoinfos` iterates
`enumerate(self.geoinfo_idx)` and does `mean_geoinfo[i]` positionally. So `mean` is the full list
and `mean_geoinfo` is already the selected subset:

```python
self.mean_geoinfo = self.mean[self.geoinfo_idx]
```

**Two methods:**

```python
def length(self) -> int              # number of samples; see the note below
def _get(self, idx, channels_idx) -> ReaderData
```

`length()` is wired only to `__len__` and nothing in the pipeline reads it. Iteration is driven
entirely by `TimeWindowHandler`. Implement it, but do not look for meaning that is not there.

**The constructor signature is fixed by the caller**
([multi_stream_data_sampler.py:227-268](../../src/weathergen/datasets/multi_stream_data_sampler.py)):

```python
def __init__(self, tw_handler: TimeWindowHandler, filename: Path, stream_info: dict, stage: Stage)
```

`stage` may be ignored (medsea ignores it), but it must be accepted. One reader instance is built
**per entry in the stream config's `filenames`**, and each is asked for every window: the ones that
do not hold it return `ReaderData.empty`, and the framework concatenates.

### Where `idx` and `channels_idx` come from

`_get` is never called by you. The base class calls it twice per direction:

```python
def get_source(self, idx):  rdata = self._get(idx, self.source_idx)
def get_target(self, idx):  rdata = self._get(idx, self.target_idx)
```

So `channels_idx` is always `source_idx` or `target_idx` — that is the only reason it is a
parameter, and it means `_get` must never consult `self.source_idx` directly, or targets would be
read with the source's channels. The column order of the returned `data` is the order of
`channels_idx`, which is the order the stream config listed: reorder it and normalisation will
quietly apply one channel's mean to another.

`idx` is a **time-window index**, not a sample number and not a row in your file:

```
window(idx) = [ t_start + idx * time_window_step ,  + time_window_len )
```

It comes from the sampler's shuffled permutation of window indices, becomes `base_idx` for a
sample, and is then offset once per input step and once per forecast step
(`multi_stream_data_sampler.py:570-595`). For one training sample your `_get` is called several
times with different `idx`.

The run's clock meets your file's clock on the first line of `_get`:

```python
(t_idxs, dtr) = self._get_dataset_idxs(idx)   # rows into your time axis, and the window as dates
```

---

## 4. What happens to your data afterwards

`collect_datasources` ([multi_stream_data_sampler.py:52-85](../../src/weathergen/datasets/multi_stream_data_sampler.py))
is the whole downstream pipeline:

```python
rdata = get_reader_data(idx).shuffle(rng, shuffle, num_subset).remove_nan_coords_and_geoinfos()
rdata.data = normalize_channels(rdata.data)
rdata.geoinfos = ds.normalize_geoinfos(rdata.geoinfos)
```

Three consequences nobody guesses:

**1. NaN in `data` stays in the point cloud, and the framework handles it.** Only NaN in `coords` or
`geoinfos` drops the row. A NaN source value becomes 0 after normalisation, i.e. the channel mean
(`stream_data.py`), and a NaN target contributes no error to the loss. So a reader may leave
missing values as NaN, as `data_reader_template.py` does. Filling them with the channel mean, as the
Mediterranean reader does, gives the same inputs, but turns those points into targets of value 0
that do count in the loss:

```python
missing = ~np.isfinite(data)
data[missing] = np.broadcast_to(self.mean[channels_idx], data.shape)[missing]
```

What a reader must never write is a plain `0`: it would mean a sea temperature of 0 °C.

**2. A NaN geoinfo deletes the point.** `remove_nan_coords_and_geoinfos` drops any row whose
geoinfos are not all finite. A deep channel used as a geoinfo would silently shrink the point
cloud, so fill the geoinfo columns too.

**3. `datetimes` must lie in `[window.start, window.end)`.** `check_reader_data` asserts it. With
daily means labelled at noon this is where off-by-one-day bugs appear. Call the checker before
returning, always:

```python
rd = ReaderData(coords=coords, geoinfos=geoinfos, data=data, datetimes=datetimes)
check_reader_data(rd, dtr)
```

One more thing to know: if your reader returns empty, the sampler substitutes a **spoofed** sample
of mean values (`multi_stream_data_sampler.py:575-586`, a workaround for a PyTorch issue with empty
tensors). Empty windows therefore do not crash training and do not appear in the logs. This is
convenient and dangerous in equal measure — see the next section.

---

## 5. Time: the part that bites

**Read the period, never assume it.** One line is enough, and it makes 3-hourly, daily and
multi-day output read identically:

```python
period = self.times[1] - self.times[0]
```

**A window shorter than your period gives data one time in `period/len`.** Measured on the daily
Mediterranean store, for a window stepping every 6 hours:

| `time_window_len` | windows with data |
| --- | --- |
| `06:00:00` | 50 / 200 |
| `12:00:00` | 100 / 200 |
| `24:00:00` | 200 / 200 |

The other windows are spoofed with mean values, silently. If your dataset is daily, the run config
needs a window of at least a day — and that is a decision about the *whole run*, since the window
length is global and every other stream pays for it.

**Declare the interval you cover, not the instant you label.** This one is still an open defect in
the medsea reader. `get_dataset_indexes_timestep` returns empty unless the window is entirely
inside `[data_start_time, data_end_time]`, and a daily mean labelled at noon covers half a day on
each side of its stamp. With the stamps as bounds, the first and last day of every store are
unreachable:

```
stamps as bounds     windows with data: 363/365   (first and last day lost)
interval as bounds   windows with data: 365/365
```

The fix is `self.times[0] - period / 2` and `self.times[-1] + period / 2`, and it is verified that
every returned stamp still lands inside its window. Know about it when you write yours.

**Two more fragilities worth a guard:**

- a store with a single time stamp raises `IndexError` on `self.times[1]`. `data_reader_grep` and
  `data_reader_mesh` fall back to a default or to a `frequency` key from the stream config.
- an irregular axis (monthly means, 28–31 days) breaks the fixed-period model. It does not fail
  silently, but it fails obscurely: measured on twelve mid-month stamps, 9 windows out of 365
  return a stamp outside their own window, so `check_reader_data` fires with a message that points
  nowhere near the cause. One `np.diff` check at open time names it properly.

---

## 6. Geoinfos come from the dataset

Geoinfos are the static-ish context the model gets alongside the values: orography, a land-sea
mask, the depth of the sea floor. The rule in this project is:

> **A reader reads geoinfos from the dataset. It does not compute them.**

If the field you want is not in the store, the answer is a preprocessing step that writes it in, not
a computation in `__init__`. A derived geoinfo hides a gap in the data inside reader code, where it
cannot be inspected, reused by another reader, or normalised from the archive's own statistics.

So `geoinfo_channels` is read from the stream config and resolved against your channel list, exactly
like `source` and `target`:

```python
self.geoinfo_channels = stream_info.get("geoinfo_channels", [])
self.geoinfo_idx = [CHANNELS.index(c) for c in self.geoinfo_channels]
```

Most readers in this package have no geoinfos at all (`geoinfo_channels = []`), which the framework
accepts: the array simply has zero columns. The Mediterranean store has no static field either — no
bathymetry, no mask — so it declares none.

Note that the framework does **not** require a geoinfo to be constant in time; `DataReaderAnemoi`
says so explicitly. If yours varies, read it inside the per-step loop next to the data instead of
repeating one array per step.

---

## 7. Statistics: keep them in the store

Normalisation statistics are per channel, which for a 3-D variable means **per level**: in the
Mediterranean the temperature goes from 20.7 °C with a deviation of 4.55 at the surface to 13.5 °C
with 0.28 at 971 m. One number per variable would flatten the deep signal entirely.

Where to keep them matters more than it sounds. This package computes them once and writes them
**into the store**, as two subgroups:

```
reanalysis-2021.zarr/
├── votemper/ vosaline/ ...        the data, untouched
├── mean/   deptht depthu depthv + one array per variable
└── std/    the same
```

Subgroups do not appear in the root dataset, so nothing else changes, and the reader reads them from
the file it already opened — no second path to configure, and no way to point a run at statistics
computed from different data.

Three scripts in this package do it, and they are worth copying as a pattern:

```
medsea_stats.py        <store>... --out stats.npz     # computes, touches no store
medsea_write_stats.py  <store>... --from-npz ...      # writes the groups (--dry-run first)
medsea_check_stats.py  <store> --days 8               # eight pass/fail checks
```

### The four traps in this area

**`fill_value` must be NaN.** Created without it, a zarr v2 array gets `fill_value: 0`, xarray reads
that as `_FillValue` and masks every value equal to it — so a standard deviation of exactly 0 comes
back as NaN. A deviation of 0 is a plausible value (a channel that does not vary), which makes the
loss silent. Pass `fill_value=np.nan` on every statistic array.

**Surface statistics are 0-d.** A variable with no level has one number, stored as a 0-d array, and
`array[0]` on a 0-d array raises `IndexError`. Hence:

```python
np.atleast_1d(statistics[group][CHANNEL_VAR[c]].values)[CHANNEL_LEVEL[c]]
```

**A zero deviation must be refused, not clipped.** `_normalize` divides without a guard, so a
channel with no variation produces `inf` in silence. The computing script *reports* a non-positive
variance with its variable and level instead of hiding it behind `np.maximum(var, 0)`, and the
reader asserts `std > 0` for every channel the stream actually selected.

**A missing group raises `KeyError`**, not `FileNotFoundError`:
`"'mean' not found in consolidated metadata."` Catch that and say how to create it.

---

## 8. Registering it, and the three config layers

### Register

One `case` in [`registry.py`](src/weathergen/readers_extra/registry.py), keyed by the `type` field
of the stream config, with a lazy import so a broken optional dependency does not break startup:

```python
case "medsea":
    from weathergen.readers_extra.data_reader_medsea import DataReaderMedSea
    return DataReaderMedSea
```

### Layer 1 — the machine

In the private repository, `hpc/<machine>/config/paths.yml`. Your dataset's directory has to be in
**`data_paths`**: that is the list the loader resolves stream `filenames` against. An absolute path
in `filenames` is used as it is and needs no entry, as in the Mediterranean stream config. The neighbouring
`data_path_*` keys are a legacy fallback for configs with no `data_paths`, and that fallback is a
hardcoded list of five names — adding `data_path_mydataset` alone does nothing.

### Layer 2 — the run config

`load_merge_configs` merges in ascending precedence: **default config → private config → each file
passed with `--config` → `--options`**. So a run config is an **overlay**, not a copy: the existing
full-length `config_*.yml` files are copies by habit, not necessity. Twenty lines are enough:

```yaml
streams_directory: "./config/streams/era5_medsea/"

training_config:
  start_date: 2021-01-02T00:00
  end_date: 2021-11-30T00:00
  time_window_len: 24:00:00      # a day, because the ocean fields are daily means
  time_window_step: 06:00:00
  samples_per_mini_epoch: 1024   # 1312 windows exist; the default 4096 would not fit
```

Two keys to get right for a new dataset: the **dates**, which must lie where your data exists, and
**`time_window_len`**, per section 5.

### Layer 3 — the stream config

One YAML per stream in `config/streams/<experiment>/`. The top-level key is the stream name, which
the loader injects back as `stream_info["name"]`, so that key always exists at runtime. Keys your
reader reads (`filenames`, `source`, `target`, `geoinfo_channels`) sit next to keys the *model*
reads (`embed`, `token_size`, `max_num_targets`) — knowing which is which saves confusion.

### Launch

```bash
weathergen train --config config/config_era5_medsea.yml \
  --private-config <private repo>/hpc/jupiter/config/paths.yml
```

Before launching anything, this is worth ten seconds: load the config with `load_merge_configs`,
resolve the filenames against `data_paths`, build the readers and ask each for a window. It catches
a wrong path, an empty stream or a window that contains no data, and it allocates no GPU.

---

## 9. Testing without the real archive

The Mediterranean archive is 1.38 TiB, so the tests build a synthetic store with the same
*pathologies* as the real one:

- the same variable names and depth coordinates;
- `nav_lat`/`nav_lon` zeroed, since the real ones are zero on 41% of the cells and the reader has to
  rebuild the grid instead of trusting them;
- timestamps at noon, like real daily means;
- land as NaN, with a **nested** mask whose sea floor shallows in one direction, so that per-level
  counts differ and an inverted level shows up.

Two habits that paid off:

**Write the statistics with the real writer**, not with a copy of it in the fixture. The test then
exercises that code, and because one synthetic statistic is exactly `0.0`, the `fill_value` trap
above became a live regression test.

**Give the analytic store an exact truth.** Fill level *L* with `10 * L` and the mean is exactly
`10 * L` and the deviation exactly `0` — assert with `assert_array_equal`, not `approx`, and assert
that the script *reported* the zero variance instead of clipping it.

One caveat: the CI runs `pytest src/`, so nothing under `tests/` runs in CI. Tests there are run by
people, deliberately. Keep that in mind when you decide how much to rely on them.

---

## 10. Checklist and traps

Before opening a pull request:

- [ ] the seven attributes are set on every path, including the skipped-stream path;
- [ ] `mean`/`stdev` cover **all** channels; `mean_geoinfo`/`stdev_geoinfo` only the selected ones;
- [ ] `_get` uses `channels_idx` and never `self.source_idx`;
- [ ] column order is the config's order;
- [ ] missing values in `geoinfos` filled; in `data`, NaN is fine and a plain 0 is not;
- [ ] `check_reader_data(rd, dtr)` called on every non-empty return;
- [ ] the period comes from the file;
- [ ] every reshape carries its shape annotation;
- [ ] unknown channel names produce an error naming the stream, the field and what is available;
- [ ] a `case` in `registry.py`, a stream config, and the dataset's directory in `data_paths`;
- [ ] tests on a synthetic store, including one exact-truth case.

| Trap | What happens | Where |
| --- | --- | --- |
| a plain 0 for a missing value | a physical value of 0 enters the inputs and the loss | §4 |
| NaN in `geoinfos` | the whole point disappears from the sample | §4 |
| `mean_geoinfo` as the full array | geoinfos normalised with another channel's statistics | §3 |
| window shorter than the period | most samples silently spoofed with mean values | §5 |
| stamps declared as bounds | first and last step of every file unreachable | §5 |
| single-stamp store | `IndexError` on `times[1]` | §5 |
| irregular time axis | obscure assert from `check_reader_data` | §5 |
| `fill_value` left at 0 | a statistic of exactly 0.0 reads back as NaN | §7 |
| `[0]` on a 0-d statistic | `IndexError` for every surface variable | §7 |
| zero deviation | `inf` after normalisation, no warning | §7 |
| `sea_points` on a differently flattened array | values from the wrong places, no error | §2 |
| first file outside the run window | channel counts collapse to zero | §3 |

### Known open points in the medsea example

Documented rather than hidden, because the code is what it is today:

- the first and last step of every store are unreachable (§5); the two-line fix is not applied;
- `sowaflup` has a value 294 standard deviations from its mean, and it is a source channel: real
  tail or artefact is not settled;
- nothing under `tests/` runs in CI;
- `batch_size` is not in `default_config.yml` and its origin was not found, which matters when
  choosing `samples_per_mini_epoch`.

---

## Licence

This package is licensed under the Apache-2.0 License.
