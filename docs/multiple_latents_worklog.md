# Multi-time latent spaces — work log

Working notes, not reference documentation. The reference is
[`multiple_latents.md`](multiple_latents.md) and the rationale is
[`multi_time_latents_proposal.md`](multi_time_latents_proposal.md). Written in English to match
the other documents in `docs/`. Append a dated section per session rather than a new file.

---

## 2026-09-23 (morning)

### 1. Where things stood (before the rebase)

Branch `multiple_latent` carried **no commits** and still pointed at `d6607d73`, one behind
`develop`. The September implementation was gone — never committed, discarded by a `git restore`
around 2026-09-17 (reflog shows a checkout to `develop` and a pull that day). Only untracked
artifacts survived: the two test files, the coupled configuration, the stream directories and
`docs/`.

The missing commit, `f730876c`, touches `common/config.py` (74 lines), `model_interface.py` and
`trainer.py` — exactly the files the implementation modifies. Aligning the branch first was a
prerequisite, not housekeeping.

### 2. Decision: the encoder is split per group

This reverses the September design. Each group now gets a complete encoder tower — embedding
through global assimilation — fed **only by its own streams**, with its own latent tensor. Groups
meet at two explicit points: a new assimilation coupler at t = 0, and the periodic rollout
coupler.

**Multi-group stream assignment was adopted at the same time.** This forced a schema distinction
that had not existed before, and it is the thing to review first because everything else rests on
it:

| Relation | Declared in | Cardinality |
|---|---|---|
| **Input membership** — which towers ingest the stream | `latent.groups.<g>.streams` | several |
| **Decode target** — which latent it is predicted from | `streams.<s>.latent_group` | exactly one |

Without the split, "multi-group" would be ambiguous: a stream in two towers would be decoded
twice. With it, SYNOP can feed both the atmosphere and the ocean while remaining predicted from
one, and the converse case works too — a diagnostic, target-only stream decoded from a tower
without being one of its inputs.

A consequence worth keeping: the `queries` key is gone. Towers hold separate tensors, so there is
no shared query axis to partition, and with it goes the whole class of
`ae_local_num_queries > 1` layout bugs that blocked the September attempt.

### 3. Work done

**Rebase.** `multiple_latent` is now at `f730876c`. The local modification to
`config/streams/era5_synop_finetuning/synop.yml` was stashed and restored intact.

**`Q > 1` fix extracted.** Branch `multi_query_fix`, commit `46e39e77`. Three fixes — multi-query
packing in the aggregation, cell-major reshape in `assimilate_local`, `9 * Q` varlen lengths in
the decoder — plus a guard on register/class tokens and `tests/test_multi_query_layout.py`. With
per-tower latents this design no longer needs `Q > 1`, so it is an independent bug fix against
`develop` rather than a prerequisite.

**Tests pruned and rewritten.** `tests/test_latent_groups.py`, 22 cases. The five that validated
the query-axis partition are gone; the new ones cover the two stream relations, multi-membership,
per-group widths, the coupling schema and the rejection of `rope_2D`. Two further files were
added: `tests/test_latent_groups_model.py` (the grouped model on CPU) and
`tests/test_data_reader_grep.py`, which the GREP reader had entirely lacked.

**Configuration.** `config/streams/era5_cglors/` and a rewritten
`config/config_forecasting_coupled.yml`. Superseded material renamed with an `_old_` prefix:
`_old_fesom_coupled/` and three orphaned figures.

**Documentation.** `multiple_latents.md` rewritten as the reference document, with
`single_latent_v2.svg` recovered as the "before" figure and the detailed analysis of the
loss-masking option preserved.

**Phases 0-2 implemented**, about 900 lines across `common/config.py`, `encoder.py`, `engines.py`,
`model.py`, `model_interface.py`, `trainer.py` and `default_config.yml`.

The one technical note worth recording: `EmbeddingEngine` and the two index helpers all derived
their work from `batch.tokens_lens`, whose axis 2 is the stream axis. Passing a **restricted
view**, `batch.tokens_lens[:, :, stream_idxs, :]`, instead of the batch keeps the packing, the
scatter, the positional-encoding indices and `cell_lens` all consistent within a group with no
further changes. That is what made the per-tower refactor contained.

### 4. Test results

| | Result |
|---|---|
| `tests/test_latent_groups.py` | **22/22 passed** (21 plus the `rope_2D` rejection) |
| `tests/test_latent_groups_model.py` | **18/18 passed**, new |
| `tests/test_data_reader_grep.py` | **15/15 passed**, new |
| `pytest src/` (CI `unit-test` target) | **3 passed** |
| `tests/test_performance_utils.py` | 11 passed |
| `tests/test_sht_roundtrip.py` | 8 passed |
| `./scripts/actions.sh lint-check` | ruff check, ruff format and pylint all clean |

End-to-end on the real configuration — load, resolve streams, validate:

```
streams caricati : ['C-GLORS', 'ERA5']
validazione      : OK
input membership : {'atmosphere': ['ERA5'], 'ocean': ['C-GLORS']}
stride           : {'atmosphere': 1, 'ocean': 4}
coupling stride  : 4
num_steps=3      : rejected
```

So phase 0 is genuinely verified: sexagesimal `24:00:00` yields stride 4, the two stream relations
resolve separately, and the validator rejects a too-short rollout on a real file.

**Phases 1 and 2 are now partly exercised.** `tests/test_latent_groups_model.py` builds the real
grouped module tree on CPU and checks the construction (one tower and one engine per group, no
shared parameters between towers, two couplers with separate weights, no engine for a group that
is never advanced, `NotImplementedError` on differing widths) and the rollout dispatch, using
forward hooks so the stride behaviour is observed rather than recomputed: `engine:fast` at every
step, `engine:slow` plus `coupler:rollout` only at steps 4 and 8, and the held group returned as
the identical tensor object in between.

What still needs a GPU is a **full** forward pass. Every attention class calls `flash_attn_func`
unconditionally — `self.att = scaled_dot_product_attention` is assigned and never used — and the
encoder's local-to-global adapter always holds a cross-attention head, so the encoder can never
be made attention-free. The CPU test sidesteps this by configuring every block stack to length 0
and asserting that the subtree it forwards through contains no attention module. Note that
`pytest.importorskip("flash_attn")` is *not* a CPU guard: flash-attn is installed and imports
fine here, which is why `tests/test_encoder_teacher.py` fails rather than skips.

#### Four pre-existing problems found

1. **`tests/test_config.py:162` has `@pytest.fixure`** instead of `fixture`. One character, from
   commit `835e0b82`, and it fails collection for the entire `tests/` directory. While it is
   there, `pytest tests/` does not run for anyone.
2. **`tests/test_cli.py`: 16 failures.** Confirmed pre-existing by stashing onto clean `develop`:
   16 failed, 8 passed either way.
3. **`test_encoder_teacher.py`: 13 failures**, `RuntimeError: Expected all tensors to be on the
   same device, but found cuda:0 and cpu` in `ema.py:104`. Not flash-attn, and not a segfault —
   the login node has a GH200 and `torch.cuda.is_available()` is True, so `EMAModel` moves part
   of the model to it. Pre-existing on `origin/develop`.
4. **`test_collapse_monitor.py` kills the interpreter.** Not flash-attn either: the aarch64 LAPACK
   does a Fortran hard stop, `** On entry to SLASCL parameter number 4 had an illegal value` then
   `STOP 1`, inside `torch.linalg.svd` in `CollapseMonitor._compute_effective_rank`, which centres
   its input and so passes an all-zero matrix on rank-1 data. A `STOP` is not catchable, so the
   surrounding `except RuntimeError` cannot help and the file has to be excluded by name.
   (`torch.linalg.svdvals` returns `nan` on the same matrix instead of aborting.)

So the working incantation is to name the files, or

```bash
pytest tests/ --ignore=tests/test_config.py --ignore=tests/test_cli.py \
              --ignore=tests/test_encoder_teacher.py --ignore=tests/test_collapse_monitor.py
```

None are in scope for this work, so none were touched.

### 5. Data: C-GLORS v8 is staged

| Stream | File | Present |
|---|---|---|
| ERA5 | `aifs-ea-an-oper-...-o96-1979-2024-1h-v3-with-era51.zarr` | yes, in `data_paths[0]` |
| C-GLORS | `nemo_cglorsv8_subset_1993_2023.zarr` | yes, `/e/scratch/weatherai/cosi1` |

The Arctic v7 subset the configuration originally named was never staged here, and none of the
three entries in `jupiter/config/paths.yml` held an ocean store: `data_path_fesom` points at an
empty directory, `eerie-ifs-fesom-control1950-o96-6h-v1.zarr` exists but no stream config
references it, and the EERIE stream configs point at `/work/ab0995/...`, a DKRZ Levante path.

v8 replaces it and is better in every respect — global rather than Arctic, 1993-2023 rather than
2000-2023, ORCA0.25 native, with per-variable statistics in the store. Reaching it needs
`/e/scratch/weatherai/cosi1` appended to `data_paths`, which leaves every other stream's
resolution order untouched since the filename is unique to that directory. It lives in another
user's scratch directory, so it is purge-exposed; a project-space copy is the real prerequisite
for a long run.

Three defects in `packages/readers_extra/.../data_reader_grep.py` stood between this store and a
working run, none of them in the latent-group code, all three now fixed:

1. `__init__` did not accept the `stage` keyword the sampler has passed since `2af44f70`, so the
   reader raised `TypeError` before it ever opened a store — it was dead for **every** dataset,
   not just this one.
2. It opened with `zarr_format=2` while the store is zarr v3, and the surrounding
   `except Exception` logged and returned, leaving the reader in the empty state. The sampler then
   substitutes spoofed mean-valued data, so the run would have trained ERA5-only with nothing but
   one `ERROR` line to show for it. The kwarg is gone (zarr auto-detects both formats) and the
   open now propagates.
3. The time dimension is `time_counter`, and the reader knew only `time` and `time_centered` —
   in the `available_vars` filter, which would have selected no channels at all, in the
   coordinate lookup, whose fallback raised an uncaught `KeyError`, and in the per-timestep
   `isel`. It is now resolved structurally to a *(dimension, coordinate)* pair, which also fixes
   stock NEMO output, where `time_centered` is an auxiliary coordinate on the `time_counter`
   dimension and `isel(time_centered=...)` was never valid.

A fourth, in the same file: the land mask is the coordinate sentinel `(-1, -1)`, but the reader
only NaN'd `(0, 0)`, of which there are none. All 288,240 masked points therefore kept a valid
coordinate and landed in a single HEALPix cell — 36,037 tokens in one cell, well past the
64-token per-cell assert in the local assimilation. The stream now sets `drop_all_nan_rows`,
which drops the rows whose every selected channel is NaN: 1,512,000 points become 1,218,909.

> Unrelated but worth fixing: `WeatherGenerator-private/hpc/jupiter/config/paths.yml` holds an
> MLflow token and two sets of AWS credentials in plaintext.

> Also worth knowing: the `weatherai` Slurm account's QOS is `suspended` (`MaxWall=00:00:00`,
> `DenyOnLimit`) while `$BUDGET_ACCOUNTS` still names it, so `launch-slurm.py` would submit
> against it and be refused — after copying both repositories and running a full `uv sync`. Pass
> `--account e-ext-2025e01-128`, whose QOS `normal` allows 12 h.

### 6. Smoke test

Testing phases 1-2 does not need a physically meaningful coupled model — it needs two groups over
two streams. `integration_tests/small_coupled.yaml` uses streams that do have data:

- groups named **`fast`** and **`slow`**, deliberately not `atmosphere`/`ocean`, because the
  pairing is not physical
- `fast: [ERA5, SurfaceCombined]`, `slow: [NPPATMS, SurfaceCombined]` — SurfaceCombined in both,
  which exercises multi-membership
- decode targets: ERA5 and SurfaceCombined from `fast`, NPPATMS from `slow`, so both towers are
  decoded and both receive gradient
- `num_steps: 4`, because `output_idxs` is `range(1, 5)` and the slow group has stride 4
- `num_class_tokens` and `num_register_tokens` set to 0 for the first run, to keep the number of
  moving parts down
- roughly small_multi_stream in size: healpix 4, width 512, one mini-epoch

Validated on CPU:

```
input membership: {'fast': ['ERA5', 'SurfaceCombined'], 'slow': ['NPPATMS', 'SurfaceCombined']}
decode target   : {'ERA5': 'fast', 'NPPATMS': 'slow', 'SurfaceCombined': 'fast'}
stride          : {'fast': 1, 'slow': 4}
multi-membership: ['SurfaceCombined']
```

#### Submission

```bash
cd /e/project1/weatherai/norberti1/WeatherGenerator
../WeatherGenerator-private/hpc/launch-slurm.py \
  --stage train \
  --run-id smokec01 \
  --account e-ext-2025e01-128 \
  --base-config $PWD/integration_tests/small_coupled.yaml \
  --nodes 1 --chain-jobs 1 --cleanup-scripts yes --no-register \
  --options training_config.samples_per_mini_epoch=64 \
            validation_config.samples_per_mini_epoch=16 \
            data_loading.num_workers=8 \
  --time=00:20:00 --partition=booster
```

`--base-config`, not `--config`: the latter is for overlays on top of `default_config.yml`, while
this is a complete configuration. It must be **absolute**, because it is the one path the launcher
never remaps into the job copy directory — a relative path would resolve against the job's own
working directory. `/e/project1` is GPFS, so the original file is readable from a compute node,
which also means editing it while the job is queued changes what the job sees.

Four things that are easy to get wrong here:

- **`--account e-ext-2025e01-128` is required.** `$BUDGET_ACCOUNTS` names `weatherai`, whose QOS
  is `suspended` (`MaxWall=00:00:00`, `DenyOnLimit`), so `sbatch` refuses — and it refuses *after*
  `_prepare_stage_directory` has copied both repositories and run a full `uv sync`.
- **Shrinking `--gres` buys nothing.** `booster` is `OverSubscribe=EXCLUSIVE`, so the allocation
  is whole 4-GPU nodes regardless. A short `--time` is the real lever on queue wait, and keeping
  all four tasks is what makes `cf.with_ddp` true so FSDP2 actually shards the towers, the engines
  and the couplers — the least-tested interaction. A single-GPU job would skip it, since
  `with_fsdp` is ignored when `world_size == 1`.
- **`small_coupled.yaml` has a relative `streams_directory`** and `copy_git_tracked_directory`
  uses `git ls-files`, so the streams are only copied into the job directory once
  `integration_tests/` is committed. The coupled config does not have this problem: everything
  under `config/` is copied whether tracked or not.
- **Never reuse a `--run-id`**: the copy directory hard-fails if it already exists. And keep
  `--cleanup-scripts yes` — that job is what creates `logs/<run_id>/`, which `--output` needs.

`launch-slurm.py` has no dry-run at all; `sbatch` is called with `check=True` and everything
expensive happens before it, so a shim on `$PATH` costs a full copy and sync to learn nothing.

### 7. Open items

- The smoke run has not been submitted. Expect the first attempt to fail on shapes rather than
  logic; read the logs before launching a second.
- Phases 0-2 are committed on `multiple_latent`; nothing is pushed.
- `forecast.time_step` must be a whole multiple of `time_window_step`, and nothing checks it. The
  target window is located by `idx + (time_step * i) // time_window_step`, so a 24 h window step
  under a 6 h forecast step collapses the +6/+12/+18 h targets onto the analysis window itself.
  Widening the window step is therefore *not* a way to align a daily stream to 00 UTC.
- C-GLORS being daily, the ocean tower has an analysis in one window out of four; in the other
  three it runs on the masked-cell path and `is_spoof` zeroes its loss. Making a daily mean valid
  for the whole day needs a reader-side `_get_dataset_idxs` override plus relabelling the
  datetimes into the window, since `check_reader_data` asserts they fall inside it. Semantic
  choice, deliberately deferred.
- Per-group `dim_embed` is inert: the coupler handles it, the decoder and forecasting engine do
  not. That is phase 3, and `ocean.dim_embed` is 2048 in the coupled config until then.
- The ocean tower carries ~42.8k source tokens against ERA5's ~11-12k at O96. There is no
  source-side subsampling anywhere in the framework (`max_num_targets` caps targets only), so a
  `spatial_stride` in the reader is the lever if that imbalance bites.
- Decoding on steps where a group did not advance is still undecided; masking the loss now looks
  clearly better. See the open-decision section of `multiple_latents.md`.
- `teacher_utils.py:70` sets `model.forecast_engine = None` for the SSL teacher and would leave
  `forecast_engines` alive. Not touched, since SSL and groups do not coexist yet.
- Warm-start and `freeze_modules` are untested: both need a pre-existing single-latent checkpoint.
