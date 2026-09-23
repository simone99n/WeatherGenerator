# Proposal — Multi-time latent spaces with per-group encoders

**Status:** accepted; phases 0-2 implemented on branch `multiple_latent`, phases 3 and 4 open.
The reference for what was built is [`multiple_latents.md`](multiple_latents.md); this document is
the decision record for why.
**Date:** 2026-09-23

---

## 1. Summary

We propose to restructure WeatherGenerator around **multi-time latent spaces**: named groups,
each representing one Earth-system component, each with its **own encoder tower**, its **own
latent subspace**, and its **own dynamical time step**. Groups exchange information at exactly
two points — once at analysis time and periodically during rollout — through explicit coupling
blocks.

This differs from the earlier latent-groups design, which partitioned only the *dynamics* and
kept one shared encoder writing every stream onto every group. Here the separation is complete:
the atmospheric tower sees only atmospheric streams, the ocean tower only ocean streams, and no
parameters are shared between them.

![Multi-time latent spaces with per-group encoders](figures/multi_time_latents_separate_encoders.svg)

## 2. Motivation

The shared encoder is the single structure that forces every Earth-system component to agree on
everything: one HEALPix level, one embedding width, one set of assimilation weights, one
tokenisation. Three consequences follow, and all three are load-bearing for where the project
wants to go.

**Resolution is uniform whether or not that makes sense.** The ocean interior and the
stratosphere do not need the level-5 grid the boundary layer needs, but they pay for it in memory
and compute at every layer. Per-group resolution was assessed previously and rejected as too
invasive precisely *because* the encoder was shared — `num_healpix_cells` is baked into the
neighbour table, the RoPE coordinates and the decoder gather. Separate towers dissolve that
objection: each tower can carry its own grid, and only the coupling block needs to bridge them.

**Components cannot be developed independently.** With one trunk, adding a land-surface or
sea-ice component means retraining the shared encoder and re-validating every existing stream.
With towers, a new component is an added module; existing towers can be frozen through
`freeze_modules`, which already matches module paths by regex.

**Cross-domain information flow is implicit and unmeasurable.** In the shared encoder, ocean
information reaches the atmospheric latent through the same attention that does everything else.
There is no way to ask how much of the atmospheric skill depends on the ocean input, because
there is no place where that exchange happens. Making the coupling an explicit object makes it
measurable, ablatable, and — for a coupled model — physically interpretable.

Panel **b** of the figure states the trade plainly: we replace continuous implicit mixing with
two explicit exchanges. That is the whole proposal, and section 7 is honest about what it risks.

## 3. Architecture

Per group *g*:

| Stage | Scope | Note |
|---|---|---|
| Stream embedding | per group | only the streams assigned to *g* |
| Local assimilation (per HEALPix cell) | per group | own `ae_local_*` |
| Local → global adapter | per group | own learnable queries `q_cells` |
| Query aggregation | per group | own `ae_aggregation_*` |
| Global assimilation | per group | own `ae_global_*`, own width `D_g` |
| **Assimilation coupling** | **across groups** | **new; runs once, at t = 0** |
| Forecasting engine | per group | own depth and own `Δt_g` |
| **Rollout coupling** | **across groups** | periodic, at its own cadence |
| Decoder | per stream | reads its group's slice only |

The rollout dispatch is unchanged from the latent-groups design: group *g* advances at output
step *i* when `i % stride_g == 0`, with `stride_g = Δt_g / <stage>.forecast.time_step`, and is
held constant in between.

### 3.1 The coupling blocks

Both couplers are cross-attention over the concatenated token axis, with per-group input and
output projections into a shared coupling width `D_c`. The projections are what allow groups to
carry different `D_g`, and what will later allow them to carry different HEALPix levels — a
level-`L` and a level-`L+1` group are related by the nested-ordering parent/child arithmetic, so
the projection becomes a pooling on one side and a broadcast on the other.

The two couplers are separate modules with separate weights. They do different jobs: the
assimilation coupler reconciles *observations of different domains taken at the same time*; the
rollout coupler exchanges *forecast state between components running at different rates*.
Sharing weights between them would be an assumption we have no reason to make.

### 3.2 Configuration sketch

```yaml
latent:
  default_group: atmosphere

  groups:
    atmosphere:
      streams: [ERA5, SYNOP]          # NEW: exclusive stream assignment
      healpix_level: 5                # NEW: per-group grid (phase 4)
      dim_embed: 2048                 # NEW: per-group width
      encoder:                        # NEW: per-group encoder overrides
        ae_local_num_blocks: 0
        ae_global_num_blocks: 4
        ae_local_num_queries: 1
      forecast:
        time_step: 06:00:00
        num_blocks: 16
    ocean:
      streams: [C-GLORS]
      healpix_level: 4
      dim_embed: 1024
      encoder:
        ae_global_num_blocks: 2
      forecast:
        time_step: 24:00:00
        num_blocks: 8

  coupling:
    dim_embed: 2048                   # shared coupling width D_c
    assimilation:                     # NEW: runs once after the encoders
      num_blocks: 2
      num_heads: 32
    rollout:
      time_step: 24:00:00
      num_blocks: 1
      num_heads: 32
```

Note what changes relative to the earlier schema: `queries` disappears. Groups no longer
partition a shared query axis — each tower produces its own latent tensor, and the "slice" is a
tensor boundary rather than an index range. This removes the contiguity constraint and the whole
class of `ae_local_num_queries > 1` layout bugs that blocked the previous attempt.

## 4. Design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Encoder separation | Complete — embedding through global assimilation | Anything shallower leaves a shared trunk, and the shared trunk is what blocks per-group resolution |
| Encoder inputs | Each tower sees only its own streams | Keeps towers independently trainable and replaceable |
| Stream assignment | A stream may feed several towers, but decodes from one | Observations do not respect domain boundaries; adopted from the start because retrofitting it would invalidate every trained checkpoint |
| Coupling | Both at assimilation and during rollout | With inputs separated, t = 0 exchange has to be restored explicitly or the model is two models |
| Cadence | Duration, not step count | Stride derived from the stage's base step, so changing the base step preserves physical Δt |

## 5. What this unlocks

- **Per-group spatial resolution** — the change that was previously out of reach. A coarse ocean
  tower at level 4 costs a quarter of the cells of a level-5 one.
- **Independent pretraining and freezing** — train the atmosphere, freeze it, train the ocean
  against it. `freeze_modules: "encoders\\.atmosphere.*"` needs no new machinery.
- **Modular growth** — land surface, sea ice, atmospheric composition each arrive as a tower plus
  a line in `latent.groups`.
- **A measurable coupling** — the ablation in section 10 becomes possible only because the
  exchange has an address.

## 6. Cost

Encoder parameters scale with the number of groups. For the coupled configuration
(`ae_global_dim_embed: 2048`, `ae_global_num_blocks: 4`), the global assimilation stack alone is
roughly `4 · (4·D² + 2·h·D²)` ≈ **1.3 × 10⁸ parameters** per tower — an estimate from the block
shapes, not a measured count, since `print_num_parameters` has not been run on this design. A
second tower at the same width roughly doubles that; the per-group `dim_embed` in the schema
above is the lever for not paying it in full, and is the reason it is in the schema from the
start rather than deferred.

Compute is closer to neutral than parameter count suggests: each tower processes only its own
streams, so the total token volume entering the encoders is unchanged. What is genuinely new is
the assimilation coupler, which is dense attention over the concatenated token axis.

## 7. Risks

**Stream assignment was the weak point, and is now handled.** Real observations do not respect
domain boundaries: satellite radiances see the whole column, SYNOP 2 m temperature over sea is set
by both the atmosphere and the skin temperature beneath it, and scatterometer winds constrain the
surface stress that couples them. An exclusive assignment would make each such stream invisible to
every tower but one until the coupling block. **Multi-group assignment is therefore adopted from
the start** (decided 2026-09-23): `latent.groups.<g>.streams` declares input membership and may
overlap between groups, while the stream's own `latent_group` names the single tower it is decoded
from. The residual cost is that a shared stream pays its embedding once per tower. A dedicated
"surface" group whose job is exactly the interface remains available if that proves insufficient.

**All cross-domain analysis now rests on one block.** If the assimilation coupler is
under-parameterised or trains poorly, the model degrades into two independent models that happen
to share a loss. This is the primary thing the evaluation in section 10 must detect, and it is
the reason we would not ship this without the ablation.

**An empty group must still emit a well-formed latent.** If a tower's streams have no data in a
window, the whole tower falls through to the masked-cell path — learnable queries plus positional
encoding, with nothing assimilated. The mechanism exists and is well-defined, but previously
other streams would partially fill those cells; now an entire component can be analysis-free.
The `is_spoof` weighting handles the loss side, but the coupler will be attending to a latent
carrying no information, and it should be told so rather than left to infer it.

**The open decision on non-advancing steps carries over, and sharpens.** A slow group's decoder
is still called at every output step while its latent advances once. With a shared encoder there
was at least a common trunk through which some information leaked; with towers there is not. The
choice between accepting it and masking the loss — set out in
[`multiple_latents.md`](multiple_latents.md) — should be settled as part of this work, and the
masking option now looks clearly better.

## 8. Open questions

1. Should the assimilation coupler be symmetric, or directed with per-pair weights?
2. Does the decoder read only its own group, or also the coupled state? (`latent_read`, designed
   but never implemented.)
3. Do per-group latent/SSL losses become necessary once towers are independent, or does the
   physical loss suffice to train them?
4. Decoding on steps where a group did not advance — accept the offset, or mask the loss? See
   [`multiple_latents.md`](multiple_latents.md); masking now looks clearly better.

## 9. Implementation plan

The previous implementation was lost — see section 11 — so this started from `develop`, with the
surviving tests and documentation defining the contract. Phases 0-2 are done.

| Phase | Content | Depends on |
|---|---|---|
| 0 | Group scaffolding: config schema, validation, stream→group dispatch | — |
| 1 | Encoder as a `ModuleDict` of towers; per-group latent tensors | 0 |
| 2 | Assimilation coupler; per-group forecasting engines and stride dispatch | 1 |
| 3 | Per-group `dim_embed` via coupler projections | 2 |
| 4 | Per-group HEALPix level via nested-ordering pooling and broadcast | 3 |

Phases 0–2 give a runnable coupled model; 3 and 4 are the payoff and are separable. Phase 4 is
where the neighbour table, RoPE coordinates and decoder gather all become per-group, and should
be scoped on its own once 0–3 are in.

Note that the earlier work's hardest problem — making `ae_local_num_queries > 1` correct — does
not arise here. Towers produce separate tensors, so there is no shared query axis to partition
and none of the cell-major/query-minor layout constraints apply.

## 10. Evaluation

Three experiments, in order:

1. **Parity.** Single group, `latent.groups` with one entry, against `develop`. Must match. This
   is the regression guard that the whole refactor rests on.
2. **Coupled skill.** Two towers versus the shared-encoder baseline at matched parameter count,
   on atmospheric and ocean scores. Matched budget matters: two towers will otherwise win on
   capacity alone and tell us nothing.
3. **Coupling ablation.** The diagnostic that justifies the architecture — degrade or withhold the
   ocean input at analysis time and measure the response in atmospheric skill, with the
   assimilation coupler enabled and disabled. If skill does not move when the coupler is on, the
   cross-domain path is not doing its job and the design has failed its main claim.

## 11. Current status

Phases 0-2 are implemented and committed on branch `multiple_latent`, rebased onto `develop`
(`f730876c`): the configuration schema and `validate_latent_groups`, per-group encoder towers
restricted to their own streams, per-group forecasting engines with stride dispatch, and the
assimilation and rollout couplers. `tests/test_latent_groups.py` covers the schema and
`tests/test_latent_groups_model.py` builds the grouped module tree on CPU and pins the stride
dispatch; neither runs a full forward pass, since every attention class calls into flash-attn
unconditionally, so the first real exercise of the towers is a GPU job. See
[`multiple_latents.md`](multiple_latents.md) for the reference and
[`multiple_latents_worklog.md`](multiple_latents_worklog.md) for the log.

Phase 3 is the next step, and it is the one that makes the cost argument in section 6 real:
per-group `dim_embed` is in the schema and honoured by the couplers, but the decoder and the
forecasting engine are not per-group yet, so the model raises `NotImplementedError` on differing
widths and the ocean tower runs at the atmosphere's width. Phase 4 is the payoff and should be
scoped on its own.

The earlier latent-groups work from early September, which partitioned a shared query axis, was
never committed and was lost to a `git restore` on 2026-09-17. It has been superseded rather than
restored; the independent `ae_local_num_queries > 1` fix extracted from it lives on branch
`multi_query_fix`.
