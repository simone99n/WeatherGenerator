# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Construction and rollout dispatch of a latent-group model, on CPU.

tests/test_latent_groups.py covers the configuration contract without building anything. These
tests build the real module tree and exercise the parts of the rollout that are pure tensor
algebra: which engine fires on which output step, the couplers, and the warm-start remap.

A full forward pass is out of reach here. Every attention class calls into flash-attn
unconditionally, and the encoder's local-to-global adapter always holds a cross-attention head,
so no encoder forward can run on CPU. What *can* run is the phase-2 control flow, provided every
block stack is configured to length zero: a ForecastingEngine is then an empty loop and a
CouplingEngine is projections, cat and split. _assert_attention_free below pins that property for
exactly the subtree these tests call, so this file can never reach a flash kernel - note that
pytest.importorskip("flash_attn") would NOT protect it, since flash-attn is installed and imports
fine on machines where it cannot run.
"""

from __future__ import annotations

import pathlib
import tempfile

import pytest
import torch
from omegaconf import OmegaConf, open_dict

import weathergen.common.config as config
from weathergen.model.engines import CouplingEngine, ForecastingEngine, IdentityEngine
from weathergen.model.model import Model
from weathergen.model.model_interface import _remap_latent_group_checkpoint

REPO = pathlib.Path(__file__).resolve().parent.parent
DIM = 32
HEALPIX_LEVEL = 2

# no data is read; the reader never runs, only the schema is needed
PRIVATE_CONF = {"data_paths": ["/nonexistent"], "secrets": {}}

LATENT = {
    "default_group": "fast",
    "groups": {
        # C feeds both towers while being decoded from one: the multi-membership case
        "fast": {
            "streams": ["A", "C"],
            "encoder": {"ae_global_num_blocks": 0},
            "forecast": {"time_step": "06:00:00", "num_blocks": 0, "num_heads": 2},
        },
        "slow": {
            "streams": ["B", "C"],
            "encoder": {"ae_global_num_blocks": 0},
            "forecast": {"time_step": "24:00:00", "num_blocks": 0, "num_heads": 2},
        },
    },
    "coupling": {
        "assimilation": {"num_blocks": 0, "num_heads": 2},
        "rollout": {"time_step": "24:00:00", "num_blocks": 0, "num_heads": 2},
    },
}

# every block stack at zero, so nothing reaches an attention kernel
BLOCKS_OFF = {
    "healpix_level": HEALPIX_LEVEL,
    "ae_local_num_blocks": 0,
    "ae_global_num_blocks": 0,
    "ae_aggregation_num_blocks": 0,
    "fe_num_blocks": 0,
    "rope_2D": False,
    "num_class_tokens": 0,
    "num_register_tokens": 0,
    "ae_local_num_queries": 1,
    "ae_local_dim_embed": DIM,
    "ae_global_dim_embed": DIM,
    "ae_adapter_embed": 8,
    "ae_local_num_heads": 2,
    "ae_global_num_heads": 2,
    "ae_adapter_num_heads": 2,
    "ae_aggregation_num_heads": 2,
    "fe_num_heads": 2,
}


def _stream(group: str) -> dict:
    return {
        "type": "anemoi",
        "filenames": ["unused.zarr"],
        "stream_id": 0,
        "source": ["10u"],
        "target": ["10u"],
        "loss_weight": 1.0,
        "token_size": 4,
        "latent_group": group,
        "embed": {
            "net": "linear",
            "num_tokens": 1,
            "num_heads": 2,
            "dim_embed": DIM,
            "num_blocks": 1,
        },
        "embed_target_coords": {"net": "linear", "dim_embed": DIM},
        "target_readout": {"num_layers": 1, "num_heads": 2},
        "pred_head": {"ens_size": 1, "num_layers": 1},
    }


def _build_config(**overrides):
    """The merged run config, without touching this machine's private configuration."""
    with tempfile.NamedTemporaryFile("w+", suffix=".yml") as private_file:
        private_file.write(OmegaConf.to_yaml(OmegaConf.create(PRIVATE_CONF)))
        private_file.flush()
        cf = config.load_merge_configs(
            pathlib.Path(private_file.name),
            None,
            None,
            REPO / "config" / "default_config.yml",
            BLOCKS_OFF | {"latent": LATENT} | overrides,
        )

    cf = config.set_run_id(cf, "tstgrp00", False)
    with open_dict(cf):
        cf.streams = OmegaConf.create(
            {"A": _stream("fast"), "B": _stream("slow"), "C": _stream("fast")}
        )
        cf.rank, cf.world_size, cf.local_rank = 0, 1, 0
    return cf


def _assert_attention_free(model: Model) -> None:
    """The subtree these tests forward through must hold no attention module.

    Only the forecasting engines and the couplers are checked, because those are the only
    modules called below. The encoder towers are deliberately excluded: their local-to-global
    adapter always holds a MultiCrossAttentionHeadVarlenSlicedQ, which is why no test here runs
    an encoder forward.
    """
    forwarded = list(model.forecast_engines.values()) + [
        model.assimilation_coupler,
        model.rollout_coupler,
    ]
    offenders = [
        f"{type(root).__name__}.{name}: {type(module).__name__}"
        for root in forwarded
        if root is not None
        for name, module in root.named_modules()
        if "Attention" in type(module).__name__
    ]
    assert not offenders, f"forwarded subtree is not attention-free: {offenders}"


@pytest.fixture(scope="module")
def grouped_model():
    with torch.device("cpu"):
        model = Model(_build_config(), [1, 1, 1], [1, 1, 1], [3, 3, 3]).create()
    _assert_attention_free(model)
    return model


def _latents(model: Model) -> dict[str, torch.Tensor]:
    """Per-group latents with deliberately unequal token counts."""
    return {
        "fast": torch.zeros(2, 7, model.latent_group_dims["fast"]),
        "slow": torch.ones(2, 5, model.latent_group_dims["slow"]),
    }


# --------------------------------------------------------------------------- construction


def test_one_tower_and_one_engine_per_group(grouped_model):
    assert list(grouped_model.encoders.keys()) == ["fast", "slow"]
    assert list(grouped_model.forecast_engines.keys()) == ["fast", "slow"]
    assert all(
        isinstance(engine, ForecastingEngine) for engine in grouped_model.forecast_engines.values()
    )
    # the single-latent attributes are stood down rather than left half-built
    assert grouped_model.encoder is None
    assert isinstance(grouped_model.forecast_engine, IdentityEngine)


def test_towers_see_only_their_own_streams(grouped_model):
    assert grouped_model.encoders["fast"].stream_names == ["A", "C"]
    assert grouped_model.encoders["slow"].stream_names == ["B", "C"]
    # the restriction is by position on the stream axis of batch.tokens_lens
    assert grouped_model.encoders["fast"].stream_idxs == [0, 2]
    assert grouped_model.encoders["slow"].stream_idxs == [1, 2]


def test_towers_share_no_parameters(grouped_model):
    fast = {id(p) for p in grouped_model.encoders["fast"].parameters()}
    slow = {id(p) for p in grouped_model.encoders["slow"].parameters()}

    assert fast and slow
    assert not fast & slow


def test_both_couplers_exist_with_separate_weights(grouped_model):
    assert isinstance(grouped_model.assimilation_coupler, CouplingEngine)
    assert isinstance(grouped_model.rollout_coupler, CouplingEngine)
    assert grouped_model.assimilation_coupler is not grouped_model.rollout_coupler

    assimilation = {id(p) for p in grouped_model.assimilation_coupler.parameters()}
    rollout = {id(p) for p in grouped_model.rollout_coupler.parameters()}
    assert not assimilation & rollout


def test_derived_cadences(grouped_model):
    assert grouped_model.latent_group_strides == {"fast": 1, "slow": 4}
    assert grouped_model.coupling_stride == 4
    assert grouped_model.stream_latent_groups == {"A": "fast", "B": "slow", "C": "fast"}


def test_a_group_with_no_forecast_block_gets_no_engine():
    """An engine whose stride never fires would hold parameters without gradient."""
    latent = OmegaConf.to_container(OmegaConf.create(LATENT))
    latent["groups"]["slow"].pop("forecast")
    latent["coupling"].pop("rollout")

    with torch.device("cpu"):
        model = Model(_build_config(latent=latent), [1, 1, 1], [1, 1, 1], [3, 3, 3]).create()

    assert list(model.encoders.keys()) == ["fast", "slow"]
    assert list(model.forecast_engines.keys()) == ["fast"]
    assert model.rollout_coupler is None


def test_differing_group_widths_are_rejected():
    """Per-group dim_embed is phase 3; the decoder and forecast engine are not per-group yet."""
    latent = OmegaConf.to_container(OmegaConf.create(LATENT))
    latent["groups"]["slow"]["dim_embed"] = DIM // 2
    latent["coupling"]["dim_embed"] = DIM

    with pytest.raises(NotImplementedError):
        with torch.device("cpu"):
            Model(_build_config(latent=latent), [1, 1, 1], [1, 1, 1], [3, 3, 3]).create()


# --------------------------------------------------------------------------- rollout dispatch


def _record_calls(model: Model) -> tuple[list[str], list]:
    """Forward hooks on every engine, so the dispatch is observed rather than recomputed."""
    fired: list[str] = []
    handles = []
    for name, engine in model.forecast_engines.items():
        handles.append(
            engine.register_forward_hook(
                lambda _m, _i, _o, name=name: fired.append(f"engine:{name}")
            )
        )
    for name, coupler in (
        ("assimilation", model.assimilation_coupler),
        ("rollout", model.rollout_coupler),
    ):
        handles.append(
            coupler.register_forward_hook(
                lambda _m, _i, _o, name=name: fired.append(f"coupler:{name}")
            )
        )
    return fired, handles


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        (1, ["engine:fast"]),
        (2, ["engine:fast"]),
        (3, ["engine:fast"]),
        (4, ["engine:fast", "engine:slow", "coupler:rollout"]),
        (5, ["engine:fast"]),
        (8, ["engine:fast", "engine:slow", "coupler:rollout"]),
    ],
)
def test_only_the_groups_whose_stride_fires_are_advanced(grouped_model, step, expected):
    fired, handles = _record_calls(grouped_model)
    try:
        grouped_model.advance_latent(_latents(grouped_model), step)
    finally:
        for handle in handles:
            handle.remove()

    assert fired == expected


def test_a_held_group_is_returned_unchanged(grouped_model):
    """The slow tower is held constant between the steps where its engine fires."""
    latents = _latents(grouped_model)
    advanced = grouped_model.advance_latent(dict(latents), step=1)

    assert advanced["slow"] is latents["slow"]
    assert set(advanced) == {"fast", "slow"}
    for name, tokens in advanced.items():
        assert tokens.shape == latents[name].shape


def test_couplers_preserve_each_group_shape(grouped_model):
    latents = _latents(grouped_model)
    coupled = grouped_model.assimilation_coupler(latents)

    assert {name: tuple(t.shape) for name, t in coupled.items()} == {
        name: tuple(t.shape) for name, t in latents.items()
    }


def test_concat_latent_groups_follows_configuration_order(grouped_model):
    latents = _latents(grouped_model)
    concatenated = grouped_model.concat_latent_groups(latents)

    assert concatenated.shape == (2, 7 + 5, DIM)
    # fast first, as declared under latent.groups, not in dict-insertion order
    assert torch.equal(concatenated[:, :7], latents["fast"])
    assert torch.equal(concatenated[:, 7:], latents["slow"])


# --------------------------------------------------------------------------- warm start


def test_single_latent_checkpoint_is_remapped_onto_the_default_group(grouped_model):
    params = {
        "encoder.q_cells": torch.zeros(1),
        "encoder.ae_global_engine.blocks.0.weight": torch.zeros(1),
        "forecast_engine.fe_blocks.0.weight": torch.zeros(1),
        "embed_target_coords.A.weight": torch.zeros(1),
    }

    remapped = _remap_latent_group_checkpoint(grouped_model, params)

    assert "encoders.fast.q_cells" in remapped
    assert "encoders.fast.ae_global_engine.blocks.0.weight" in remapped
    assert "forecast_engines.fast.fe_blocks.0.weight" in remapped
    # keys outside the encoder and the engine are untouched, and nothing single-latent survives
    assert "embed_target_coords.A.weight" in remapped
    assert not [key for key in remapped if key.startswith(("encoder.", "forecast_engine."))]


def test_an_already_grouped_checkpoint_is_left_alone(grouped_model):
    params = {"encoders.slow.q_cells": torch.zeros(1)}

    assert _remap_latent_group_checkpoint(grouped_model, params) == params
