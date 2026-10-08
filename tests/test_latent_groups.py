# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Validation of the latent group schema.

Latent groups are per-component encoder towers: each has its own encoder, its own latent
tensor and its own dynamical time step. Two orthogonal relations tie streams to groups:

  input membership   latent.groups.<g>.streams   a stream may feed several towers
  decode target      streams.<s>.latent_group    a stream is predicted from exactly one

These tests define the contract implemented by phases 0-2 of
docs/multi_time_latents_proposal.md; see docs/multiple_latents.md for the reference.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from weathergen.common.config import (
    get_latent_coupling_stride,
    get_latent_group_dims,
    get_latent_group_streams,
    get_latent_group_strides,
    get_latent_groups,
    get_stream_latent_group,
    validate_latent_groups,
)

MODE = "training_config"


def make_config(**overrides):
    """Two towers: a fast atmosphere and a slower, narrower ocean."""
    config = OmegaConf.create(
        {
            "num_register_tokens": 0,
            "num_class_tokens": 0,
            "ae_global_dim_embed": 2048,
            "latent": {
                "default_group": "atmosphere",
                "groups": {
                    "atmosphere": {
                        "streams": ["ERA5"],
                        "dim_embed": 2048,
                        "forecast": {"time_step": "06:00:00", "num_blocks": 16},
                    },
                    "ocean": {
                        "streams": ["C-GLORS"],
                        "dim_embed": 1024,
                        "forecast": {"time_step": "24:00:00", "num_blocks": 8},
                    },
                },
                "coupling": {
                    "dim_embed": 2048,
                    "assimilation": {"num_blocks": 2, "num_heads": 32},
                    "rollout": {"time_step": "24:00:00", "num_blocks": 1, "num_heads": 32},
                },
            },
            "streams": {
                "ERA5": {"latent_group": "atmosphere"},
                "C-GLORS": {"latent_group": "ocean"},
            },
        }
    )
    return OmegaConf.merge(config, OmegaConf.create(overrides))


def make_mode_config(num_steps=4, offset=1, time_step="06:00:00"):
    return OmegaConf.create(
        {
            "forecast": {
                "time_step": time_step,
                "offset": offset,
                "num_steps": num_steps,
                "policy": "fixed",
            }
        }
    )


# --------------------------------------------------------------------------- happy path


def test_valid_configuration_passes():
    validate_latent_groups(make_config(), make_mode_config(), MODE)


def test_derived_strides_and_dims():
    config = make_config()
    forecast_cfg = make_mode_config().forecast

    assert get_latent_group_strides(config, forecast_cfg) == {"atmosphere": 1, "ocean": 4}
    assert get_latent_coupling_stride(config, forecast_cfg) == 4
    assert get_latent_group_dims(config) == {"atmosphere": 2048, "ocean": 1024}


def test_group_dim_falls_back_to_the_global_width():
    config = make_config()
    config.latent.groups.ocean.pop("dim_embed")

    assert get_latent_group_dims(config)["ocean"] == 2048


def test_single_latent_is_unaffected():
    config = OmegaConf.create({"latent": {"groups": None}, "streams": {"ERA5": {}}})

    assert get_latent_groups(config) is None
    assert get_latent_group_streams(config) == {}
    validate_latent_groups(config, make_mode_config(), MODE)


# ------------------------------------------------------------------- stream <-> group


def test_input_membership_and_decode_target_are_separate():
    config = make_config()

    assert get_latent_group_streams(config) == {
        "atmosphere": ["ERA5"],
        "ocean": ["C-GLORS"],
    }
    assert get_stream_latent_group(config, config.streams["C-GLORS"]) == "ocean"


def test_a_stream_may_feed_several_groups():
    """Radiances and surface obs constrain more than one component."""
    config = make_config()
    config.streams["SYNOP"] = {"latent_group": "atmosphere"}
    config.latent.groups.atmosphere.streams = ["ERA5", "SYNOP"]
    config.latent.groups.ocean.streams = ["C-GLORS", "SYNOP"]

    validate_latent_groups(config, make_mode_config(), MODE)

    streams = get_latent_group_streams(config)
    assert "SYNOP" in streams["atmosphere"]
    assert "SYNOP" in streams["ocean"]
    # it still decodes from exactly one
    assert get_stream_latent_group(config, config.streams["SYNOP"]) == "atmosphere"


def test_decode_target_falls_back_to_default_group():
    config = make_config()
    config.streams.ERA5.pop("latent_group")

    assert get_stream_latent_group(config, config.streams.ERA5) == "atmosphere"


def test_diagnostic_stream_need_not_feed_its_decode_group():
    """A target-only stream is decoded from a tower without being one of its inputs."""
    config = make_config()
    config.streams["SYNOP"] = {"latent_group": "atmosphere", "diagnostic": True}

    validate_latent_groups(config, make_mode_config(), MODE)


def test_stream_group_without_groups_is_rejected():
    config = OmegaConf.create(
        {"latent": {"groups": None}, "streams": {"ERA5": {"latent_group": "ocean"}}}
    )

    with pytest.raises(ValueError, match="latent.groups"):
        validate_latent_groups(config, make_mode_config(), MODE)


def test_unknown_decode_group_is_rejected():
    config = make_config()
    config.streams["C-GLORS"].latent_group = "stratosphere"

    with pytest.raises(ValueError, match="unknown latent group"):
        validate_latent_groups(config, make_mode_config(), MODE)


def test_unknown_input_stream_is_rejected():
    config = make_config()
    config.latent.groups.ocean.streams = ["C-GLORS", "NEMO"]

    with pytest.raises(ValueError, match="NEMO"):
        validate_latent_groups(config, make_mode_config(), MODE)


def test_group_without_input_streams_is_rejected():
    config = make_config()
    config.latent.groups.ocean.streams = []

    with pytest.raises(ValueError, match="no input streams"):
        validate_latent_groups(config, make_mode_config(), MODE)


def test_shared_prediction_head_across_groups_is_rejected():
    config = make_config()
    config.streams["C-GLORS"].pred_spatial_shared = "ERA5"

    with pytest.raises(ValueError, match="different latent groups"):
        validate_latent_groups(config, make_mode_config(), MODE)


# ------------------------------------------------------------------------- cadence


def test_time_step_must_be_a_multiple_of_the_base_step():
    config = make_config()
    config.latent.groups.ocean.forecast.time_step = "09:00:00"

    with pytest.raises(ValueError, match="whole multiple"):
        validate_latent_groups(config, make_mode_config(), MODE)


def test_rollout_too_short_for_the_slowest_group_is_rejected():
    # output_idxs is range(1, 4) == [1, 2, 3], so a stride of 4 never fires
    with pytest.raises(ValueError, match="would never run"):
        validate_latent_groups(make_config(), make_mode_config(num_steps=3), MODE)


def test_shortest_rollout_of_a_curriculum_is_used():
    # num_steps may vary per mini-epoch; the shortest one has to fire every group
    with pytest.raises(ValueError, match="would never run"):
        validate_latent_groups(make_config(), make_mode_config(num_steps=[2, 4, 8]), MODE)


def test_group_without_forecast_block_is_never_advanced():
    config = make_config()
    config.latent.groups.ocean.pop("forecast")

    validate_latent_groups(config, make_mode_config(num_steps=2), MODE)
    assert "ocean" not in get_latent_group_strides(config, make_mode_config().forecast)


def test_sexagesimal_yaml_time_steps_are_read_as_durations():
    """PyYAML parses 24:00:00 as the integer 86400, which must still mean 24 hours."""
    config = OmegaConf.create(
        """
        ae_global_dim_embed: 2048
        num_register_tokens: 0
        num_class_tokens: 0
        latent:
          default_group: atmosphere
          groups:
            atmosphere:
              streams: [ERA5]
              forecast: {time_step: 06:00:00}
            ocean:
              streams: [C-GLORS]
              forecast: {time_step: 24:00:00}
          coupling: null
        streams: {ERA5: {}, C-GLORS: {latent_group: ocean}}
        """
    )

    assert get_latent_group_strides(config, make_mode_config().forecast) == {
        "atmosphere": 1,
        "ocean": 4,
    }


# ------------------------------------------------------------------------- coupling


def test_mixed_widths_require_a_coupling_width():
    """Towers of different width can only be mixed through a shared projection."""
    config = make_config()
    config.latent.coupling.pop("dim_embed")

    with pytest.raises(ValueError, match="coupling.dim_embed"):
        validate_latent_groups(config, make_mode_config(), MODE)


def test_uniform_widths_need_no_coupling_width():
    config = make_config()
    config.latent.groups.ocean.dim_embed = 2048
    config.latent.coupling.pop("dim_embed")

    validate_latent_groups(config, make_mode_config(), MODE)


def test_coupling_can_be_disabled_entirely():
    config = make_config()
    config.latent.groups.ocean.dim_embed = 2048
    config.latent.coupling = None

    validate_latent_groups(config, make_mode_config(), MODE)
    assert get_latent_coupling_stride(config, make_mode_config().forecast) is None


def test_rope_2d_is_rejected_with_latent_groups():
    """The couplers attend over the concatenated token axis, which carries no coordinates."""
    config = make_config(rope_2D=True)

    with pytest.raises(ValueError, match="rope_2D"):
        validate_latent_groups(config, make_mode_config(), MODE)
