# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""analysis_window_step: which windows may start a sample.

The coupled ERA5 + C-GLORS configuration needs 6 h windows, so that the +6/+12/+18 h targets
exist, but only the 00 UTC windows hold a daily ocean record. Setting time_window_step to 24 h
instead makes steps 1-3 target the analysis window itself, because the target window is located
by integer division on the window grid. analysis_window_step restricts where a sample starts
without coarsening that grid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from weathergen.common.config import _check_time_interpolation, _sanitize_time_keys
from weathergen.datasets.data_reader_base import TimeIndexRange, TimeWindowHandler
from weathergen.datasets.multi_stream_data_sampler import (
    MultiStreamDataSampler,
    analysis_window_mask,
    get_analysis_window_step,
)
from weathergen.train.utils import get_active_stage_config

H = np.timedelta64(1, "h").astype("timedelta64[ms]")
COUPLED_CONFIG = Path(__file__).parent.parent / "config" / "config_forecasting_coupled.yml"


def _handler(start: str, end: str = "1993-03-01T00:00") -> TimeWindowHandler:
    return TimeWindowHandler(np.datetime64(start), np.datetime64(end), 6 * H, 6 * H)


def _window_hours(tw: TimeWindowHandler, idxs) -> set[int]:
    return {int(tw.window(i).start.astype("datetime64[h]").astype(int) % 24) for i in idxs}


# --------------------------------------------------------------------------- the mask


def test_no_analysis_step_keeps_every_window():
    idxs = np.arange(40)
    assert analysis_window_mask(idxs, _handler("1993-01-01T00:00"), None).all()


def test_daily_analysis_step_keeps_the_00_utc_windows():
    tw = _handler("1993-01-01T00:00")
    idxs = np.arange(40)
    kept = idxs[analysis_window_mask(idxs, tw, 24 * H)]

    np.testing.assert_array_equal(kept, np.arange(0, 40, 4))
    assert _window_hours(tw, kept) == {0}


def test_alignment_is_to_utc_not_to_start_date():
    # a stage starting at 06 UTC must still start its samples at 00 UTC, where the daily
    # record is, not every fourth window counted from its own start
    tw = _handler("1993-01-01T06:00")
    idxs = np.arange(40)
    kept = idxs[analysis_window_mask(idxs, tw, 24 * H)]

    np.testing.assert_array_equal(kept, np.arange(3, 40, 4))
    assert _window_hours(tw, kept) == {0}


def test_twelve_hourly_analysis_step():
    tw = _handler("1993-01-01T00:00")
    idxs = np.arange(40)
    kept = idxs[analysis_window_mask(idxs, tw, 12 * H)]

    assert _window_hours(tw, kept) == {0, 12}
    assert kept.size == 20


# --------------------------------------------------------------------------- the option


def _stage(**keys):
    """A training_config whose durations are resolved the way the config loader resolves them."""
    return _sanitize_time_keys(OmegaConf.create({"training_config": keys})).training_config


def test_option_absent_means_no_restriction():
    assert get_analysis_window_step(_stage(time_window_step="06:00:00")) is None


def test_option_is_read_as_a_duration():
    stage = _stage(time_window_step="06:00:00", analysis_window_step="24:00:00")
    assert get_analysis_window_step(stage) == 24 * H


@pytest.mark.parametrize("analysis_step", ["09:00:00", "00:00:00", "-24:00:00"])
def test_option_must_be_a_positive_multiple_of_the_window_step(analysis_step):
    stage = _stage(time_window_step="06:00:00", analysis_window_step=analysis_step)
    with pytest.raises(ValueError, match="analysis_window_step"):
        get_analysis_window_step(stage)


# --------------------------------------------------------------------------- the sampler


def _bare_sampler(start: str, end: str, analysis_step, samples: int = 64):
    """A sampler carrying only what check_samples and _calc_baseperms read.

    Building a real one opens every stream's store; these two methods only do index arithmetic.
    """
    tw = TimeWindowHandler(np.datetime64(start), np.datetime64(end), 6 * H, 6 * H)
    sampler = object.__new__(MultiStreamDataSampler)
    sampler.time_window_handler = tw
    sampler.index_range = tw.get_index_range()
    sampler.analysis_step = analysis_step
    sampler.time_step = 6 * H
    sampler.output_offset = 1
    sampler.len_timedelta = 6 * H
    sampler.step_timedelta = 6 * H
    sampler.max_input_steps = 1
    sampler.batch_size = 1
    sampler.world_size = 1
    sampler.repeat_data = False
    sampler.samples_per_mini_epoch = samples
    return sampler


def test_perms_contain_only_00_utc_windows():
    sampler = _bare_sampler("1993-01-01T00:00", "1993-03-01T00:00", 24 * H)
    perms = sampler._calc_baseperms(fsm=4)

    assert perms.size > 0
    assert _window_hours(sampler.time_window_handler, perms) == {0}
    # the last sample still reaches its step-4 target inside the date range
    last_target = sampler.time_window_handler.window(perms.max() + 4)
    assert last_target.end <= sampler.time_window_handler.t_end


def test_perms_unchanged_without_the_option():
    sampler = _bare_sampler("1993-01-01T00:00", "1993-03-01T00:00", None)
    perms = sampler._calc_baseperms(fsm=4)

    idx_range = sampler.index_range
    np.testing.assert_array_equal(perms, np.arange(1, idx_range.end - idx_range.start - 5))


def test_available_samples_count_only_analysis_windows():
    # 59 days: 236 windows, of which 59 start at 00 UTC; asking for 100 samples must therefore
    # be cut down when anchored, and fits the 230 windows that leave room for four steps when not
    anchored = _bare_sampler("1993-01-01T00:00", "1993-03-01T00:00", 24 * H, samples=100)
    anchored.check_samples(fsm=4)
    assert anchored.samples_per_mini_epoch < 59

    free = _bare_sampler("1993-01-01T00:00", "1993-03-01T00:00", None, samples=100)
    free.check_samples(fsm=4)
    assert free.samples_per_mini_epoch == 100


def test_a_range_without_analysis_windows_fails_loudly():
    # two days: the 00 UTC windows are 0 and 4, but with four steps only 1 and 2 can start one
    sampler = _bare_sampler("1993-01-01T00:00", "1993-01-03T00:00", 24 * H)
    with pytest.raises(AssertionError, match="analysis window"):
        sampler._calc_baseperms(fsm=4)


# --------------------------------------------------------------------------- the config


def test_coupled_config_starts_every_stage_at_00_utc():
    conf = _sanitize_time_keys(OmegaConf.load(COUPLED_CONFIG))
    training = conf.training_config
    validation = get_active_stage_config(training, conf.validation_config, [])
    test = get_active_stage_config(validation, conf.test_config, [])

    for stage in (training, validation, test):
        assert stage.time_window_step == 6 * H
        assert stage.time_window_len == 6 * H
        assert get_analysis_window_step(stage) == 24 * H
        # the target grid must stay fine enough for the 6 h forecast step
        assert stage.forecast.time_step % stage.time_window_step == np.timedelta64(0, "ms")


def test_option_survives_the_config_round_trip():
    conf = _sanitize_time_keys(
        OmegaConf.create({"training_config": {"analysis_window_step": "24:00:00"}})
    )
    # what gets written back to disk is the original string, not a resolved timedelta
    assert _check_time_interpolation(conf).training_config.analysis_window_step == "24:00:00"


def test_time_index_range_type_is_what_the_sampler_expects():
    # guards the bare sampler above against drifting from the real handler
    assert isinstance(_handler("1993-01-01T00:00").get_index_range(), TimeIndexRange)
