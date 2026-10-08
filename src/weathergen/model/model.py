# ruff: noqa: T201
# (C) Copyright 2025 WeatherGenerator contributors.

#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import math
import typing
import warnings

import astropy_healpix as hp
import astropy_healpix.healpy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from weathergen.common.config import (
    Config,
    get_latent_coupling_stride,
    get_latent_group_dims,
    get_latent_group_streams,
    get_latent_group_strides,
    get_latent_groups,
    get_stream_latent_group,
)
from weathergen.datasets.batch import ModelBatch
from weathergen.datasets.utils import healpix_verts_rots, r3tos2
from weathergen.model.encoder import EncoderModule
from weathergen.model.engines import (
    BilinearDecoder,
    CouplingEngine,
    EnsPredictionHead,
    ForecastingEngine,
    IdentityEngine,
    LatentPredictionHeadIdentity,
    LatentPredictionHeadMLP,
    LatentPredictionHeadTransformer,
    LatentState,
    TargetPredictionEngine,
    TargetPredictionEngineClassic,
)
from weathergen.model.layers import MLP, NamedLinear
from weathergen.model.norms import AdaLayerNorm
from weathergen.model.utils import get_num_parameters
from weathergen.utils.distributed import is_root
from weathergen.utils.utils import get_dtype, is_stream_forcing

logger = logging.getLogger(__name__)

type StreamName = str


class ModelOutput:
    """
    Representation of model output
    """

    physical: list[dict[StreamName, torch.Tensor]]
    latent: list[dict[str, torch.Tensor | LatentState]]

    def __init__(self, len_output: int) -> None:
        self.physical = [{} for _ in range(len_output)]
        self.latent = [{} for _ in range(len_output)]

    def add_physical_prediction(
        self, fstep: int, stream_name: StreamName, pred: torch.Tensor
    ) -> None:
        self.physical[fstep][stream_name] = pred

    def add_latent_prediction(self, fstep: int, latent_name: str, pred: torch.Tensor) -> None:
        self.latent[fstep][latent_name] = pred

    def get_physical_prediction(
        self, fstep: int, stream_name: StreamName | None = None, sample_idx: int | None = None
    ):
        pred = self.physical[fstep]
        if stream_name is not None:
            pred = pred.get(stream_name, None)
            if sample_idx is not None:
                assert sample_idx < len(pred), "Invalid sample index."
                pred = pred[sample_idx]
        return pred

    def get_latent_prediction(self, fstep: int):
        return self.latent[fstep]


class ModelParams(torch.nn.Module):
    """Creation of query and embedding parameters of the model."""

    def __init__(self, cf) -> None:
        super(ModelParams, self).__init__()

        self.cf = cf

        self.healpix_level = cf.healpix_level
        self.num_healpix_cells = 12 * 4**cf.healpix_level
        self.dtype = get_dtype(cf.attention_dtype)

        # Positional embeddings
        self.max_tokens_local_per_cell = cf.get("ae_local_max_tokens_per_cell", 64)
        self.pe_embed = torch.nn.Parameter(
            torch.zeros(self.max_tokens_local_per_cell, cf.ae_local_dim_embed, dtype=self.dtype),
            requires_grad=False,
        )

        pe = torch.zeros(
            self.num_healpix_cells,
            cf.ae_local_num_queries,
            cf.ae_global_dim_embed,
            dtype=self.dtype,
        )
        self.pe_global = torch.nn.Parameter(pe, requires_grad=False)

        # RoPE coordinates
        self.rope_2D = cf.get("rope_2D", False)
        if self.rope_2D:
            self.num_extra_tokens = cf.num_register_tokens + cf.num_class_tokens
            total_tokens = (
                self.num_healpix_cells + self.num_extra_tokens
            ) * cf.ae_local_num_queries
            self.register_buffer(
                "rope_coords",
                torch.zeros(
                    1,
                    total_tokens,
                    2,
                    dtype=self.dtype,
                ),
            )
            self.register_buffer(
                "rope_cell_coords",
                torch.zeros(
                    self.num_healpix_cells,
                    2,
                    dtype=self.dtype,
                ),
            )
        else:
            self.rope_coords = None
            self.rope_cell_coords = None

        # HEALPix neighbours
        hlc = self.healpix_level
        with warnings.catch_warnings(action="ignore"):
            temp = hp.neighbours(
                np.arange(self.num_healpix_cells), 2**hlc, order="nested"
            ).transpose()
        # fix missing nbors with references to self
        for i, row in enumerate(temp):
            temp[i][row == -1] = i
        self.hp_nbours = torch.nn.Parameter(
            torch.empty((temp.shape[0], (temp.shape[1] + 1)), dtype=torch.int32),
            requires_grad=False,
        )

        self.q_cells_lens = torch.nn.Parameter(
            torch.ones(self.num_healpix_cells + 1, dtype=torch.int32), requires_grad=False
        )
        self.q_cells_lens.data[0] = 0

    def create(self, cf: Config) -> "ModelParams":
        self.reset_parameters(cf)
        return self

    def reset_parameters(self, cf: Config) -> "ModelParams":
        """Creates positional embedding for each grid point for each stream used after stream
        embedding, positional embedding for all stream assimilated cell-level local embedding,
        initializing queries for local-to-global adapters, HEALPix neighbourhood based parameter
        initializing for target prediction.

        Sinusoidal positional encoding: Harmonic positional encoding based upon sine and cosine for
            both per stream after stream embedding and per cell level for local assimilation.

        HEALPix neighbourhood structure: Determine the neighbors for each cell and initialize each
            with its own cell number as well as the cell numbers of its neighbors. If a cell has
            fewer than eight neighbors, use its own cell number to fill the remaining slots.

        Query len based parameter creation: Calculate parameters for the calculated token length at
            each cell after local assimilation.

        Args:
            cf : Configuration
        """

        # positional encodings

        dim_embed = cf.ae_local_dim_embed
        token_idx_bias = 16
        freq_bias = 8
        self.pe_embed.data.fill_(0.0)
        position = torch.arange(
            token_idx_bias,
            token_idx_bias + self.max_tokens_local_per_cell,
            device=self.pe_embed.device,
        ).unsqueeze(1)
        div = torch.exp(
            torch.arange(freq_bias, freq_bias + dim_embed, 2, device=self.pe_embed.device)
            * -(math.log(self.max_tokens_local_per_cell) / dim_embed),
        )
        self.pe_embed.data[:, 0::2] = torch.sin(position * div[: self.pe_embed[:, 0::2].shape[1]])
        self.pe_embed.data[:, 1::2] = torch.cos(position * div[: self.pe_embed[:, 1::2].shape[1]])

        dim_embed = cf.ae_global_dim_embed

        if self.rope_2D:
            # Precompute per-cell center coordinates (lat, lon in radians) for 2D RoPE.
            # Shape: (num_healpix_cells, ae_local_num_queries, 2)
            verts, _ = healpix_verts_rots(self.healpix_level, 0.5, 0.5)
            coords = r3tos2(verts.to(self.rope_coords.device)).to(self.rope_coords.dtype)
            # Per-cell coords for QueryAggregationEngine (no query expansion)
            self.rope_cell_coords.data.copy_(coords)
            coords = coords.unsqueeze(1).repeat(1, cf.ae_local_num_queries, 1)
            coords_flat = coords.flatten(0, 1).unsqueeze(0)
            offset = self.num_extra_tokens * cf.ae_local_num_queries
            self.rope_coords.data.fill_(0.0)
            self.rope_coords.data[:, offset : offset + coords_flat.shape[1], :].copy_(coords_flat)

        # pe_global: always initialized. RoPE handles relative position in Q/K, but pe_global
        # provides per-cell token identity which is critical for masked cells that have no
        # content from local assimilation. Without it, masked cells are identical and the
        # teacher representation (evaluated without dropout) collapses to low rank.
        self.pe_global.data.fill_(0.0)
        xs = 2.0 * np.pi * torch.arange(0, dim_embed, 2, device=self.pe_global.device) / dim_embed
        self.pe_global.data[..., 0::2] = 0.5 * torch.sin(
            torch.outer(8 * torch.arange(cf.ae_local_num_queries, device=self.pe_global.device), xs)
        )
        self.pe_global.data[..., 0::2] += (
            torch.sin(
                torch.outer(torch.arange(self.num_healpix_cells, device=self.pe_global.device), xs)
            )
            .unsqueeze(1)
            .repeat((1, cf.ae_local_num_queries, 1))
        )
        self.pe_global.data[..., 1::2] = 0.5 * torch.cos(
            torch.outer(8 * torch.arange(cf.ae_local_num_queries, device=self.pe_global.device), xs)
        )
        self.pe_global.data[..., 1::2] += (
            torch.cos(
                torch.outer(torch.arange(self.num_healpix_cells, device=self.pe_global.device), xs)
            )
            .unsqueeze(1)
            .repeat((1, cf.ae_local_num_queries, 1))
        )

        # healpix neighborhood structure

        hlc = self.healpix_level
        num_healpix_cells = self.num_healpix_cells
        with warnings.catch_warnings(action="ignore"):
            temp = hp.neighbours(np.arange(num_healpix_cells), 2**hlc, order="nested").transpose()
        # fix missing nbors with references to self
        for i, row in enumerate(temp):
            temp[i][row == -1] = i
        # nbors *and* self
        self.hp_nbours.data[:, 0] = torch.arange(temp.shape[0], device=self.hp_nbours.device)
        self.hp_nbours.data[:, 1:] = torch.from_numpy(temp).to(self.hp_nbours.device)

        # precompute for varlen attention
        self.q_cells_lens.data.fill_(1)
        self.q_cells_lens.data[0] = 0

        # ensure all params have grad set to False

        return


class Model(torch.nn.Module):
    """WeatherGenerator model architecture

    WeatherGenerator consists of the following components:

    embeds: embedding networks: Stream specific embedding networks.

    ae_local_blocks: Local assimilation engine: transformer based network to combine different input
        streams per healpix cell.

    ae_adapter: Assimilation engine adapter: Adapter to transform local assimilation engine
        information to the global assimilation engine.

    ae_aggregation_blocks: Query aggregation engine: after the learnable queries are created per
        non-masked healpix cell, this engine combines information from all non-masked cells by
        using dense attention layers.

    ae_global_blocks: Global assimilation engine: Transformer network alternating between local and
        global attention based upon global attention density rate.

    fe_blocks: Forecasting engine: Transformer network using the output of global attention to
        advance the latent representation in time.

    embed_target_coords: Embedding networks for coordinates: Initializes embedding networks tailored
        for metadata embedded target coordinates. The architecture is either a linear layer or a
        multi-layer perceptron, determined by the configuration of the embedding target coordinate
        networks.

    pred_adapter_kv: Prediction adapter: Adapter to transform the global assimilation/forecasting
        engine output to the prediction engine. Uses an MLP if `cf.pred_adapter_kv` is True,
        otherwise it uses an identity function.

    target_token_engines: Prediction engine: Transformer based prediction network that generates
        output corresponding to target coordinates.

    pred_heads: Prediction head: Final layers using target token engines output for mapping target
        coordinates to its physical space.
    """

    def __init__(self, cf: Config, sources_size, targets_num_channels, targets_coords_size):
        """
        Args:
            cf : Configuration with model parameters
            sources_size : List of number of channels for models
            targets_num_channels : List with size of each output sample for coordinates target
                embedding
            targets_coords_size : List with size of each input sample for coordinates target
                embedding
        """
        super(Model, self).__init__()

        self.healpix_level = cf.healpix_level
        self.num_healpix_cells = 12 * 4**self.healpix_level

        self.cf = cf
        self.dtype = get_dtype(self.cf.attention_dtype)
        self.sources_size = sources_size
        self.targets_num_channels = targets_num_channels
        self.targets_coords_size = targets_coords_size

        self.embed_target_coords = None
        self.encoder: EncoderModule | None = None
        self.forecast_engine: ForecastingEngine | IdentityEngine | None = None
        self.pred_heads = None
        self.q_cells: torch.Tensor | None = None
        self.streams: dict[str, typing.Any] = cf.streams
        self.target_token_engines = None

        # Latent groups: per-component encoder towers, each with its own latent tensor and its
        # own dynamical time step. None keeps the single encoder and self.forecast_engine.
        self.latent_groups = get_latent_groups(cf)
        self.latent_group_streams = get_latent_group_streams(cf)
        self.latent_group_dims = get_latent_group_dims(cf)
        forecast_cfg = cf.training_config.get("forecast", {})
        self.latent_group_strides = get_latent_group_strides(cf, forecast_cfg)
        self.coupling_stride = get_latent_coupling_stride(cf, forecast_cfg)
        self.stream_latent_groups = {
            name: get_stream_latent_group(cf, si) for name, si in self.streams.items()
        }
        self.encoders = None
        self.forecast_engines = None
        self.assimilation_coupler = None
        self.rollout_coupler = None

        assert cf.get("forecast", {}).get("att_dense_rate", 1.0) == 1.0, (
            "Local attention not adapted for register tokens"
        )
        self.num_register_tokens = cf.num_register_tokens
        self.latent_heads = None
        self.latent_pre_norm = None
        # auxiliary tokens
        self.class_token_idxs = list(
            range(cf.num_register_tokens, cf.num_register_tokens + cf.num_class_tokens)
        )
        self.register_token_idxs = list(range(cf.num_register_tokens))
        self.aux_token_idxs = list(range(cf.num_register_tokens + cf.num_class_tokens))
        self.num_aux_tokens = cf.num_register_tokens + cf.num_class_tokens

    def _create_latent_pred_head(
        self, global_cfg, name, loss_cfg, use_class_token, use_patch_token
    ):
        if loss_cfg["head"].lower() == "mlp":
            return LatentPredictionHeadMLP(
                name,
                global_cfg.ae_global_dim_embed,
                loss_cfg,
                use_class_token=use_class_token,
                use_patch_token=use_patch_token,
            )
        elif loss_cfg["head"].lower() == "transformer":
            return LatentPredictionHeadTransformer(
                global_cfg,
                name,
                global_cfg.ae_global_dim_embed,
                loss_cfg,
                use_class_token=use_class_token,
                use_patch_token=use_patch_token,
            )
        elif loss_cfg["head"].lower() == "identity":
            return LatentPredictionHeadIdentity()
        else:
            assert False, f"Unknown latent prediction head type {loss_cfg['head']}"

    def _group_config(self, cf: Config, group) -> Config:
        """Config for one latent group's tower, with its encoder and fe_* keys overridden.

        Args:
            cf : Configuration
            group : the latent.groups entry for this group
        Returns:
            A shallow copy of cf describing this group's encoder and forecasting engine.
        """
        group_cf = cf.copy()

        for key, value in (group.get("encoder") or {}).items():
            group_cf[key] = value

        forecast = group.get("forecast")
        if forecast is not None:
            group_cf.fe_num_blocks = forecast.get("num_blocks", cf.fe_num_blocks)
            group_cf.fe_num_heads = forecast.get("num_heads", cf.fe_num_heads)
            group_cf.fe_dropout_rate = forecast.get("dropout_rate", cf.fe_dropout_rate)
            group_cf.fe_with_qk_lnorm = forecast.get("with_qk_lnorm", cf.fe_with_qk_lnorm)

        return group_cf

    def _create_latent_groups(self, cf: Config, mode_cfg) -> None:
        """Create one encoder tower and forecasting engine per latent group, plus the couplers."""

        dims = set(self.latent_group_dims.values())
        if len(dims) > 1:
            raise NotImplementedError(
                f"Latent groups of differing width {sorted(dims)} would need per-group decoder "
                "and forecasting-engine widths, which is phase 3 of "
                "docs/multi_time_latents_proposal.md. Give every group the same dim_embed."
            )

        self.encoders = nn.ModuleDict()
        self.forecast_engines = nn.ModuleDict()

        for group_name, group in self.latent_groups.items():
            group_cf = self._group_config(cf, group)
            self.encoders[group_name] = EncoderModule(
                group_cf,
                self.sources_size,
                self.targets_num_channels,
                self.targets_coords_size,
                stream_names=self.latent_group_streams[group_name],
            )
            # only groups that are actually advanced get an engine, so that no parameter is
            # left without a gradient, which FSDP2 does not tolerate
            if group.get("forecast") is not None and group_name in self.latent_group_strides:
                self.forecast_engines[group_name] = ForecastingEngine(
                    group_cf, mode_cfg, self.num_healpix_cells
                )

        coupling_cfg = cf.latent.get("coupling")
        if coupling_cfg is not None:

            def coupler_cfg(key):
                """The sub-block for one coupler, carrying the shared coupling width."""
                sub = coupling_cfg.get(key)
                if sub is None:
                    return None
                sub = sub.copy()
                if coupling_cfg.get("dim_embed") is not None:
                    sub.dim_embed = coupling_cfg.dim_embed
                return sub

            assimilation_cfg = coupler_cfg("assimilation")
            if assimilation_cfg is not None:
                self.assimilation_coupler = CouplingEngine(
                    cf, assimilation_cfg, self.latent_group_dims
                )

            rollout_cfg = coupler_cfg("rollout")
            if rollout_cfg is not None and self.coupling_stride is not None:
                self.rollout_coupler = CouplingEngine(cf, rollout_cfg, self.latent_group_dims)

        # the single-latent attributes stay unset; an identity keeps code that reaches for the
        # one forecasting engine harmless
        self.forecast_engine = IdentityEngine()

    def create(self) -> "Model":
        """Create each individual module of the model"""
        cf = self.cf

        mode_cfg = cf.training_config

        if self.latent_groups is not None:
            self._create_latent_groups(cf, mode_cfg)
        else:
            self.encoder = EncoderModule(
                cf, self.sources_size, self.targets_num_channels, self.targets_coords_size
            )
            if cf.fe_num_blocks > 0:
                self.forecast_engine = ForecastingEngine(cf, mode_cfg, self.num_healpix_cells)
            else:
                self.forecast_engine = IdentityEngine()

        # embed coordinates yielding one query token for each target token
        dropout_rate = cf.embed_dropout_rate
        self.embed_target_coords = torch.nn.ModuleDict()
        self.target_token_engines = torch.nn.ModuleDict()
        self.pred_heads = torch.nn.ModuleDict()

        # determine stream names once so downstream components use consistent keys
        loss_terms = [
            v.type for _, v in cf.training_config.losses.items() if v.get("enabled", True)
        ]
        if cf.validation_config.get("losses"):
            loss_terms += [
                v.type for _, v in cf.validation_config.losses.items() if v.get("enabled", True)
            ]

        if "LossPhysical" in loss_terms:
            for i_stream, (stream_name, si) in enumerate(self.streams.items()):
                # skip decoder if channels are empty
                if is_stream_forcing(si):
                    continue

                # skip for the moment to ensure target embedding and tte exist (ordering of
                # cf.streams is random)
                if si.get("pred_spatial_shared") is None:
                    # extract and setup relevant parameters
                    etc = si["embed_target_coords"]
                    tr = si["target_readout"]
                    num_layers = tr["num_layers"]
                    tr_mlp_hidden_factor = (
                        tr["mlp_hidden_factor"] if "mlp_hidden_factor" in tr else 2
                    )
                    tr_dim_head_proj = tr["dim_head_proj"] if "dim_head_proj" in tr else None
                    softcap = tr["softcap"] if "softcap" in tr else 0.0

                    dims_embed = [
                        si["embed_target_coords"]["dim_embed"] for _ in range(num_layers + 1)
                    ]

                    if is_root():
                        logger.info("{} :: coord embed: :: {}".format(si["name"], dims_embed))

                    dim_coord_in = self.targets_coords_size[i_stream]

                    # embedding network for coordinates
                    if etc["net"] == "linear":
                        self.embed_target_coords[stream_name] = NamedLinear(
                            f"embed_target_coords_{stream_name}",
                            in_features=dim_coord_in,
                            out_features=dims_embed[0],
                            bias=False,
                        )
                    elif etc["net"] == "mlp":
                        self.embed_target_coords[stream_name] = MLP(
                            dim_coord_in,
                            dims_embed[0],
                            hidden_factor=8,
                            with_residual=False,
                            dropout_rate=dropout_rate,
                            norm_eps=self.cf.mlp_norm_eps,
                            name=f"embed_target_coords_{stream_name}",
                        )
                    else:
                        assert False

                    if cf.decoder_type == "Linear":
                        tte = BilinearDecoder(
                            stream_name,
                            dims_embed[0],
                            cf.ae_global_dim_embed,
                            self.targets_num_channels[i_stream],
                        )
                    else:
                        # target prediction engines
                        tte_version = (
                            TargetPredictionEngine
                            if cf.decoder_type != "PerceiverIOCoordConditioning"
                            else TargetPredictionEngineClassic
                        )
                        tte = tte_version(
                            cf,
                            dims_embed,
                            dim_coord_in,
                            tr_dim_head_proj,
                            tr_mlp_hidden_factor,
                            softcap,
                            stream_config=si,
                        )

                    self.target_token_engines[stream_name] = tte

                    # ensemble prediction heads to provide probabilistic prediction
                    final_activation = si["pred_head"].get("final_activation", "Identity")
                    if is_root():
                        logger.debug(
                            f"{final_activation} activation of pred head of {si['name']} stream"
                        )
                    self.pred_heads[stream_name] = EnsPredictionHead(
                        dims_embed[-1],
                        self.targets_num_channels[i_stream],
                        si["pred_head"]["num_layers"],
                        si["pred_head"]["ens_size"],
                        norm_type=cf.norm_type,
                        final_activation=final_activation,
                        stream_name=stream_name,
                    )

            # iterate again to setup shared spatial pred heads if specified in config
            for i_stream, (stream_name, si) in enumerate(self.streams.items()):
                # skip decoder if channels are empty
                if is_stream_forcing(si):
                    continue

                pred_spatial_shared = si.get("pred_spatial_shared")
                if pred_spatial_shared is not None:
                    if pred_spatial_shared not in self.streams.keys():
                        msg = f"Stream {stream_name} has pred_spatial_shared={pred_spatial_shared}"
                        msg += " but no stream with that name found."
                        raise ValueError(msg)
                    if pred_spatial_shared == stream_name:
                        msg = f"Stream {stream_name} has pred_spatial_shared={pred_spatial_shared}"
                        msg += "but cannot share with itself."
                        raise ValueError(msg)
                    logger.debug(
                        f"{stream_name} shares spatial prediction head with {pred_spatial_shared}."
                    )

                    self.embed_target_coords[stream_name] = self.embed_target_coords[
                        pred_spatial_shared
                    ]
                    self.target_token_engines[stream_name] = self.target_token_engines[
                        pred_spatial_shared
                    ]

                    assert pred_spatial_shared in self.streams.keys()
                    si_other = self.streams[pred_spatial_shared]
                    dims_embed = [
                        si_other["embed_target_coords"]["dim_embed"] for _ in range(num_layers + 1)
                    ]

                    # ensemble prediction heads to provide probabilistic prediction
                    final_activation = si["pred_head"].get("final_activation", "Identity")
                    if is_root():
                        logger.debug(
                            f"{final_activation} activation of pred head of {si['name']} stream"
                        )
                    self.pred_heads[stream_name] = EnsPredictionHead(
                        dims_embed[-1],
                        self.targets_num_channels[i_stream],
                        si["pred_head"]["num_layers"],
                        si["pred_head"]["ens_size"],
                        norm_type=cf.norm_type,
                        final_activation=final_activation,
                        stream_name=stream_name,
                    )

        # Latent heads for losses
        self.latent_heads = nn.ModuleDict()
        self.latent_pre_norm = nn.LayerNorm(cf.ae_global_dim_embed)

        ssl_losses_cfgs = [
            v
            for _, v in cf.training_config.losses.items()
            if v.type == "LossLatentSSLStudentTeacher" and v.get("enabled", True)
        ]

        # TODO: support multiple LossLatentSSLStudentTeacher terms
        assert len(ssl_losses_cfgs) <= 1, "To be implemented."
        for ssl_target_losses in ssl_losses_cfgs:
            self.latent_pre_norm = nn.LayerNorm(cf.ae_global_dim_embed)
            for loss, loss_conf in ssl_target_losses.loss_fcts.items():
                if loss == "iBOT":
                    self.latent_heads[loss] = self._create_latent_pred_head(
                        cf,
                        f"{loss}-head",
                        loss_conf,
                        use_class_token=True,
                        use_patch_token=True,
                    )
                elif loss == "JEPA":
                    self.latent_heads[loss] = self._create_latent_pred_head(
                        cf,
                        f"{loss}-head",
                        loss_conf,
                        use_class_token=False,
                        use_patch_token=True,
                    )
                elif loss == "DINO":
                    self.latent_heads[loss] = self._create_latent_pred_head(
                        cf,
                        f"{loss}-head",
                        loss_conf,
                        use_class_token=True,
                        use_patch_token=False,
                    )

        return self

    def reset_parameters(self):
        def _reset_params(module):
            if isinstance(module, AdaLayerNorm | nn.Linear | nn.LayerNorm):
                module.reset_parameters()
            else:
                pass

        self.apply(_reset_params)

    def _print_num_parameters_groups(self) -> None:
        """Per-tower parameter report, used when latent groups are configured."""

        print("-----------------")
        print(f"Total number of trainable parameters: {get_num_parameters(self):,}")
        for group_name, encoder in self.encoders.items():
            print(f"  Latent group '{group_name}':")
            for stream_name in encoder.stream_names:
                num_params = get_num_parameters(encoder.embed_engine.embeds[stream_name])
                print(f"    embedding {stream_name} : {num_params:,}")
            print(
                "    local assimilation: "
                f"{get_num_parameters(encoder.ae_local_engine.ae_local_blocks):,}"
            )
            print(
                f"    local-global adapter: {get_num_parameters(encoder.ae_local_global_engine):,}"
            )
            print(
                "    query aggregation: "
                f"{get_num_parameters(encoder.ae_aggregation_engine.ae_aggregation_blocks):,}"
            )
            print(
                "    global assimilation: "
                f"{get_num_parameters(encoder.ae_global_engine.ae_global_blocks):,}"
            )
            if group_name in self.forecast_engines:
                num_params = get_num_parameters(self.forecast_engines[group_name].fe_blocks)
                print(f"    forecasting engine: {num_params:,}")
            else:
                print("    forecasting engine: none, this group is never advanced")

        for label, coupler in (
            ("assimilation", self.assimilation_coupler),
            ("rollout", self.rollout_coupler),
        ):
            num_params = get_num_parameters(coupler.fe_blocks) if coupler is not None else 0
            print(f"  Coupling engine, {label}: {num_params:,}")

        print(" coordinate embedding, prediction networks and prediction heads:")
        for stream_name in self.streams.keys():
            nps = [
                get_num_parameters(mdict[stream_name]) if mdict and stream_name in mdict else 0
                for mdict in (self.embed_target_coords, self.target_token_engines, self.pred_heads)
            ]
            print(f"   {stream_name} : {nps[0]:,} / {nps[1]:,} / {nps[2]:,}")
        print("-----------------")

    def print_num_parameters(self) -> None:
        """Print number of parameters for entire model and each module used to build the model"""

        if self.latent_groups is not None:
            self._print_num_parameters_groups()
            return

        num_params_embed = [
            get_num_parameters(self.encoder.embed_engine.embeds[name])
            for name in self.streams.keys()
        ]
        num_params_total = get_num_parameters(self)
        num_params_ae_local = get_num_parameters(self.encoder.ae_local_engine.ae_local_blocks)
        num_params_ae_global = get_num_parameters(self.encoder.ae_global_engine.ae_global_blocks)

        num_params_q_cells = (
            np.prod(self.encoder.q_cells.shape) if self.encoder.q_cells.requires_grad else 0
        )
        num_params_ae_adapter = get_num_parameters(self.encoder.ae_local_global_engine)

        num_params_ae_aggregation = get_num_parameters(
            self.encoder.ae_aggregation_engine.ae_aggregation_blocks
        )

        num_params_latent_heads = get_num_parameters(self.latent_heads)
        num_params_latent_heads += get_num_parameters(self.latent_pre_norm)

        num_params_fe = get_num_parameters(self.forecast_engine.fe_blocks)

        mdict = self.embed_target_coords
        num_params_embed_tcs = [
            get_num_parameters(mdict[name]) if mdict and name in mdict else 0
            for name in self.streams.keys()
        ]
        mdict = self.target_token_engines
        num_params_tte = [
            get_num_parameters(mdict[name]) if mdict and name in mdict else 0
            for name in self.streams.keys()
        ]
        mdict = self.pred_heads
        num_params_preds = [
            get_num_parameters(mdict[name]) if mdict and name in mdict else 0
            for name in self.streams.keys()
        ]

        print("-----------------")
        print(f"Total number of trainable parameters: {num_params_total:,}")
        print("Number of parameters:")
        print("  Embedding networks:")
        [
            print("    {} : {:,}".format(si["name"], np))
            for si, np in zip(self.streams.values(), num_params_embed, strict=False)
        ]
        print(f" Local assimilation engine: {num_params_ae_local:,}")
        print(f" Local-global adapter: {num_params_ae_adapter:,}")
        print(f" Learnable queries: {num_params_q_cells:,}")
        print(f" Query Aggregation engine: {num_params_ae_aggregation:,}")
        print(f" Global assimilation engine: {num_params_ae_global:,}")
        print(f" Latent prediction heads and pre-norm: {num_params_latent_heads:,}")
        print(f" Forecast engine: {num_params_fe:,}")
        print(" coordinate embedding, prediction networks and prediction heads:")
        zps = zip(
            self.streams.keys(),
            num_params_embed_tcs,
            num_params_tte,
            num_params_preds,
            strict=False,
        )
        for stream_name, np0, np1, np2 in zps:
            print(f"   {stream_name} : {np0:,} / {np1:,} / {np2:,}")
        print("-----------------")

    def tokens_to_latent_state(self, tokens_post_norm, tokens) -> LatentState:
        """
        Extract separate parts from global latent space representation and store in LatentState
        """
        toks_pn = tokens_post_norm
        return LatentState(
            register_tokens=toks_pn[:, self.register_token_idxs] if toks_pn is not None else None,
            class_token=toks_pn[:, self.class_token_idxs] if tokens_post_norm is not None else None,
            patch_tokens=toks_pn[:, self.num_aux_tokens :] if toks_pn is not None else None,
            z_pre_norm=tokens,
        )

    def forward(self, model_params: ModelParams, batch: ModelBatch) -> ModelOutput:
        """Forward pass of the model

        Tokens are processed through the model components, which were defined in the create method.
        Args:
            model_params : Query and embedding parameters
            batch
        Returns:
            A list containing all prediction results
        """

        output = ModelOutput(batch.get_output_len())

        tokens, posteriors = self.encode(model_params, batch)
        output.add_latent_prediction(0, "posteriors", posteriors)

        # Allow for pushforward trick
        p_fwd = self.cf.training_config.get("forecast", {}).get("pushforward", False)
        # roll-out in latent space, iterate and generate output over requested output steps
        for step in batch.get_output_idxs():
            without_grad = p_fwd and self.training and step != max(batch.get_output_idxs())
            if without_grad:
                # Pushforward mode: advance tokens without grad; no decoding with torch.no_grad():
                tokens = self.advance_latent(tokens, step, model_params.rope_coords)
                continue

            tokens = self.advance_latent(tokens, step, model_params.rope_coords)
            # decoder predictions
            output = self.predict_decoders(model_params, step, tokens, batch, output)
            # latent predictions (raw and with SSL heads)
            output = self.predict_latent(model_params, step, tokens, batch, output)

        return output

    def encode(self, model_params: ModelParams, batch: ModelBatch):
        """Run the encoder, or one tower per latent group, and couple them at analysis time.

        Args:
            model_params : Query and embedding parameters
            batch
        Returns:
            (latent, posteriors). The latent is a tensor for a single latent space, and a
            mapping from group name to tensor when latent groups are configured.
        """

        def collapse_input_steps(tokens: torch.Tensor) -> torch.Tensor:
            # recover batch dimension, separate input steps, then sum over them
            shape = (len(batch), batch.get_num_source_steps(), *tokens.shape[1:])
            return tokens.reshape(shape).sum(axis=1)

        if self.latent_groups is None:
            tokens, posteriors = self.encoder(model_params, batch)
            return collapse_input_steps(tokens), posteriors

        latents = {}
        posteriors = []
        for group_name, encoder in self.encoders.items():
            tokens, group_posteriors = encoder(model_params, batch)
            latents[group_name] = collapse_input_steps(tokens)
            posteriors += (
                list(group_posteriors) if isinstance(group_posteriors, list) else [group_posteriors]
            )

        # the towers ingest disjoint sets of parameters and, in general, different streams, so
        # this is where cross-domain information first reaches each of them
        if self.assimilation_coupler is not None:
            latents = self.assimilation_coupler(latents)

        return latents, posteriors

    def advance_latent(self, latents, step: int, coords: torch.Tensor | None = None):
        """Advance the latent state by one output step.

        With a single latent space this is the forecasting engine. With latent groups, each
        group is advanced by its own engine only on the steps where its stride fires, so a slow
        group is held constant in between; the rollout coupler then mixes the groups at its own
        cadence and is the only place they exchange information during rollout.

        Args:
            latents : the latent state, a tensor or a mapping from group name to tensor
            step : Index of the output step, as given by batch.get_output_idxs()
            coords : Coordinates for 2D RoPE, or None
        Returns:
            The advanced latent state, in the same form as the input.
        """
        if self.latent_groups is None:
            return self.forecast_engine(latents, step, coords)

        advanced = {}
        for group_name, tokens in latents.items():
            stride = self.latent_group_strides.get(group_name)
            if stride is not None and step % stride == 0:
                # rope_2D is rejected for latent groups, so no coordinates are threaded here
                tokens = self.forecast_engines[group_name](tokens, step, None)
            advanced[group_name] = tokens

        if self.rollout_coupler is not None and step % self.coupling_stride == 0:
            advanced = self.rollout_coupler(advanced)

        return advanced

    def concat_latent_groups(self, latents) -> torch.Tensor:
        """Concatenate the groups' latents along the token axis, in configuration order."""
        if self.latent_groups is None:
            return latents
        return torch.cat([latents[name] for name in self.latent_groups if name in latents], dim=1)

    def predict_latent(
        self,
        model_params: ModelParams,
        step: int,
        tokens: torch.Tensor,
        batch: ModelBatch,
        output: ModelOutput,
    ) -> ModelOutput:
        """
        Compute latent predictions
        """

        # latent losses are not per group yet, so the towers are viewed as one token axis
        tokens = self.concat_latent_groups(tokens)

        # safe latent prediction
        tokens_post_norm = self.latent_pre_norm(tokens) if step == 0 else None
        latent_state = self.tokens_to_latent_state(tokens_post_norm, tokens)
        output.add_latent_prediction(step, "latent_state", latent_state)

        # latent predictions for SSL training
        for name, head in self.latent_heads.items():
            output.add_latent_prediction(step, name, head(latent_state))

        return output

    def predict_decoders(
        self,
        model_params: ModelParams,
        step: int,
        tokens: torch.Tensor,
        batch: ModelBatch,
        output: ModelOutput,
    ) -> ModelOutput:
        """
        Compute decoder-based predictions

        Predict outputs at the specific target coordinates based on the input weather state and
        pre-training task and projects the latent space representation back to physical space.

        Args:
            model_params : Query and embedding parameters
            fstep : Number of forecast steps
            tokens : Tokens from global assimilation engine
            streams_data : Used to initialize target coordinates tokens and index information
                List of StreamData len(streams_data) == batch_size_per_gpu
            target_coords_idxs : Indices of target coordinates
        Returns:
            Prediction output tokens in physical representation for each target_coords.
        """
        # Empty dicts evaluate to False in python
        if not self.pred_heads:
            return output

        batch_size = len(batch)
        idxs = model_params.hp_nbours.unsqueeze(0).repeat((batch_size, 1, 1)).flatten(0, 1)

        # Latents to decode from, per latent group. Streams decoding from the same group share
        # the gather; without groups there is a single entry keyed by None.
        latents_per_group: dict[str | None, tuple] = {}

        def latents_for_group(group_name: str | None) -> tuple:
            if group_name not in latents_per_group:
                group_tokens = tokens if group_name is None else tokens[group_name]
                # remove register and class tokens
                group_tokens = group_tokens[:, self.num_aux_tokens :]

                s = [
                    batch_size,
                    self.num_healpix_cells,
                    self.cf.ae_local_num_queries,
                    group_tokens.shape[-1],
                ]
                nbors = group_tokens.reshape(s).flatten(0, 1)[idxs.flatten()].flatten(0, 1)
                # TODO: precompute in model_params?
                # the 1-ring gather yields 9 cells per cell, each contributing s[2] queries
                nbors_lens = torch.full(
                    (s[0] * s[1] + 1,),
                    fill_value=9 * s[2],
                    dtype=torch.int32,
                    device=nbors.device,
                )
                nbors_lens[0] = 0
                latents_per_group[group_name] = (group_tokens, nbors, nbors_lens, s)

            return latents_per_group[group_name]

        # pair with tokens from assimilation engine to obtain target tokens
        for stream_name in self.streams.keys():
            # extract target coords for current stream and fstep and convert to one tensor
            t_coords = [
                batch.samples[i_b].streams_data[stream_name].target_coords[step]
                for i_b in range(batch_size)
            ]
            t_coords_lens = [len(t) for t in t_coords]
            t_coords = torch.cat(t_coords)

            if len(t_coords) == 0:
                continue

            # embed token coords
            tc_embed = self.embed_target_coords[stream_name]
            tc_tokens = checkpoint(tc_embed, t_coords, use_reentrant=False)

            # skip when coordinate embeddings yields nan (i.e. the coord embedding network diverged)
            if torch.isnan(tc_tokens).any():
                logger.warning(
                    (
                        f"Skipping prediction for {stream_name} because",
                        f" of {torch.isnan(tc_tokens).sum()} NaN in tc_tokens.",
                    )
                )
                pred = torch.tensor([], device=tc_tokens.device)

            # skip empty lengths
            elif tc_tokens.shape[0] == 0:
                pred = torch.tensor([], device=tc_tokens.device)

            else:
                # lens for varlen attention
                tcls = torch.cat(
                    [
                        sample.streams_data[stream_name].target_coords_lens[step]
                        for sample in batch.samples
                    ]
                )
                tcs_lens = torch.cat([torch.zeros(1, dtype=torch.int32, device=tcls.device), tcls])

                # decode from the tower this stream is routed to
                group_tokens, tokens_nbors, tokens_nbors_lens, s = latents_for_group(
                    self.stream_latent_groups.get(stream_name)
                )

                if self.cf.decoder_type == "Linear":
                    pred = self.target_token_engines[stream_name](
                        tc_tokens,
                        group_tokens.reshape(-1, s[-1]),  # collapse batch and token dimensions
                        tcs_lens,
                    ).unsqueeze(0)  # add ensemble dim: shape is then [1, preds_per_coord, channels]
                else:
                    tc_tokens = self.target_token_engines[stream_name](
                        latent=tokens_nbors,
                        output=tc_tokens,
                        latent_lens=tokens_nbors_lens,
                        output_lens=tcs_lens,
                        coordinates=t_coords,
                    )

                    # final prediction head to map back to physical space
                    pred = self.pred_heads[stream_name](tc_tokens)

            # recover batch dimension (ragged, so as list)
            pred = torch.split(pred, t_coords_lens, dim=1)
            output.add_physical_prediction(step, stream_name, pred)

        return output
