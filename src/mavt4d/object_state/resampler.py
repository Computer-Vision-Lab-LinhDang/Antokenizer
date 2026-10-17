"""Competitive slot binding, evidence correction, and canonical content writes."""

from dataclasses import replace
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ObjectStateConfig
from .state import Evidence, ObjectState


def _masked_softmax(logits: Tensor, valid: Tensor):
    scores = logits.masked_fill(~valid, torch.finfo(logits.dtype).min).softmax(-1)
    scores = scores * valid
    return scores / scores.sum(-1, keepdim=True).clamp_min(1e-12)


class ObjectResampler(nn.Module):
    def __init__(self, config: ObjectStateConfig):
        super().__init__()
        self.config = config
        z, e = config.width, config.evidence_width
        self.descriptor = nn.Linear(
            (config.content_tokens + config.dynamic_tokens) * z, z
        )
        self.birth_descriptor = nn.Parameter(
            torch.randn(config.slots, z) / math.sqrt(z)
        )
        self.birth_d = nn.Parameter(
            torch.randn(config.slots, config.dynamic_tokens, z) * 0.02
        )
        self.birth_evidence = nn.Linear(e, config.dynamic_tokens * z, bias=False)
        self.background_descriptor = nn.Parameter(torch.randn(z) / math.sqrt(z))
        self.ownership_key = nn.Linear(e, z)
        self.dynamic_query = nn.Linear(z, z)
        self.content_summary = nn.Linear(config.content_tokens * z, z)
        self.dynamic_seed = nn.Parameter(
            torch.randn(config.dynamic_tokens, z) / math.sqrt(z)
        )
        self.evidence_key = nn.Linear(e, z)
        self.evidence_value = nn.Linear(e, z, bias=False)
        self.dynamic_output = nn.Linear(z, z)
        self.correction = nn.Sequential(
            nn.Linear(z * 3, z * 2), nn.SiLU(), nn.Linear(z * 2, z)
        )
        self.correction_gate = nn.Linear(z * 3, z)
        self.canonical_queries = nn.Parameter(
            torch.randn(config.content_tokens, z) / math.sqrt(z)
        )
        self.canonical_geometry = nn.Sequential(
            nn.Linear(10, z), nn.SiLU(), nn.Linear(z, z)
        )
        self.content_output = nn.Linear(z, z, bias=False)
        self.content_gate = nn.Linear(z + 3, 1)
        self.background_queries = nn.Parameter(
            torch.randn(config.background_tokens, z) / math.sqrt(z)
        )
        self.background_gate = nn.Linear(z * 2, 1)
        self.existence = nn.Linear(config.dynamic_tokens * z + 1, 1)

    def _descriptors(self, state, *, proposals):
        descriptors = self.descriptor(
            torch.cat((state.c.flatten(-2), state.d.flatten(-2)), dim=-1)
        )
        if proposals:
            descriptors = torch.where(
                state.allocated[..., None],
                descriptors,
                self.birth_descriptor[None].expand_as(descriptors),
            )
        return descriptors

    def ownership(
        self, state: ObjectState, evidence: Evidence, controls, *, proposals=False
    ):
        tokens = torch.where(
            evidence.valid[..., None],
            evidence.tokens.float(),
            torch.zeros_like(evidence.tokens, dtype=torch.float32),
        )
        keys = self.ownership_key(tokens)
        descriptors = self._descriptors(state, proposals=proposals)
        bg = self.background_descriptor[None, None].expand(tokens.shape[0], 1, -1)
        logits = torch.einsum(
            "bnz,bkz->bnk", keys, torch.cat((descriptors, bg), dim=1)
        ) / math.sqrt(self.config.width)
        # Metric ray distance is a soft bias; never discard global content matches.
        geo = (
            evidence.valid
            if evidence.geometry_valid is None
            else evidence.valid & evidence.geometry_valid
        )
        origins = torch.where(
            geo[..., None],
            evidence.ray_origins.float(),
            torch.zeros_like(evidence.ray_origins, dtype=torch.float32),
        )
        dirs = F.normalize(
            torch.where(
                geo[..., None],
                evidence.ray_directions.float(),
                torch.zeros_like(evidence.ray_directions, dtype=torch.float32),
            ),
            dim=-1,
        )
        centers = controls(state.d)[..., :3, 3]
        offset = centers[:, None] - origins[:, :, None]
        along = (offset * dirs[:, :, None]).sum(-1, keepdim=True)
        distance = (offset - along * dirs[:, :, None]).square().sum(-1)
        # Bounded bias avoids destroying recovery if the predicted pose is wrong.
        bias = -self.config.geometry_bias_scale * torch.log1p(distance)
        bias = torch.where(
            geo[:, :, None] & state.allocated[:, None], bias, torch.zeros_like(bias)
        )
        logits = logits + torch.cat((bias, torch.zeros_like(bias[..., :1])), dim=-1)
        if not proposals:
            allowed = torch.cat(
                (state.allocated, torch.ones_like(state.allocated[:, :1])), dim=1
            )
            logits = logits.masked_fill(
                ~allowed[:, None], torch.finfo(logits.dtype).min
            )
        return logits.softmax(-1) * evidence.valid[..., None]

    def _read(self, queries, keys, values, weights, valid):
        # Shapes [B,K,Q,Z], [B,K,N,Z], ownership [B,K,N].
        b, k, q, z = queries.shape
        n = keys.shape[-2]
        heads = self.config.resampler_heads
        h = z // heads
        qs = queries.reshape(b, k, q, heads, h).transpose(-3, -2)
        ks = keys.reshape(b, k, n, heads, h).transpose(-3, -2)
        vs = values.reshape(b, k, n, heads, h).transpose(-3, -2)
        logits = qs @ ks.transpose(-1, -2) / math.sqrt(h)
        logits = logits + weights.clamp_min(1e-12).log()[:, :, None, None, :]
        attention = _masked_softmax(logits, valid[:, :, None, None, :])
        return (attention @ vs).transpose(-3, -2).reshape(b, k, q, z)

    def _candidates(self, evidence: Evidence, ownership: Tensor, candidate_budget=None):
        b, n, _ = evidence.tokens.shape
        k = self.config.slots
        budget = (
            self.config.candidates if candidate_budget is None else candidate_budget
        )
        if type(budget) is not int or budget < 0:
            raise ValueError("candidate_budget must be a nonnegative integer or None")
        count = n if budget == 0 else min(budget, n)
        if budget == 0:
            # Warm-up retains all soft ownership weights, without a top-k gate.
            weights = ownership[..., :k].transpose(1, 2)
            indices = torch.arange(n, device=ownership.device)[None, None].expand(
                b, k, -1
            )
        else:
            weights, indices = (
                ownership[..., :k].transpose(1, 2).topk(count, dim=-1, sorted=True)
            )

        def gather(value):
            # [B,N,...] -> [B,K,M,...], indices are always current-generation.
            extra = value.shape[2:]
            source = value[:, None].expand(b, k, n, *extra)
            index = indices.reshape(b, k, count, *([1] * len(extra))).expand(
                b, k, count, *extra
            )
            return source.gather(2, index)

        tokens = gather(
            torch.where(
                evidence.valid[..., None],
                evidence.tokens.float(),
                torch.zeros_like(evidence.tokens, dtype=torch.float32),
            )
        )
        validity = gather(evidence.valid) & (weights > 0)
        geo = (
            evidence.valid
            if evidence.geometry_valid is None
            else evidence.valid & evidence.geometry_valid
        )
        origins = gather(
            torch.where(
                geo[..., None],
                evidence.ray_origins.float(),
                torch.zeros_like(evidence.ray_origins, dtype=torch.float32),
            )
        )
        directions = gather(
            torch.where(
                geo[..., None],
                evidence.ray_directions.float(),
                torch.zeros_like(evidence.ray_directions, dtype=torch.float32),
            )
        )
        points = torch.zeros_like(origins)
        point_valid = torch.zeros_like(validity)
        if evidence.points is not None:
            usable = evidence.valid & evidence.point_valid
            points = gather(
                torch.where(
                    usable[..., None],
                    evidence.points.float(),
                    torch.zeros_like(evidence.points, dtype=torch.float32),
                )
            )
            point_valid = gather(usable)
        return (
            tokens,
            validity,
            weights,
            origins,
            directions,
            gather(geo),
            points,
            point_valid,
            indices,
        )

    def _canonical_keys(
        self, keys, origins, directions, points, point_valid, transform
    ):
        rotation = transform[..., :3, :3]
        translation = transform[..., :3, 3]
        origin_can = torch.einsum(
            "bknj,bkji->bkni", origins - translation[:, :, None], rotation
        )
        direction_can = torch.einsum("bknj,bkji->bkni", directions, rotation)
        point_can = torch.einsum(
            "bknj,bkji->bkni", points - translation[:, :, None], rotation
        )
        point_can = torch.where(
            point_valid[..., None], point_can, torch.zeros_like(point_can)
        )
        geometry = torch.cat(
            (origin_can, direction_can, point_can, point_valid[..., None].float()),
            dim=-1,
        )
        return keys + self.canonical_geometry(geometry)

    def forward(
        self,
        state: ObjectState,
        evidence: Evidence,
        controls,
        relations: Tensor,
        *,
        candidate_budget=None,
    ):
        """Caller selects observed batch rows; no all-missing attention is executed."""
        b, k = state.slot_status.shape
        proposals = self.ownership(state, evidence, controls, proposals=True)
        proposal_mass = proposals[..., :k].sum(1)
        proposal_peak = proposals[..., :k].amax(1)
        winners = (
            F.one_hot(proposals.argmax(-1), k + 1)[..., :k] * evidence.valid[..., None]
        )
        births = (
            (~state.allocated)
            & (proposal_mass >= self.config.birth_min_mass)
            & (proposal_peak >= self.config.birth_min_peak)
            & winners.any(1)
        )
        # The discrete lifecycle decision is not differentiable. The selected
        # proposal's evidence summary is, so birth descriptors receive task loss.
        safe_tokens = torch.where(
            evidence.valid[..., None],
            evidence.tokens.float(),
            torch.zeros_like(evidence.tokens, dtype=torch.float32),
        )
        proposal_values = torch.einsum("bnk,bne->bke", proposals[..., :k], safe_tokens)
        proposal_values = proposal_values / proposal_mass.clamp_min(1e-12)[..., None]
        born_d = self.birth_d[None] + self.birth_evidence(proposal_values).reshape(
            b, k, self.config.dynamic_tokens, self.config.width
        )
        initialized = replace(
            state,
            c=torch.where(births[..., None, None], torch.zeros_like(state.c), state.c),
            d=torch.where(births[..., None, None], born_d, state.d),
            slot_status=torch.where(
                births, torch.ones_like(state.slot_status), state.slot_status
            ),
            slot_uid=state.slot_uid + births.long(),
            p_exist=torch.where(births, torch.zeros_like(state.p_exist), state.p_exist),
            last_seen=torch.where(births, state.t[:, None], state.last_seen),
        )
        # Mandatory second association: candidate indices bind only to initialized generations.
        ownership = self.ownership(initialized, evidence, controls, proposals=False)
        # Soft tails over a large image are not positive evidence for a slot.
        # Require it to win local object competition and beat background.
        objects = ownership[..., :k]
        winning = F.one_hot(ownership.argmax(-1), k + 1)[..., :k].bool()
        confident = (
            winning & (objects > ownership[..., -1:]) & evidence.valid[..., None]
        )
        support = (objects * confident).sum(1)
        observed = initialized.allocated & (support >= self.config.observation_min_mass)
        (
            tokens,
            valid,
            weights,
            origins,
            directions,
            geo,
            points,
            point_valid,
            indices,
        ) = self._candidates(evidence, ownership, candidate_budget)
        keys = self.evidence_key(tokens)
        values = self.evidence_value(tokens)
        queries = (
            self.dynamic_query(initialized.d)
            + self.content_summary(initialized.c.flatten(-2))[:, :, None]
            + self.dynamic_seed[None, None]
        )
        read = self.dynamic_output(self._read(queries, keys, values, weights, valid))
        correction_input = torch.cat(
            (
                initialized.d,
                read,
                relations[:, :, None].expand(-1, -1, self.config.dynamic_tokens, -1),
            ),
            dim=-1,
        )
        updated_d = initialized.d + torch.sigmoid(
            self.correction_gate(correction_input)
        ) * self.correction(correction_input)
        d = torch.where(observed[..., None, None], updated_d, initialized.d)
        transforms = controls(d)
        canonical_keys = self._canonical_keys(
            keys, origins, directions, points, point_valid, transforms
        )
        can_valid = valid & geo & observed[..., None]
        c_query = self.canonical_queries[None, None].expand(b, k, -1, -1)
        candidate_c = self.content_output(
            self._read(c_query, canonical_keys, values, weights, can_valid)
        )
        entropy = -(ownership[..., :k].clamp_min(1e-12).log() * ownership[..., :k]).sum(
            1
        ) / ownership[..., :k].sum(1).clamp_min(1e-12)
        reliability = torch.stack(
            (torch.log1p(support), entropy, can_valid.float().mean(-1)), dim=-1
        )
        alpha = torch.sigmoid(
            self.content_gate(
                torch.cat(
                    (
                        initialized.c,
                        reliability[:, :, None].expand(
                            -1, -1, self.config.content_tokens, -1
                        ),
                    ),
                    dim=-1,
                )
            )
        )
        can_update = observed & can_valid.any(-1)
        c = torch.where(
            can_update[..., None, None],
            (1 - alpha) * initialized.c + alpha * candidate_c,
            initialized.c,
        )
        # Background uses world-frame geometry; content values still come only from evidence.
        all_tokens = torch.where(
            evidence.valid[..., None],
            evidence.tokens.float(),
            torch.zeros_like(evidence.tokens, dtype=torch.float32),
        )
        bg_keys = self.evidence_key(all_tokens)[:, None]
        bg_values = self.evidence_value(all_tokens)[:, None]
        all_geo = (
            evidence.valid
            if evidence.geometry_valid is None
            else evidence.valid & evidence.geometry_valid
        )
        all_origins = torch.where(
            all_geo[..., None],
            evidence.ray_origins.float(),
            torch.zeros_like(evidence.ray_origins, dtype=torch.float32),
        )
        all_dirs = torch.where(
            all_geo[..., None],
            evidence.ray_directions.float(),
            torch.zeros_like(evidence.ray_directions, dtype=torch.float32),
        )
        bg_geometry = torch.cat(
            (
                all_origins,
                all_dirs,
                torch.zeros_like(all_origins),
                torch.zeros_like(all_geo[..., None], dtype=torch.float32),
            ),
            dim=-1,
        )
        bg_keys = bg_keys + self.canonical_geometry(bg_geometry)[:, None]
        bg_read = self.content_output(
            self._read(
                self.background_queries[None, None].expand(b, 1, -1, -1),
                bg_keys,
                bg_values,
                ownership[..., -1][:, None],
                all_geo[:, None],
            )
        )[:, 0]
        bg_gate = torch.sigmoid(
            self.background_gate(torch.cat((state.c_bg, bg_read), dim=-1))
        )
        bg = torch.where(
            all_geo.any(-1)[:, None, None],
            (1 - bg_gate) * state.c_bg + bg_gate * bg_read,
            state.c_bg,
        )
        existence = torch.sigmoid(
            self.existence(
                torch.cat((d.flatten(-2), torch.log1p(support)[..., None]), dim=-1)
            )
        ).squeeze(-1)
        updated = replace(
            initialized,
            c=c,
            d=d,
            c_bg=bg,
            slot_status=torch.where(
                observed, torch.ones_like(state.slot_status), initialized.slot_status
            ),
            last_seen=torch.where(observed, state.t[:, None], initialized.last_seen),
            p_exist=torch.where(observed, existence, initialized.p_exist),
        )
        details = {
            "ownership": ownership,
            "birth_proposals": proposals,
            "candidate_uid": updated.slot_uid.clone(),
            "candidate_indices": indices,
            "support": support,
            "births": births,
            "alpha": alpha,
            "observed": observed,
        }
        return updated, details
