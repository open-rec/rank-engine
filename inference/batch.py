"""Request feature encoding and lazy dense-batch assembly."""

from dataclasses import dataclass

import numpy as np


@dataclass
class InferenceBatch:
    request: object
    dim: int
    snapshot: object
    space: object
    user_features: object
    candidate_features: list
    candidate_ids: list
    dynamic_parts: list
    missing_scores: dict

    def matrix(self):
        item_width = self.candidate_features[0].size
        user_width = (self.space.user_width if self.space is not None
                      else self.dim - item_width)
        if self.user_features is None:
            self.user_features = np.zeros(user_width, dtype=np.float32)
        parts = [self.user_features, np.asarray(self.candidate_features, dtype=np.float32)]
        if self.space is not None:
            session = self.snapshot.get("sessions", {}).get(self.request.session_id)
            if session is None:
                session = np.zeros(self.space.session_width, dtype=np.float32)
            parts.append(session)
            parts.extend(self.dynamic_parts)
        if sum(part.shape[-1] for part in parts) != self.dim:
            raise ValueError("feature dimension does not match loaded model")
        # Broadcast shared vectors and cast directly into one float32
        # allocation instead of concatenating and copying each row.
        matrix = np.empty((len(self.candidate_ids), self.dim), dtype=np.float32)
        offset = 0
        for part in parts:
            width = part.shape[-1]
            matrix[:, offset:offset + width] = part
            offset += width
        return matrix


def prepare_batch(request, runtime, feature_service):
    snapshot = runtime.snapshot
    candidate_ids = request.candidate_ids or request.item_ids
    target_type = runtime.info.get("target_type", "item")
    space = snapshot.get("space") if snapshot is not None else None
    user_features = (
        snapshot["users"].get(request.user_id)
        if snapshot is not None
        else feature_service.get_user_feature_by_id(
            request.user_id, namespace=target_type
        )
    )
    candidate_features = []
    item_score_map = {}
    hit_items = []
    hit_positions = []
    for position, item_id in enumerate(candidate_ids):
        item_features = (
            snapshot["items"].get(item_id)
            if snapshot is not None
            else (
                feature_service.get_item_feature_by_id(item_id)
                if target_type == "item"
                else feature_service.get_candidate_user_feature_by_id(
                    item_id
                )
            )
        )
        if item_features is None:
            item_score_map[item_id] = 0.0
            continue
        candidate_features.append(item_features)
        hit_items.append(item_id)
        hit_positions.append(position)
    if hit_items:
        dynamic_parts = []
        if space is not None and (space.context_width or space.interaction_width):
            from algorithm.feature.context_feature import materialize_request_context
            # Positions and counts refer to the full request, including
            # missing candidates. Encode all hit rows in one batch.
            contexts = materialize_request_context(
                request.context, candidate_ids,
                request.context.get("request_time"),
                request.candidate_contexts,
            ).iloc[hit_positions]
            if space.context_width:
                dynamic_parts.append(space.transform_contexts(contexts))
            if space.interaction_width:
                dynamic_parts.append(space.transform_interactions(contexts))
    return InferenceBatch(
        request, runtime.dim, snapshot, space, user_features,
        candidate_features, hit_items, dynamic_parts if hit_items else [], item_score_map,
    )
