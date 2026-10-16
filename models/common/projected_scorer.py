"""CPU inference projections tied to one immutable model/feature snapshot."""

from itertools import islice

import numpy as np
import torch

from algorithm.rank.fm import FMModel
from algorithm.rank.lr import LRModel


class ProjectedScorer:
    """Move static LR/FM work to publication and feature refresh.

    FM retains each block's linear term, squared correction and factor
    projection. Summing projections before squaring preserves interactions
    between users, candidates, sessions and request features.
    """

    def __init__(self, model, snapshot):
        space = snapshot["space"]
        self.weight = model.linear.weight.detach().numpy().reshape(-1).copy()
        self.bias = model.linear.bias.detach().numpy().copy()
        self.factors = (model.factors.detach().numpy().copy()
                        if type(model) is FMModel else None)
        self.squared_factors = (np.square(self.factors)
                                if self.factors is not None else None)
        self.width = 2 + model.factor_dim if self.factors is not None else 1
        self.zero = np.zeros(self.width, dtype=np.float32)
        offset = 0
        self.blocks = {}
        for name, width in (("users", space.user_width),
                            ("items", space.item_width),
                            ("sessions", space.session_width)):
            rows = snapshot.get(name, {})
            indices = {}
            values = np.empty((len(rows), self.width), dtype=np.float32)
            iterator = iter(rows.items())
            # Bound temporary dense matrices during full feature refresh.
            for start in range(0, len(rows), 1024):
                chunk = list(islice(iterator, 1024))
                matrix = np.asarray([row for _, row in chunk], dtype=np.float32)
                if matrix.shape != (len(chunk), width):
                    raise ValueError("feature block does not match model dimension")
                values[start:start + len(chunk)] = self._project(matrix, offset)
                indices.update((identity, start + index)
                               for index, (identity, _) in enumerate(chunk))
            self.blocks[name] = (indices, values)
            offset += width
        self.dynamic_offset = offset

    def _project(self, matrix, offset):
        end = offset + matrix.shape[1]
        linear = matrix @ self.weight[offset:end]
        if self.factors is None:
            return linear[:, None]
        projected = matrix @ self.factors[offset:end]
        correction = (np.square(matrix) @ self.squared_factors[offset:end]).sum(axis=1)
        return np.column_stack((linear, correction, projected))

    def _shared(self, name, identity):
        indices, values = self.blocks[name]
        index = indices.get(identity)
        return self.zero if index is None else values[index]

    def score(self, user_id, candidate_ids, session_id, dynamic_parts):
        indices, values = self.blocks["items"]
        # Advanced indexing gives this request its own writable array.
        projected = values[[indices[identity] for identity in candidate_ids]]
        projected += self._shared("users", user_id)
        projected += self._shared("sessions", session_id)
        if dynamic_parts:
            matrix = np.concatenate(dynamic_parts, axis=1).astype(np.float32, copy=False)
            projected += self._project(matrix, self.dynamic_offset)
        logits = projected[:, 0] + self.bias
        if self.factors is not None:
            logits += .5 * (np.square(projected[:, 2:]).sum(axis=1) - projected[:, 1])
        with torch.inference_mode():
            return torch.sigmoid(torch.from_numpy(logits)).tolist()
