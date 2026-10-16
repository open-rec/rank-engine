"""Coordinate scoring against one fixed runtime for the entire request."""

import logging

from error_code import ErrorCode, ReException
from inference.batch import prepare_batch


class InferenceService:
    def __init__(self, model_service, feature_service):
        self.model_service = model_service
        self.feature_service = feature_service

    def score(self, request):
        runtime = self.model_service.current(request.target_type)
        if runtime is None:
            try:
                self.model_service.restore(request.target_type)
                runtime = self.model_service.current(request.target_type)
            except Exception:
                logging.exception("lazy model load failed")
        if runtime is None:
            raise ReException(ErrorCode.MODEL_NOT_LOAD_YET)
        target = runtime.info.get("target_type", "item")
        if request.target_type != target:
            raise ReException(ErrorCode.INVALID_MODEL)
        if not (request.candidate_ids or request.item_ids):
            return {}
        try:
            try:
                self.model_service.schedule_feature_refresh(target)
            except Exception:
                logging.exception("feature refresh failed; retaining last usable snapshot")
            batch = prepare_batch(request, runtime, self.feature_service)
            scores = batch.missing_scores
            if batch.candidate_ids:
                predictions = runtime.executor.score(batch)
                if len(predictions) != len(batch.candidate_ids):
                    raise ValueError("prediction count does not match candidate count")
                scores.update(zip(batch.candidate_ids, predictions))
            return scores
        except Exception:
            logging.exception("inference failed")
            raise ReException(ErrorCode.INFERENCE_FAILED)
