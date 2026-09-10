from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cvsuite.common.fm.providers.classify.siglip2 import _prediction_from_prompt_logits


def test_siglip2_scores_are_softmax_normalized_not_raw_sigmoid() -> None:
    # SigLIP logits are large-magnitude negatives; raw sigmoid would be ~1e-4
    # and never clear a CLIP-style threshold. Normalized across labels the
    # winner should be a real probability.
    logits = torch.tensor([-8.0, -14.0, -13.0])
    label, score, scores = _prediction_from_prompt_logits(logits, ["animal", "person", "object"], ["animal", "person", "object"])

    assert label == "animal"
    assert 0.9 < score <= 1.0
    assert abs(sum(scores.values()) - 1.0) < 1e-5
