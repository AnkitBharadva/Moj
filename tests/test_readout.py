import math
import numpy as np
import pytest
from src.readout import CommentClassifier, TaskConfig, TASKS


def test_extract_logprobs_renormalization():
    classifier = CommentClassifier()
    letters = "ABC"

    # Mock Ollama response with top_logprobs
    mock_resp = {
        "logprobs": [
            {
                "token": "A",
                "logprob": math.log(0.6),
                "top_logprobs": [
                    {"token": "A", "logprob": math.log(0.6)},
                    {"token": " B", "logprob": math.log(0.3)},  # Leading space token
                    {"token": "C", "logprob": math.log(0.1)},
                    {"token": "The", "logprob": -5.0},          # Irrelevant token
                ],
            }
        ]
    }

    probs = classifier._extract_logprobs(mock_resp, letters)

    assert len(probs) == 3
    assert np.isclose(probs.sum(), 1.0)
    assert probs[0] > probs[1] > probs[2]
    assert np.isclose(probs[0], 0.6, atol=0.05)
    assert np.isclose(probs[1], 0.3, atol=0.05)


def test_extract_logprobs_fallback():
    classifier = CommentClassifier()
    letters = "ABC"

    # Response without logprobs (falls back to message content)
    mock_resp = {"message": {"content": "B"}}
    probs = classifier._extract_logprobs(mock_resp, letters)

    assert len(probs) == 3
    assert probs.argmax() == 1  # 'B' is index 1
    assert probs[1] > 0.99


def test_build_prompt_target_and_context():
    classifier = CommentClassifier()
    task = TaskConfig(
        name="support",
        question="What stance does this take regarding {target}?",
        options=["supports", "opposes", "neither"],
        default_target="Generic Entity",
    )

    prompt = classifier._build_prompt(
        comment="Great initiative!",
        task=task,
        target="Education Reform Bill",
        post_context="Government announces new education budget.",
    )

    assert "Education Reform Bill" in prompt
    assert "Government announces new education budget." in prompt
    assert "Great initiative!" in prompt
    assert "A) supports" in prompt
    assert "B) opposes" in prompt


def test_cache_key_uniqueness():
    classifier = CommentClassifier()
    k1 = classifier._cache_key("test comment", "sentiment", "TargetA", ["pos", "neg"], False)
    k2 = classifier._cache_key("test comment", "sentiment", "TargetB", ["pos", "neg"], False)
    k3 = classifier._cache_key("test comment", "sentiment", "TargetA", ["pos", "neg"], True)
    k4 = classifier._cache_key("test comment", "sentiment", "TargetA", ["pos", "neg"], False, post_context="Context")

    assert k1 != k2
    assert k1 != k3
    assert k1 != k4
