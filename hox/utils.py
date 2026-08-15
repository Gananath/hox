#!/usr/bin/env python
# coding: utf-8

# In[4]:


import glob
import os
from collections.abc import Sequence
import numpy as np
from huggingface_hub import snapshot_download
from . import config


# In[5]:


def hf_download(
    repo_id: str,
    precision: str | None = None,
    required_files: Sequence[str] | None = None,
    save_dir: str | None = None,
) -> str:
    suffix = "" if precision is None else f"_{precision}"
    repo_name = repo_id.rsplit("/", 1)[-1]

    if save_dir is None:
        save_dir = os.path.join(
            os.path.expanduser("~"),
            ".cache",
            config.DEFAULT_FOLDER,
            repo_name,
        )

    os.makedirs(save_dir, exist_ok=True)

    if required_files is None:
        required_files = [
            f"onnx/decoder_model_merged{suffix}.onnx*",
            f"onnx/decoder_model_merged{suffix}.onnx_data*",
            f"onnx/vision_encoder{suffix}.onnx*",
            f"onnx/vision_encoder{suffix}.onnx_data*",
            f"onnx/embed_tokens{suffix}.onnx",
            f"onnx/embed_tokens{suffix}.onnx_data*",
            f"onnx/model{suffix}.onnx",
            f"onnx/model{suffix}.onnx_data*",
            "chat_template.jinja",
            "tokenizer.json",
            "config.json",
            "tokenizer_config.json",
            "generation_config.json",
        ]

    # Correctly handle wildcard patterns such as "*.onnx_data*".
    if all(glob.glob(os.path.join(save_dir, pattern)) for pattern in required_files):
        return save_dir

    snapshot_download(
        repo_id=repo_id,
        local_dir=save_dir,
        allow_patterns=list(required_files),
    )

    return save_dir


# In[ ]:


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute numerically stable softmax."""

    logits = np.asarray(logits, dtype=np.float32)
    logits = logits - np.max(logits)

    exp_logits = np.exp(logits)

    return exp_logits / np.sum(exp_logits)


def apply_repetition_penalty(
    logits: np.ndarray,
    token_ids: list[int],
    repetition_penalty: float = 1.0,
) -> np.ndarray:
    """Apply repetition penalty to previously seen tokens."""

    if repetition_penalty <= 0:
        raise ValueError(
            "repetition_penalty must be > 0"
        )

    if repetition_penalty == 1.0:
        return logits

    logits = logits.copy()

    for token_id in set(token_ids):
        if logits[token_id] < 0:
            logits[token_id] *= repetition_penalty
        else:
            logits[token_id] /= repetition_penalty

    return logits


def apply_top_k(
    logits: np.ndarray,
    top_k: int = 0,
) -> np.ndarray:
    """Keep only the top-k logits."""

    if top_k <= 0:
        return logits

    top_k = min(top_k, logits.shape[-1])

    if top_k == logits.shape[-1]:
        return logits

    indices = np.argpartition(
        logits,
        -top_k,
    )[-top_k:]

    filtered = np.full_like(
        logits,
        -np.inf,
    )

    filtered[indices] = logits[indices]

    return filtered


def select_token(
    logits: np.ndarray,
    token_ids: list[int] | None = None,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
) -> int:
    """Select the next token using greedy decoding or sampling."""

    logits = np.asarray(
        logits,
        dtype=np.float32,
    )

    token_ids = token_ids or []

    # Repetition penalty.
    logits = apply_repetition_penalty(
        logits,
        token_ids,
        repetition_penalty,
    )

    # Greedy decoding.
    if not do_sample:
        return int(np.argmax(logits))

    if temperature <= 0:
        raise ValueError(
            "temperature must be > 0 when do_sample=True"
        )

    # Temperature.
    logits = logits / temperature

    # Top-k.
    logits = apply_top_k(
        logits,
        top_k,
    )

    # Probabilities.
    probabilities = softmax(logits)

    # Numerical safety.
    probabilities = np.nan_to_num(
        probabilities,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    total = probabilities.sum()

    if total <= 0:
        raise RuntimeError(
            "Invalid probability distribution."
        )

    probabilities /= total

    # Sampling.
    return int(
        np.random.choice(
            len(probabilities),
            p=probabilities,
        )
    )

