"""Save / load: architecture + config, state_dict, row coordinates, ids, dtype, seed, histories.

The checkpoint is a plain dict of tensors and Python primitives (torch.load(weights_only=True)
works); no pickled model object is the only record.
"""
from __future__ import annotations

import json
import math

import numpy as np
import torch

from .config import NeuralConfig
from .decoder import NeuralManifoldModel


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None if math.isnan(x) else (1e308 if x > 0 else -1e308)
    if torch.is_tensor(x) or isinstance(x, np.ndarray):
        return np.asarray(x).tolist()
    return x


def save_model(path, model: NeuralManifoldModel, history=None, extra=None):
    ck = dict(format="bernoulli_neural/v1", config=json.dumps(model.config.to_dict()),
              n=model.n, m=model.m, H=model.decoder.H, dtype=model.config.dtype,
              state_dict={k: v.detach().clone() for k, v in model.state_dict().items()},
              row_ids=None if model.row_ids is None else list(map(str, model.row_ids)),
              col_ids=None if model.col_ids is None else list(map(str, model.col_ids)),
              meta=json.dumps(_clean(model.meta)),
              history=json.dumps(_clean(history)) if history is not None else None,
              extra=json.dumps(_clean(extra)) if extra is not None else None,
              torch_version=str(torch.__version__))
    torch.save(ck, path)


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    cfg = NeuralConfig.from_dict(json.loads(ck["config"]))
    model = NeuralManifoldModel(cfg, ck["n"], ck["m"])
    model.load_state_dict(ck["state_dict"])
    model.row_ids, model.col_ids = ck["row_ids"], ck["col_ids"]
    model.meta = json.loads(ck["meta"])
    hist = json.loads(ck["history"]) if ck.get("history") else None
    extra = json.loads(ck["extra"]) if ck.get("extra") else None
    return model, hist, extra
