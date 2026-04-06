from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import savemat


def cell_to_num(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.empty(values.shape, dtype=float)

    out = np.full(values.shape, np.nan, dtype=float)
    it = np.nditer(values, flags=["multi_index", "refs_ok"], op_flags=["readonly"])
    for x in it:
        v = x.item()
        idx = it.multi_index
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        if isinstance(v, (int, float, np.number)):
            out[idx] = float(v)
            continue
        s = str(v).strip()
        if s == "":
            continue
        try:
            out[idx] = float(s)
        except ValueError:
            out[idx] = np.nan
    return out


def robust_zscore(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x, axis=0)
    sigma = np.nanstd(x, axis=0)
    sigma[sigma == 0] = 1.0
    return (x - mu) / sigma


def to_object_array(strings: Iterable[str]) -> np.ndarray:
    return np.asarray(list(strings), dtype=object)


def find_column_ci(columns: list[str], needle: str) -> int | None:
    needle_low = needle.lower()
    for i, c in enumerate(columns):
        if str(c).lower() == needle_low:
            return i
    return None


def regex_last_int(text: str) -> int | None:
    matches = re.findall(r"(\d+)", text)
    if not matches:
        return None
    return int(matches[-1])


def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    json_path = run_dir / "run_manifest.json"
    json_path.write_text(json.dumps(manifest, indent=2, default=_json_default))
    try:
        savemat(
            run_dir / "run_manifest.mat",
            {"manifest": _to_mat_compatible(manifest)},
            long_field_names=True,
        )
    except Exception as exc:
        # Keep JSON manifest as source of truth if MAT export fails.
        print(f"Warning: failed to write run_manifest.mat: {exc}")


def save_struct_mat(path: Path, data: dict[str, Any]) -> None:
    savemat(path, _to_mat_compatible(data), do_compression=True, long_field_names=True)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if sparse.issparse(obj):
        return obj.toarray().tolist()
    return str(obj)


def _to_mat_compatible(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _to_mat_compatible(v) for k, v in data.items()}
    if isinstance(data, list):
        return np.array([_to_mat_compatible(v) for v in data], dtype=object)
    if isinstance(data, tuple):
        return np.array([_to_mat_compatible(v) for v in data], dtype=object)
    if isinstance(data, pd.DataFrame):
        return {
            "columns": np.array(data.columns.astype(str).tolist(), dtype=object),
            "values": data.astype(object).to_numpy(),
        }
    if sparse.issparse(data):
        return data
    if isinstance(data, Path):
        return str(data)
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, (str, int, float, bool, np.number)):
        return data
    if data is None:
        return ""
    return str(data)
