"""Select the final model on validation and write best.json, fits/best.pt, embedding.npy,
embedding_hidden.npy, calibration tables and fitted-logit spectra.

embedding.npy         fitted row coordinates Z (n x d, float32) of the selected model, rows in the
                      order of full/data/Y.npy. A joint-MAP point estimate; identified only up to
                      reparametrisations absorbed by the decoder (rotations etc.).
embedding_hidden.npy  h(Z) = last hidden layer (n x H, float32). With the affine head the logits are
                      b + V h, so this is the representation the output layer actually reads.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bernoulli_neural import load_model  # noqa: E402

torch.set_num_threads(3)
ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / "outputs/bernfac_20261020/models/neural"


def spectrum(model, n_rows=2000, seed=0, k=None):
    idx = np.sort(np.random.default_rng(seed).choice(model.n, n_rows, replace=False))
    with torch.no_grad():
        L = model.decoder(model.Z[torch.as_tensor(idx)]).to(torch.float64).numpy()
    s = np.linalg.svd(L, compute_uv=False)
    return s[: k or len(s)]


def main():
    d = pd.read_csv(OUT / "tables/selection.csv")
    ch = d[d.chosen_restart]
    best = ch.loc[ch.val_nll.idxmin()]
    name = best["name"]
    model, hist, extra = load_model(OUT / f"fits/{name}.pt")
    shutil.copy(OUT / f"fits/{name}.pt", OUT / "fits/best.pt")
    Z = model.Z.detach().numpy().astype(np.float32)
    with torch.no_grad():
        Hh = torch.cat([model.decoder.h(model.Z[s:s + 4096]) for s in range(0, model.n, 4096)]).numpy()
    np.save(OUT / "embedding.npy", Z)
    np.save(OUT / "embedding_hidden.npy", Hh.astype(np.float32))
    res = json.loads((OUT / f"results/{name}.json").read_text())
    for split in ("val", "test"):
        pd.DataFrame(res[f"{split}_calibration"]).to_csv(OUT / f"tables/best_calibration_{split}.csv", index=False)
    cfg = res["config"]
    out = dict(selected=name, selection_rule="lowest validation NLL among configurations (restart within a "
                                             "configuration chosen by lowest training objective F)",
               prediction_mode="map_plugin", estimator="joint MAP auto-decoder (no encoder; not a VAE)",
               val=dict(nll=res["val_nll"], auc=res["val_auc"], brier=res["val_brier"]),
               test=dict(nll=res["test_nll"], auc=res["test_auc"], brier=res["test_brier"]),
               final_F=res["final_F"], iterations=res["iterations"], termination=res["termination"],
               monotone=res["monotone"], fit_seconds=res["seconds"], config=cfg,
               affine_rank_bound=(cfg["hidden_widths"][-1] + 1) if cfg["head_mode"] == "affine" else None,
               embedding=dict(file="embedding.npy", shape=list(Z.shape), what="fitted Z (n x d)"),
               embedding_hidden=dict(file="embedding_hidden.npy", shape=list(Hh.shape), what="h(Z), n x H"),
               row_order="rows of outputs/bernfac_20261020/full/data/Y.npy",
               precision="float32 tensors, float64 accumulation; float64 reference only in tests/synthetic",
               scoring="NLL/Brier on all val/test cells; AUC on a 2M subsample (rng seed 0); 10 equal-width bins")
    (OUT / "best.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float)[:1500])
    # spectra: best affine and best nonlinear (pg_gem)
    sp = {}
    H = None
    for head in ("affine", "nonlinear"):
        c = ch[(ch.head_mode == head) & (ch.optimizer_mode == "pg_gem")]
        if not len(c):
            continue
        nm = c.loc[c.val_nll.idxmin(), "name"]
        m, _, _ = load_model(OUT / f"fits/{nm}.pt")
        H = m.decoder.H if H is None else H
        s = spectrum(m, k=400)
        sp[nm] = dict(head_mode=head, H=m.decoder.H, s=s.tolist(),
                      rank_rel_1e6=int((s > 1e-6 * s[0]).sum()),
                      energy_99=int(np.searchsorted(np.cumsum(s ** 2) / np.sum(s ** 2), 0.99) + 1))
        print(nm, "numerical rank (s > 1e-6 s1):", sp[nm]["rank_rel_1e6"], "99% energy:", sp[nm]["energy_99"])
    (OUT / "tables/logit_spectra.json").write_text(json.dumps(dict(n_rows=2000, seed=0, H=H, spectra=sp)))


if __name__ == "__main__":
    main()
