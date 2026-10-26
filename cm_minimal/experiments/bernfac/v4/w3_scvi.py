"""R11: scVI (scvi-tools, default SCVI model) on raw counts of the 2,000 R1 HVGs; latent mean, 20 dims.

No batch covariate: library mixing is one of the evaluation metrics, so it must not be removed by
construction (the same holds for every other representation). CPU, 4 threads.
Output: v4/cache/w3_R11_raw.npz (E, info)
"""
import json
import time

import numpy as np
import torch

from w3_common import CACHE, hvg_index, load_counts

torch.set_num_threads(4)


def main(max_epochs=400):
    import anndata as ad
    import scvi
    scvi.settings.seed = 0
    scvi.settings.num_threads = 4
    X, _ = load_counts()
    idx = hvg_index(X)
    a = ad.AnnData(X[:, idx].astype(np.float32).tocsr())
    scvi.model.SCVI.setup_anndata(a)
    m = scvi.model.SCVI(a, n_latent=20)
    t0 = time.time()
    m.train(max_epochs=max_epochs, accelerator="cpu", devices=1, early_stopping=True,
            check_val_every_n_epoch=5, batch_size=256, enable_progress_bar=False)
    secs = time.time() - t0
    E = m.get_latent_representation()
    hist = m.history
    el = hist.get("elbo_validation")
    info = dict(model="scvi.model.SCVI defaults (n_hidden 128, n_layers 1, gene_likelihood zinb)",
                n_latent=20, n_genes=int(len(idx)), batch_key=None, max_epochs=max_epochs,
                epochs_run=int(len(hist["elbo_train"])), train_seconds=round(secs, 1),
                scvi_version=scvi.__version__,
                final_elbo_train=float(hist["elbo_train"].iloc[-1, 0]),
                final_elbo_validation=None if el is None else float(el.iloc[-1, 0]))
    np.savez_compressed(CACHE / "w3_R11_raw.npz", E=E.astype(np.float32), info=json.dumps(info))
    print(json.dumps(info, indent=1), flush=True)


if __name__ == "__main__":
    main()
