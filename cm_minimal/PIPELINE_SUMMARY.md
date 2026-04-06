# ConnectionMiner (Python) Pipeline Summary

## Flow

1. **Load**
`cm_minimal.loaders.cm_load_inputs` (or `cm_load_inputs_pg`) reads preMN/MN expression, cluster labels, covariates, connectome, and builds:
- `raw.G_cells` (cells x shared genes)
- `raw.P_constraints_cells`
- `raw.C_counts`, `raw.C_mask`
- metadata (`Ncells`, `Ntypes`, names, lineage, motor pool)

2. **Preprocess**
`cm_minimal.preprocess.cm_preprocess_binary` (or `cm_preprocess_pg`) builds metacells within identical constraint signatures:
- HVG/solver-gene selection
- cell -> metacell assignments
- metacell-level gene probabilities `G_metacell_p`
- metacell-level constraints `P_constraints_metacell`

3. **Solve**
`cm_minimal.solver.cm_solve` alternates:
- `beta` update via multiplicative weighted regression
- `P` update via two-pass entropic Sinkhorn-style optimization

Outputs include `P`, `beta`, per-iteration objectives, and reconstructed connectome.

4. **Post-process**
`cm_minimal.postprocess.cm_build_type_gene_probabilities` computes full type-gene probabilities and identifiability metrics.

5. **Export**
`cm_minimal.exports` can write:
- `type_gene_probabilities.xlsx`
- `synaptic_interaction_table.xlsx` (pruned per synapse)

6. **Visualize**
`cm_minimal.viz` writes diagnostics to `run_dir/viz/`:
- constraint diagnostics
- metacell heatmaps and diagnostics
- UMAP panels
- connectome fit
- identifiability
- type heatmap

## Run Commands

```bash
python3 -m cm_minimal.run --mode binary
python3 -m cm_minimal.run --mode binary --smoke --max-cells 500 --max-genes 500 --num-iter 2
python3 -m cm_minimal.test_binary_smoke
python3 -m cm_minimal.test_export_one_iter
```

## Output Layout

`cm_minimal/runs/run_YYYYMMDD_HHMMSS/`
- `prep.mat`, `cm.mat`
- `run_manifest.json`, `run_manifest.mat`
- `solver_objectives.txt`
- optional Excel exports
- `viz/*`
