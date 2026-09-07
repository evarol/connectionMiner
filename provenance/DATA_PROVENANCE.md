# ConnectionMiner curated-data provenance

Snapshot: 7 September 2026. This record freezes the identities of the curated inputs used by the model-validation study. It does not resolve the biological/source-version uncertainties below. Existing raw data and curation products are unchanged. The accompanying `curated_inputs_2026-09-07.json` contains exact paths, resolved symlink destinations, SHA256 checksums, dimensions, canonical target indices, gene identifiers and morphology crosswalk hashes.

## Dataset views and common axes

| View | Transcriptomic sources | C dimensions | Metacells | Single cells |
|---|---|---:|---:|---:|
| A1 / A2 / A3 | P20 MN + P24 pupal VNC preMN | 730 / 1,617 / 1,218 square | 18,334 | 105,752 |
| B1 / B2 / B3 | P45 MN + P48 pupal VNC preMN | 730 / 1,617 / 1,218 square | 12,170 | 88,794 |
| C1 / C2 / C3 | Adult MN + adult VNC preMN | 730 / 1,617 / 1,218 square | 2,454 | 18,975 |
| D | Desplan visual Adult | 741 square | 13,947 | 98,879 |
| E | Chris Doe central-complex Adult | 274 square | 3,670 | 50,148 |

Versions 1, 2 and 3 denote historical motor, all-class expanded pilot, and its VNC-resident induced subset. The viewer shows eight rows A1/B1/C1/A2/B2/C2/D/E, with A3/B3/C3 accessible as resident alternatives. These are views of three anatomical systems, not eleven independent connectomes. The same adult motor connectivity is reused at every RNA stage. P20/P24 and P45/P48 are approximate cross-source developmental pairings.

`C[i,j]` is retained directed connectivity from presynaptic entry i to postsynaptic entry j. Use `W_measured` with C: unmeasured MN-source rows and other unknown entries are not observed zeros. The current fitting target is binary retained connectivity, not synapse counts. Diagonal entries are excluded from evaluation. Expanded targets mix individual neurons, historical aliases and MN aggregates; their number is not a validated count of distinct biological cell types.

`P_constraint[i,k]` is binary permission for metacell k to contribute to target i. It is not a probability and its rows/columns do not sum to one. `G[k,g]` is the mean of the single-cell expression-positive indicator within metacell k. G is therefore bounded in [0,1], retaining assay/detection-depth dependence; it is not calibrated transcription probability. Membership files identify every constituent cell. No metacells, counts, optimal transport or annotation labels were refitted for this snapshot.

## Motor sources and processing

Connectivity and SWCs originate in the local Tony/FANC material and curated motor source mappings. The underlying circuit paper is [Lesser and colleagues, bioRxiv 2023.05.30.542725 version 2](https://www.biorxiv.org/content/10.1101/2023.05.30.542725v2). The peer-reviewed circuit article is [Lesser et al., Nature (2024)](https://doi.org/10.1038/s41586-024-07600-z). The later local roster is not identical to that paper's release. The expanded hybrid preserves the original 730-entry block and adds 887 anatomical roots. Its 1,218-resident subset removes 225 descending and 174 sensory entries; these 399 external-soma targets have no candidates in the current VNC RNA constraints. Their genetic predictions must remain unsupported. Thresholded source zeros can include subthreshold connections. See `outputs/gene_panels_motor_expansion_pilot/report/motor_expansion_study.md` for the release comparison and exact evidence.

Adult preMN input derives from the Goodwin laboratory adult VNC atlas: [Allen et al., eLife 9:e54074 (2020)](https://doi.org/10.7554/eLife.54074), GEO GSE141807. C1 retains the local adult export; C2/C3 use reaggregation of public counts for the same selected cells and metacells. These are expression-source variants, not independent samples. The 161-gene matrices agree, but some broader-gene values differ. Pupal VNC input derives from [Cachero et al., Nature (2026)](https://doi.org/10.1038/s41586-026-10797-w), GEO GSE304221, with source hemilineage and saved coarse-cluster labels. The adult MN and developmental MN inputs are local files under `/Users/erdem/Dropbox/scRNAseq_DGE/Matrix and umap raw files/Merged`, including `matched_clusters_all29_v030625_consensus.xlsx`; precomputed cross-time mappings come from `/Users/erdem/Dropbox/scRNAseq_DGE/code/trajectory_analysis_MN_all29.m`. This study does not fit new optimal transport.

Constraints retain annotated MN clusters, ambiguous few-to-few candidate sets, and residual cluster-0 candidates; preMN hemilineages restrict many-to-many assignments where source evidence establishes a compatible lineage. Unresolved annotations remain explicit rather than becoming confident type labels. Metacells respect the previously curated permission/source-cluster boundaries; small or uncertain groups are not pooled across conflicting labels. Detailed rules, full source locations and membership reasons are in `outputs/gene_panels_motor_expansion_pilot/transcriptome_scope/` and the linked bundle/membership files in the JSON manifest.

An independent comparison of supplied adult preMN metadata with the archived Allen author RNA-cluster mapping found a source-version disagreement for cluster 38: local `lin11A` versus author-code `lin11B` (271 original cells; 170 retained in the current adult joint transcriptome). This comparison establishes neither lineage as biologically correct and does not demonstrate that these labels came from connectivity fitting. The curated G and permission matrices remain unchanged; model fitting and held-out evaluation are conditional on these prepared priors. The active constraint builder explicitly excludes connectivity-derived MN consensus vote labels, but historical manual crosswalks and the exact saved developmental trajectory writer/version are not a fully reconstructed, fold-specific annotation pipeline. Evidence and exact affected-cell records are in `outputs/model_validation_study/upstream_prior_audit.md` and `checks/adult_pred_lineage_origin.json` / `checks/adult_pred_lineage_mismatches.csv` within that study directory.

## Visual and central-complex sources

Visual transcriptomics and the anatomical root/type bridge are local Desplan lab exports under `/Users/erdem/Dropbox/scRNAseq_DGE/Desplan lab`. The source bridge `consolidated_anchor_types.csv.gz` defines 97,105 anatomical contributors across 740 populated target entries; the 741st is an empty uncertain placeholder. Numbered/orphan cluster permissions remain ambiguous after neuronal selection. The previously generated visual C was independently reproduced from source synapses, retaining 24,970,663 within-target synapses. Local expression and annotations, not an inferred reproduction of the student's web model, define these inputs. The visual atlas references include [Özel et al., Nature (2021)](https://doi.org/10.1038/s41586-020-2879-3); exact source exports and population attribution are recorded in the preceding data-expansion and premapping audits; a local label does not establish equivalence to every published atlas population.

CX uses the supplied partially annotated transcriptome and connectivity/primary morphology membership under `/Users/erdem/Dropbox/ConnectionMinder CX` and `/Users/erdem/Dropbox/scRNAseq_DGE/Central complex - Chris Doe`. The 274 retained named connectivity entries have 2,348 primary anatomical contributors. Primary membership is distinguished from alternative candidate annotations. Twenty-five unnamed local SWCs fall outside this named C; candidate labels are not silently promoted to anatomical contributors. Exact collection release/registration is not fully established by the local exports.

## Gene space

The baseline full-common matrices contain 9,549 canonical FlyBase genes. Requiring availability in the public adult expression variant yields 8,927 genes shared across the eight simultaneous views. Missing genes are excluded, never zero-imputed. The broader 25-source assay intersection contains 8,124 genes. Saved HVG rankings and curated sets are preserved in `outputs/gene_panels_motor_expansion_pilot/genes/panels_manifest.json`; the manifest freezes exact gene IDs for each experiment panel. The present screen uses strict all-25 HVGs (70), established strict HVGs (161), curated synaptic genes (217), and consensus HVGs (2,000), filtered to public availability. These are distinct selection rules, not independently replicated discoveries. Common positivity encoding does not remove platform, age or sampling biases.

## Morphology mapping and native access

| System | Target entries with geometry | Distinct mapped geometries / neurons | Native reference |
|---|---:|---:|---|
| Motor historical | 707 / 730 | 707 | Tony SWCs, preserved selected bytes |
| Motor expanded | 1,593 / 1,617 | 1,591 | Same native inventory plus exact source-ID recoveries |
| Motor resident | 1,194 / 1,218 | 1,192 | Exact expanded subset |
| Visual | 740 / 741 | 97,105 | FlyWire LOD1 healed downsampled skeleton Parquet |
| CX | 274 / 274 | 2,348 primary | Supplied native SWCs |

Crosswalks under `outputs/eight_panel_morphology/morphology/{motor,visual,central_complex}/` give one target-to-contributor link per row, with full source paths or archive locators. They map C entries, not individual matrix edges, to skeletons. Motor fractional MN contributor weights and historical geometry aliases are retained. Six source-token recoveries retain explicit release/identifier-precision cautions; current target roots and geometry representative roots are separate fields. Missing contributors are queued, not fabricated. FANC retrieval required authentication unavailable during the curation audit.

Visual native skeletons are stored in `outputs/connectionminer_ready_inputs/morphology/shared_flywire/sk_lod1_783_healed_ds2.parquet`, from [Zenodo 10877326](https://doi.org/10.5281/zenodo.10877326), CC BY 4.0. Its source MD5 is `a4c104776f33ec539ef859064c4de3df`. The crosswalk plus `outputs/eight_panel_morphology/scripts/export_visual_swc.py` exports individual or type-group SWCs. Avoid materializing another 97,105-file copy unnecessarily. Coordinates are native FAFB14.1 nanometers; CX headers indicate nanometers but do not prove matched registration. Motor units/release remain unverified. Each anatomical system is normalized separately for geometric features/display.

Quantitative morphology uses all stored native edges. Simplified viewer overview proxies are display-only. Forty anatomical roots occur in both the visual and CX primary inventories, involving 22 types per system; validation must handle this overlap explicitly. Within motor, repeated stages and source-version aliases likewise require synchronized splits.

## Verification and regeneration

Run from the repository root:

```sh
outputs/connectionminer_ready_inputs/.venv/bin/python provenance/build_manifest.py
```

The JSON checksums make silent input changes detectable. The builder reads `outputs/model_validation_study/scripts/study_data.py`; retain that script and previous output directories when regenerating. Existing audits include `outputs/eight_panel_morphology/checks/independent_geometry_crosswalk_checks.md`, its `final_delivery.json`, and `outputs/model_validation_study/checks/independent_split_audit.json`. They verify stored-data consistency and dimensions, not biological annotation correctness, causal gene interactions or independent replication. Raw matrices and multi-gigabyte morphology are intentionally not added to Git; this commit records provenance and reproducibility code.
