# Native KODAMA graph clustering

This opt-in path clusters the saved **KODAMA feature-dissimilarity graph**.
It is not a physical-neighbourhood graph: edges may connect observations in
separate tissue regions and do not enforce tissue-gap boundaries. It does not
reconstruct a graph from PCA or UMAP coordinates. UMAP remains display-only.
The default `cluster_representation: umap2d` is unchanged.

## Configuration

```yaml
cluster_representation: kodama_graph
cluster_representation_dimensions: 0
cluster_landmark_cells: 0
cluster_algorithm: leiden
cluster_resolution: '0.3'
cluster_primary_variant: standard
cluster_secondary_variant: none
cluster_target_clusters: 0
```

Selecting graph mode automatically requests native export in the KODAMA stage.
`kodama_export_native_graph: true` can instead request the portable artifact
without changing the clustering representation; its default is `false`.
Neither option changes `kodama_ncomp`, PCA dimensions, the native classifier,
or KODAMA's own landmark optimization. Parameter `ncomp` applicability depends
on the actual classifier and is recorded in the export receipt; it must not be
assumed to be a fitted PLS rank for a KNN classifier.

The initial consumer supports Leiden, positive fixed resolution, and the
standard profile only. CLI coordinate-neighbor, landmark-assignment and fine
options are rejected; the Nextflow module passes only applicable graph options
and requires an explicit zero landmark count. `cluster_snn_k`, coordinate
landmark strategy/assignment settings, and fine tuning do not construct or
modify native graph edges. The native neighbor setting is recorded by the
producer, not overridden by the consumer's coordinate SNN parameter.

To preserve the requested K=2 experiment, set `cluster_target_clusters: 2` and
`cluster_forced_count_sensitivity_acknowledged: true`. This remains a
**forced-count sensitivity analysis**, not independently discovered classes.
The same acknowledgement requirement applies to the other representations.

## Portable source and identity

Graph mode requires `kodama_graph.rds` plus `kodama_graph.json`, schema `1.0.0`,
and an explicit graph-available/SHA256 binding in the selected
`kodama_full_*.RData` representation metadata. A legacy file with coincidentally
matching numeric observation IDs cannot acquire this provenance automatically.
Restarting from older results requires rerunning the KODAMA producer with a
compatible native materialization API. Missing, projected/subset, stale, or
unbound graphs fail closed; there is no PCA/UMAP fallback.

The plain RDS contains ordered observation IDs and a directed `dgCMatrix` of
finite nonnegative native corrected distances. The receipt binds its exact
SHA256, ordered IDs, dimensions, explicit edges, source PCA hash, native
parameters, package version/revision and index policy. Native infinite distances
are omitted, self edges are omitted, and finite zero distances are retained.
All input vertices—including isolates—remain present. Exporting all observation
rows does not imply that KODAMA's internal optimization used no landmarks or
within-run label projection; those native algorithm settings remain recorded.

## Affinity, forced counts and uncertainty

For each **stored** directed distance, the consumer uses `affinity = 1/(1+d)`.
Reciprocal edges are symmetrized by maximum affinity (union), with no self loops.
An explicitly stored distance zero becomes affinity one; an absent sparse
entry remains no edge. The rule and source hash appear in clustering tables,
summary and representation-comparison metadata.

Leiden runs on this weighted graph. Isolates retain distinct raw singleton IDs
but always have missing `interpretable_cluster`, even if optional uncertainty
abstention is disabled. Their affinity fraction/margin are unavailable (`NA`),
not zero-confidence measurements or invented memberships.

If a target count requires merging communities, each merge chooses the pair
with the largest positive total cross-community affinity, with deterministic
ID tie-breaking. No centroid/PCA/UMAP distance is used. When no supported merge
can reach the target—such as too many disconnected components or isolates—the
run fails rather than inventing connections. No unsupported splitting is done.

After any forced merge, assignment evidence is recomputed for the **final**
communities. The legacy `assignment_vote_fraction` column contains the fraction
of incident affinity supporting the assigned community; `assignment_vote_margin`
contains `(own affinity - strongest other-community affinity)/total affinity`.
The explicitly recorded score semantics distinguish these graph heuristics
from literal landmark votes and from calibrated probabilities. A negative
margin indicates stronger support for another community. Degree and total
strength are retained. The existing threshold and seed-stability checks flag
ambiguous supported vertices; isolates always abstain.

Seed stability repeats native graph clustering, not graph extraction. With
three seeds the current `0.67` stability threshold requires all three assignments
to agree, because `2/3 < 0.67`. UMAP silhouettes are not computed as native graph
quality. Plotting in a UMAP layout is visualization, not evidence of accuracy.
Portable-fixture and producer/consumer tests verify execution and identity;
biological quality and cross-slide robustness require independent evaluation.
