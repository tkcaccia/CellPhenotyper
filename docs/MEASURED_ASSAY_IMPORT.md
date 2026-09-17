# Measured assays: immutable import and SpatialData linkage

Measured assay values are a separate modality. They never replace the pipeline's
H&E-predicted marker scores, UNI-2 features, canonical geometry, or cell IDs.
The importer performs no registration, matching, normalization, imputation, or
biological validation. Review the gates in
[VIRTUAL_MARKER_VALIDATION_PROTOCOL.md](VIRTUAL_MARKER_VALIDATION_PROTOCOL.md).

## Cell-level workflow

Use the final canonical profile directory that will also be exported. The import
is bound to the exact profile manifest and table hashes; modifying that profile
later requires a new import against the new canonical product.

```bash
python bin/integrate_measured_assay.py \
  --cell-profiles /path/to/final_cell_profiles \
  --measured /path/to/explicitly_matched_assay.csv \
  --assay-manifest /path/to/assay_contract.json \
  --outdir /path/to/new_measured_package
```

Input CSV or Parquet must contain:

- `sample_id`, exact canonical `cell_uid`, and unique `assay_observation_id`;
- finite `assay_x`, `assay_y` in the declared source frame and units;
- finite `registered_x_um`, `registered_y_um` in original-slide micrometres;
- each marker's declared source column. Empty numeric fields/NaNs are preserved.

Every supplied match must be one-to-one, independently established, and within
the locked maximum physical matching distance. Missing eligible cells remain in
the output and matching denominator. Marker missingness is distinct from an
unmatched cell. Duplicate or foreign IDs fail; correlation never establishes a
match. Cell-level data must be same-section registered data. Serial sections and
Visium must not be imported as individual-cell measurements.

## Assay contract

The following is a **schema template, not experimental data or an approved
registration**. Replace every placeholder and `null` with independently reviewed
evidence. The unfilled template intentionally fails validation. Set the two
independence booleans to `true` only when those declarations are true.

```json
{
  "schema_version": "cellphenotyper.measured_assay_input.v1",
  "assay_id": "<unique immutable assay identifier>",
  "assay_type": "<protein_imaging or declared assay type>",
  "assay_platform": "<instrument/platform>",
  "assay_protocol": "<locked independent assay protocol identifier>",
  "panel_version": "<panel identifier/version>",
  "sample_id": "<exact canonical sample_id>",
  "observation_unit": "cell",
  "reference_design": "same_section_registered",
  "coordinates": {
    "source_frame": "<explicit assay coordinate frame>",
    "source_units": "pixel",
    "target_frame": "original_slide_micrometres",
    "target_units": "um"
  },
  "registration": {
    "status": "<passed only after reviewed registration>",
    "locked_before_prediction_review": null,
    "method": "<locked registration method>",
    "independent_landmarks": null,
    "median_error_um": null,
    "p95_error_um": null,
    "acceptance_p95_um": null,
    "target_observations_sha256": "<canonical cell_profiles.parquet SHA256>",
    "source_observations_sha256": "<exact matched assay table SHA256>",
    "transform_artifact": {
      "path": "<registration transform artifact>",
      "sha256": "<artifact SHA256>"
    }
  },
  "matching": {
    "method": "provided_one_to_one_ids",
    "independent_of_predicted_markers": null,
    "protocol": "<prespecified mapping protocol>",
    "eligible_observation_count": null,
    "minimum_matched_fraction": null,
    "matched_fraction": null,
    "maximum_match_distance_um": null
  },
  "source_artifacts": [
    {
      "path": "<independent assay source artifact>",
      "sha256": "<artifact SHA256>",
      "role": "raw_assay_measurements"
    }
  ],
  "markers": [
    {
      "name": "<marker name>",
      "column": "<numeric source table column>",
      "units": "<assay-specific units>",
      "measurement_type": "<protein intensity, transcript count, etc.>",
      "normalization": "<independent preprocessing definition, or none>"
    }
  ]
}
```

`source_units` accepts `pixel` or `um`. Paths inside provenance records may be
absolute or relative to the assay manifest. SHA256 values are verified against
the supplied files, not just recorded. The target hash must match the table
selected from the canonical profile manifest (Parquet is preferred; CSV is the
fallback). Registration requires at least three independent landmarks,
`0 <= median_error_um <= p95_error_um <= acceptance_p95_um`, and a positive
acceptance threshold. Matching fractions must describe the actual provided
matches over the complete supplied registry, with a positive locked minimum.

## Output package

Each new output directory contains:

- `measured_observations.csv` and `.parquet`: canonical-left-aligned cells,
  matching status, physical coordinates, and separate
  `measured__<URL-escaped assay_id>__<URL-escaped marker>` columns;
- `measured_values.npy`: ordered `N × markers` float64 matrix, with original
  values and NaNs intact;
- `measured_rows.csv`: ordered `sample_id,cell_uid` row index;
- `measured_assay_manifest.json`, schema `cellphenotyper.measured_assay.v1`:
  file hashes, canonical lineage, assay definitions, marker units, registration
  evidence, counts, and explicit claim limitations.

An existing output directory is never overwritten. Do not place this output
inside the immutable canonical profile directory. Import size guards default to
5 million observations and 50 million matrix values and can be set explicitly
using `--max-observations` and `--max-matrix-values`.

## Optional SpatialData export

Add `--measured-assay /path/to/new_measured_package` to the usual
`bin/export_spatialdata.py` command. Repeat it for distinct assays. Every assay ID
must be unique within one export.
Optionally pass `--sample-id` to require the exact expected canonical specimen
identity before any output is written.

Each package becomes a separate `measured_assay_<assay-ID hash>` table linked to
the `canonical_cells` labels through the original integer `instance_id`.
Measured values occupy that table's `X`, in float64 with NaNs preserved. `var`
records marker names, units, normalization and measurement type; `obs` preserves
all eligible cells and explicit matching/missingness. The existing `cells` table
and its predicted feature blocks remain separate and unchanged.

Before writing, export checks all package output hashes, exact canonical profile
manifest/table lineage, row order, cell identities, coordinates, registration
and matching gates, and matrix/table value agreement. Export does not require
original raw assay files to remain mounted: their import-time verified hashes
remain provenance. The exported measured matrix is read back and compared in
bounded row blocks, including every NaN. If the profile contains image and label
hashes, both exact rasters must match; legacy profiles without them are explicitly
marked as weaker raster provenance.

## Region-level assays

For serial sections or Visium, use `--regions REGISTRY.csv|parquet` and
`--regions-manifest REGISTRY.json` instead of `--cell-profiles`. The registry needs
unique `region_uid`, `sample_id`, `x_um`, `y_um`, and positive `area_um2`.
Its manifest uses `schema_version=cellphenotyper.region_registry.v1`,
`observation_unit=spatial_bin`, `coordinate_system=original_slide_micrometres`,
`sample_id`, `region_count`, and `files` mapping the registry filename to its
SHA256. `region_definition` must declare `kind`, `description`, and a hashed
geometry/region-definition `artifact` with `path` and `sha256`.

The assay table uses `region_uid` instead of `cell_uid`; matching method is
`provided_region_ids`. Use `reference_design=serial_section_region_level` for
serial sections, or `same_section_registered` for a registered same-section
region assay. All registration, hash, coordinate, and matching gates still apply.

## SpatialData linkage for measured regions

Region packages can be exported only with their own explicit registered polygon
shapes. Pass the package and its shape-link manifest together:

```bash
python bin/export_spatialdata.py \
  --profile-dir /path/to/final_cell_profiles \
  --image /path/to/he_crop.tif --labels /path/to/canonical_labels.tif \
  --shift /path/to/shift.json --sample-id '<exact sample_id>' \
  --tissue-geojson /path/to/tissue_domains.geojson \
  --tissue-coordinates crop_pixels \
  --measured-assay /path/to/region_measured_package \
  --measured-region-shapes /path/to/registered_region_bundle/link.json \
  --outdir /path/to/new_spatialdata.zarr
```

If physical calibration requires a passed resolution report, also pass
`--resolution-json`. Both measured options are repeatable. One shape-link is
required for each spatial-bin assay, and every supplied link must be used once;
duplicate assay links and unused links fail. Cell-level packages do not accept
region links. Python callers use `measured_assay=[...]`,
`measured_region_shapes=[...]`, and optionally `expected_sample_id=...`.

The shape-link contract below contains placeholders only, not experimental
registration evidence:

```json
{
  "schema_version": "cellphenotyper.measured_region_shapes.v1",
  "assay_id": "<exact imported assay_id>",
  "sample_id": "<exact exported sample_id>",
  "source_coordinates": "crop_pixels",
  "target_coordinates": "original_slide_micrometres",
  "coordinate_anchor": "geometry_centroid",
  "target_image_sha256": "<exact exported image SHA256>",
  "target_shift_sha256": "<exact shift.json SHA256>",
  "registration_transform_sha256": "<imported registration transform SHA256>",
  "shapes": {
    "path": "registered_regions.geojson",
    "sha256": "<registered polygon GeoJSON SHA256>"
  },
  "registry_table": {
    "path": "regions.parquet",
    "sha256": "<exact imported region-registry table SHA256>"
  },
  "registry_manifest": {
    "path": "regions_manifest.json",
    "sha256": "<exact imported region-registry manifest SHA256>"
  }
}
```

All artifact paths resolve relative to the link JSON's directory, unless
absolute. For workflow staging, keep these files in one directory bundle and
preserve its relative layout. The explicit registry copies must match the
imported hashes; original filesystem locations need not remain mounted.

Registered shapes must be a GeoJSON FeatureCollection with one valid Polygon or
MultiPolygon per region. Each feature must explicitly provide
`properties.region_uid`; IDs must be unique and exactly equal the complete
region registry. GeoJSON feature order may differ because the ID correspondence
is explicit; the output retains registry/table order. The polygon artifact must
be the same hash-bound geometry recorded in the imported region definition.

`source_coordinates` accepts `crop_pixels`, `original_pixels`, or `original_um`.
These map to original-slide micrometres through the declared calibrated image
scale and crop offset, or identity for original micrometres. After this transform,
each polygon's centroid and area must agree with the registry's `x_um`, `y_um`,
and `area_um2`. The full polygon must lie within the exported image crop. The
exporter never infers a frame, clips/subsets regions silently, or substitutes a
nearest cell. Registries using a different coordinate anchor must be explicitly
redefined before import; they are not relabelled as centroids during export.

Each assay receives its own `measured_regions_<assay-ID hash>` ShapesModel and
its own `measured_assay_<assay-ID hash>` table. Table `instance_id` links only to
that assay's shape index, **not** to cell label values. Stable `region_uid`, all
unmatched observations, declared assay values and NaNs are retained. The shape
identity, exact geometry, physical transform and measured matrix are checked
again after writing and reading the SpatialData store.

Same-section cell tables, serial-section/Visium region tables, and H&E-predicted
cell features remain separate. A region signal is never duplicated onto cells
or described as a measured value of an individual cell.

Technical import/export tests do not establish measured-assay interchangeability
or biological accuracy of the virtual markers. Independent, patient-separated,
cellularity-adjusted validation remains necessary.
