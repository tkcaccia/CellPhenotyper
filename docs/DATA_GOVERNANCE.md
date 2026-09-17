# Data governance and privacy requirements

Revision: 2026-09-03

This is a minimum operational policy for CellPhenotyper projects, not a substitute for institutional ethics, information-security, consent or data-transfer approval. The study owner must document the stricter applicable rules before any clinical image is processed.

## Data classification

| Asset | Minimum classification | Reason |
|---|---|---|
| Raw WSI and associated label/macro images | Sensitive research data | Slides and embedded metadata may contain accession numbers, labels or other identifiers. |
| GeoJSON, cell coordinates and TMA core maps | Sensitive derived data | Spatial data can remain linkable to the source specimen even without a patient name. |
| Model embeddings, cluster tables and virtual-marker values | Sensitive derived data | These are specimen-derived measurements, not anonymous merely because they are numeric. |
| QC screenshots and review packets | Same as source data | Tissue appearance, filenames, annotations and visible labels may identify a specimen or study. |
| Execution logs and manifests | Internal research metadata | Absolute paths, usernames, hostnames, sample codes and software locations may be disclosed. |
| Public pretrained model caches | Licensed software/model asset | Access and redistribution depend on upstream terms; caches must not contain user tokens. |
| Private PathoFMPred registry | Restricted model asset | Keep outside public containers, repositories and shared output archives. |

## Before ingestion

1. Confirm ethics approval, consent or waiver, permitted purpose, data location and transfer route.
2. Remove direct identifiers from filenames, directory names, TIFF/OME metadata, slide labels, macro images and associated files before pipeline access.
3. Assign a study-specific sample code through a separately controlled linkage table.
4. Keep the linkage table outside the pipeline input, work, output, cache and log directories.
5. Verify that de-identification did not remove required scientific metadata such as MPP, stain, scanner, site, block and section identity; replace it with non-identifying structured metadata where permitted.
6. Record whether label and macro image series were removed or retained and why.

CellPhenotyper validates image geometry and calibration but is not a de-identification tool. Successful input QC does not prove that protected information is absent.

## Access and execution

| Control | Requirement |
|---|---|
| Authorization | Grant least-privilege access by project role; review membership periodically and remove access when work ends. |
| Authentication | Use named accounts and institution-approved multifactor authentication where available; do not share accounts. |
| Local and remote storage | Use approved encrypted storage. Do not place sensitive runs in consumer-synced folders unless explicitly authorized. |
| Transfer | Use an approved encrypted protocol such as SSH/SFTP or managed institutional transfer and verify checksums after transfer. |
| Scratch | Treat Nextflow `work/`, temporary TIFF/Zarr files and scheduler scratch as sensitive duplicates. Apply the same permissions and cleanup policy as the source. |
| Containers | Do not bake images, annotations, tokens, private models or linkage tables into container layers. |
| Network | Prefer offline model caches for protected runs; document any external model download or telemetry exception. |
| Multi-user host | Restrict process, directory and GPU visibility according to institutional controls; a shared GPU scheduler is not an access-control boundary. |

## Credentials and model access

`tokens.env`, `GHCRtoken.env`, Hugging Face tokens, registry credentials and SSH secrets must remain outside Git history and published results. Restrict token files to the owning account, use read-only/scoped credentials where possible, and rotate credentials after suspected exposure. The repository `.gitignore` reduces accidental commits but is not a secret-management system.

Do not print tokens into Nextflow command lines, reports, logs or notebooks. Before sharing an execution directory, search it for credential names and host-specific secrets. Private or gated checkpoints may be copied only when their upstream license and data-use terms permit it.

## Retention schedule

Every project must replace the placeholders below with approved periods before execution:

| Asset class | Retention owner | Retention period | End-of-life action |
|---|---|---|---|
| Raw images and linkage table | Study custodian | Project-specific | Archive or securely delete under the approved protocol |
| Durable pipeline outputs | Study custodian | Project-specific | Archive immutable release outputs or securely delete |
| Nextflow work and temporary scratch | Pipeline operator | Shortest period compatible with verified completion/resume | Delete only after durable outputs and checksums are verified |
| Execution logs and audit records | Study custodian | Project-specific | Retain with analysis provenance; redact before public release |
| Public model caches | Platform owner | Version lifecycle | Retain by digest or remove when no approved run depends on them |
| Private/gated model assets | Model custodian | License/access term | Remove access and copies when authorization expires |

Deletion must include known scratch, backups and transferred copies according to the approved platform process. Never infer that removing the Nextflow work directory removes all copies.

## Audit record

For every production or manuscript run, retain:

1. Study ID and coded sample IDs.
2. Data source, permitted use and transfer approval reference.
3. Input and ROI hashes, pipeline commit, container digests and model/checkpoint identities.
4. Parameter file, study manifest, analysis contract and evidence-gate result.
5. Start/end timestamps, host or scheduler job IDs, operator and reviewer roles.
6. Output manifest, run traces, failures, restarts, exclusions and human-review decisions.
7. Every export destination and the checksum verified after transfer.
8. Retention or deletion decision and responsible custodian.

Avoid placing direct patient identifiers in this audit record. Use the controlled study linkage process when re-identification is authorized and necessary.

## Sharing and publication

Before any public upload, verify that images, previews, GeoJSON, tables, HTML reports, file paths and embedded metadata are permitted for release. Derived embeddings and coordinates may remain controlled human-subject data. Publish aggregate results only at the granularity allowed by the protocol and verify small-cell or rare-group disclosure risk.

Open-source code does not grant permission to redistribute third-party checkpoints or private models. Record license, access restriction, attribution and redistribution status for each release asset separately.

## Incident and exception handling

Stop transfer or processing after suspected identifier, credential or access exposure. Preserve necessary audit evidence without spreading the exposed content, notify the institutional contact, rotate affected credentials and follow the approved incident process. Do not silently delete evidence or continue on a different machine.

Any policy exception must record the reason, approver, scope, start date, expiry date and compensating control. A pipeline parameter override is not governance approval.
