process RUN_GIGATIME_NUCLEAR_QC {
    tag "${sample_id}"
    label 'compute_medium'

    publishDir "${params.outdir_base}/05_gigatime/${sample_id}", mode: (params.publish_dir_mode ?: 'rellink'), overwrite: true

    cpus 1
    memory '4 GB'
    time '2h'

    input:
    tuple val(sample_id), path(gigatime_dir), path(nuclei_mask_tif)

    output:
    tuple val(sample_id), path("${sample_id}_gigatime_nuclear_qc.json"), emit: report
    tuple val(sample_id), path("${sample_id}_gigatime_nuclear_qc.png"), emit: preview
    tuple val(sample_id), path("${sample_id}_gigatime_shadow_qc.png"), emit: shadow_preview

    script:
    def qcScript = "${projectDir}/bin/gigatime_nuclear_qc.py"
    def codeDigest = java.security.MessageDigest.getInstance('SHA-256')
    codeDigest.update(new File(qcScript).bytes)
    def codeFingerprint = codeDigest.digest().encodeHex().toString()
    """
    set -euo pipefail
    echo "[INFO] GigaTIME nuclear and shadow QC code fingerprint: ${codeFingerprint}"
    python "${qcScript}" \\
      --image "${gigatime_dir}" \\
      --nuclei "${nuclei_mask_tif}" \\
      --output "${sample_id}_gigatime_nuclear_qc.json" \\
      --min-auc ${params.gigatime_nuclear_qc_min_auc} \\
      --min-dice ${params.gigatime_nuclear_qc_min_dice} \\
      --min-edge-ratio ${params.gigatime_nuclear_qc_min_edge_ratio} \\
      --min-boundary-contrast ${params.gigatime_nuclear_qc_min_boundary_contrast} \\
      --min-local-auc ${params.gigatime_nuclear_qc_min_local_auc} \\
      --min-local-pixels ${params.gigatime_nuclear_qc_min_local_pixels} \\
      --mode ${params.gigatime_nuclear_qc_mode}
    """
}
