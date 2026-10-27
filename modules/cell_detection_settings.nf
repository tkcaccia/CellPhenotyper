// Explicit primary detector choice; do not infer the cell population from hardware.
def resolveCellDetectionSettings(Map params, String compute_device) {
    def truth = { value -> value != null && value.toString().trim().toLowerCase() in ['true','1','yes','y','on'] }
    def legacyStarDist = params.cell_consensus_enable != null && !truth(params.cell_consensus_enable)
    def mode = (params.cell_detection_mode ?: (legacyStarDist ? 'stardist' : 'cellvitpp')).toString().trim().toLowerCase()
    if (!(mode in ['stardist','cellvitpp']))
        throw new IllegalArgumentException("Invalid --cell_detection_mode '${mode}'. Choose exactly one: cellvitpp or stardist. Combined consensus is no longer a pipeline detector choice.")
    if (mode == 'stardist' && truth(params.cell_identification_enable))
        throw new IllegalArgumentException('StarDist and CellViT++ cannot run together. Choose --cell_detection_mode cellvitpp for typed cells, or set --cell_identification_enable false for StarDist nuclei.')
    if (mode == 'cellvitpp' && compute_device != 'gpu')
        throw new IllegalArgumentException("cell_detection_mode=${mode} requires GPU execution. Choose --compute_device gpu, or explicitly choose --cell_detection_mode stardist for StarDist; the detector is never changed automatically.")
    return [mode:mode,run_stardist:mode == 'stardist',
            run_cellvitpp:mode == 'cellvitpp',run_hovernet:false,
            primary_cellvitpp:mode == 'cellvitpp']
}
