import groovy.json.JsonSlurper

// Kept outside the main workflow to stay below the Groovy script constant limit.
def writePipelineExecutionReport(Map pipelineParams, workflowMetadata, baseDirPath) {
def outdir = pipelineParams.outdir_base ?: 'results'
def executionDir = new File("${outdir}/00_execution")
executionDir.mkdirs()
def reporterScript = new File("${baseDirPath}/bin/write_pipeline_execution_reports.py").canonicalPath
def analysisContractFile = new File(executionDir, 'analysis_contract.json')
def analysisContractReport = [:]
try {
  if (analysisContractFile.exists()) {
    analysisContractReport = (Map) new JsonSlurper().parse(analysisContractFile)
  }
} catch (Throwable t) {
  println "WARN: failed to read analysis contract for execution report: ${t.message}"
}
def scientificRouteReport = (analysisContractReport.scientific_route instanceof Map)
  ? analysisContractReport.scientific_route
  : [:]
def stageWindowReport = (analysisContractReport.stage_window instanceof Map)
  ? analysisContractReport.stage_window
  : [:]
def reportStartPoint = (stageWindowReport.start ?: pipelineParams.start_point ?: 'convert').toString()
def reportEndPoint = (stageWindowReport.end ?: pipelineParams.end_point ?: 'cluster_geojson').toString()
def reportCellDetectionMode = (scientificRouteReport.cell_detection_mode ?: pipelineParams.cell_detection_mode ?: 'not_applicable').toString()
def reportAnalysisIntent = (analysisContractReport.analysis_intent ?: pipelineParams.analysis_intent ?: 'exploratory').toString()
def reportUni2SamplingMode = (scientificRouteReport.uni2_sampling_mode ?: pipelineParams.uni2_sampling_mode ?: 'cells').toString()
def cmd = [
'python3',
reporterScript,
'--outdir', new File(outdir).canonicalPath,
'--run-name', workflowMetadata.runName,
'--success', workflowMetadata.success.toString(),
'--start-point', reportStartPoint,
'--end-point', reportEndPoint,
'--image-input', (pipelineParams.image_input ?: '').toString(),
'--roi-geojson', (pipelineParams.roi_geojson ?: '').toString(),
'--cell-detection-mode', reportCellDetectionMode,
'--analysis-intent', reportAnalysisIntent,
'--uni2-sampling-mode', reportUni2SamplingMode,
'--analysis-contract', analysisContractFile.exists() ? analysisContractFile.canonicalPath : ''
]
try {
def pb = new ProcessBuilder(cmd)
pb.redirectErrorStream(true)
def proc = pb.start()
def output = proc.inputStream.getText('UTF-8').trim()
def rc = proc.waitFor()
if (output) {
  println output
}
if (rc != 0) {
  println "WARN: execution report writer exited with code ${rc}"
}
} catch (Throwable t) {
println "WARN: failed to write execution reports: ${t.message}"
}
if (workflowMetadata.success) {
println "PIPELINE COMPLETED SUCCESSFULLY"
println "Stage window: ${reportStartPoint} -> ${reportEndPoint}"
if (reportEndPoint in ['cluster_geojson', 'neoplastic_section', 'titan', 'pathofmpred']) {
  println "Cluster GeoJSON output dir: ${pipelineParams.outdir_base}/15_cluster_geojson"
}
}
}
