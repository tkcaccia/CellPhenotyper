// Pure input naming helpers. Kept outside main.nf to keep its generated Groovy
// script below the JVM's 65,535-character constant limit.
def supportedImageSuffixes() {
  [
    [suffix: '.ome.tif', priority: 80], [suffix: '.ome.tiff', priority: 75],
    [suffix: '.btf', priority: 70], [suffix: '.czi', priority: 69],
    [suffix: '.svs', priority: 68], [suffix: '.ndpi', priority: 67],
    [suffix: '.scn', priority: 66], [suffix: '.mrxs', priority: 65],
    [suffix: '.vms', priority: 64], [suffix: '.vmu', priority: 63],
    [suffix: '.tif', priority: 60], [suffix: '.tiff', priority: 55],
    [suffix: '.png', priority: 50], [suffix: '.jpg', priority: 45],
    [suffix: '.jpeg', priority: 40]
  ]
}

def detectImageSuffix(String fileName) {
  def lower = (fileName ?: '').toLowerCase()
  supportedImageSuffixes().find { lower.endsWith(it.suffix as String) }?.suffix ?: ''
}

def imageSuffixPriority(String suffix) {
  (supportedImageSuffixes().find { it.suffix == suffix }?.priority ?: 0) as int
}

def deriveSampleId(imageFile) {
  def fileObj = imageFile instanceof File ? imageFile : new File(imageFile.toString())
  def name = fileObj.name
  def suffix = detectImageSuffix(name)
  if (suffix) return name.substring(0, name.length() - suffix.length())
  def dot = name.lastIndexOf('.')
  dot > 0 ? name.substring(0, dot) : name
}

def extractCziRegionLabel(String rawName) {
  def matcher = ((rawName ?: '') =~ /(?i)(ScanRegion\d+)/)
  matcher.find() ? matcher.group(1) : ''
}

def extractCziRegionOrder(String rawName) {
  def matcher = ((rawName ?: '') =~ /(?i)ScanRegion(\d+)/)
  if (!matcher.find()) return Integer.MAX_VALUE
  try { matcher.group(1).toInteger() } catch (Throwable ignored) { Integer.MAX_VALUE }
}

def buildSampleId(imageFile, String regionLabel = '') {
  def baseId = deriveSampleId(imageFile)
  def rawId = regionLabel ? "${baseId}__${regionLabel}" : baseId
  rawId.replaceAll(/[^A-Za-z0-9._-]+/, '_').replaceAll(/^_+|_+$/, '') ?: 'sample'
}
