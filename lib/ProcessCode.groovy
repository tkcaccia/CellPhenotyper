import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths

/** Cache identity for source files actually referenced by a pipeline module.
 * No host Python, model import, network access or filesystem metadata shortcut.
 * External executable environments/weights are not a source-code identity.
 */
class ProcessCode {
    static String fingerprint(def projectDirectory, String module, def parameters) {
        def paths = dependencies(projectDirectory, module, parameters)
        PipelineHelpers.contentFingerprint(paths)
    }

    static List<Path> dependencies(def projectDirectory, String module, def parameters) {
        if (!(module ==~ /[A-Za-z0-9_]+/))
            throw new IllegalArgumentException('Invalid process module name')
        Path project = real(projectDirectory)
        Path source = project.resolve("modules/${module}.nf")
        if (!Files.isRegularFile(source)) throw new IllegalArgumentException("Missing module: ${module}")
        def selected = new LinkedHashSet<Path>()
        def queue = [source, project.resolve('lib/ProcessCode.groovy'), project.resolve('lib/PipelineHelpers.groovy')]
        def text = Files.readString(source)
        // These are the same project-relative script parameters interpolated
        // by module commands. Never fingerprint interpreter/token parameters.
        def configured = text =~ /\$\{projectDir\}\/\$\{params\.([A-Za-z_][A-Za-z_0-9]*)\}/
        configured.each { match ->
            def key = match[1]
            if (!key.endsWith('_script')) return
            def value = parameters[key]
            if (!(value instanceof CharSequence) || !value.toString().trim())
                throw new IllegalArgumentException("Missing configured source parameter: ${key}")
            // Mirror the module's literal projectDir/value path construction.
            Path path = Paths.get(project.toString() + '/' + value.toString())
            if (!Files.isRegularFile(path))
                throw new IllegalArgumentException("Configured source file is missing: ${key}")
            queue.add(path)
        }
        while (queue) {
            Path path = queue.remove(0).toRealPath()
            if (!selected.add(path)) continue
            // Hash shared Groovy helper code, but do not mistake other stages'
            // dependency declarations inside it for this module's sources.
            if (path.fileName.toString().endsWith('.groovy')) continue
            String body = Files.readString(path)
            if (path.fileName.toString().endsWith('.py')) {
                // Local Python imports are flat bin modules in this project.
                // Conservative lexical collection is checked against Python's
                // AST in tests; standard/installed packages are not bundled code.
                def from = body =~ /(?m)^\s*from\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s+import\b/
                from.each { match -> addImport(queue, path.parent, project, match[1]) }
                def imports = body =~ /(?m)^\s*import\s+([^\r\n#;]+)/
                imports.each { match ->
                    match[1].split(',').each { item ->
                        def name = item.trim().split(/\s+/)[0]
                        if (name ==~ /[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*/)
                            addImport(queue, path.parent, project, name)
                    }
                }
            }
            // Covers explicit R source helpers, Python subprocess helpers,
            // static resource schemas and lists of source filenames in modules.
            def literals = body =~ /["']([^"'\r\n]*\.(?:py|R|sh|json))["']/
            literals.each { match ->
                String relative = match[1]
                if (relative.startsWith('${projectDir}/')) relative = relative.substring(14)
                if (relative.contains('$') || relative.contains('{') || relative.contains('://')) return
                if (Paths.get(relative).isAbsolute() || relative.split('/').contains('..')) return
                def candidates = [path.parent.resolve(relative), project.resolve(relative),
                                  project.resolve('bin').resolve(relative), project.resolve('resources').resolve(relative)]
                def local = candidates.find { Files.isRegularFile(it) }
                if (local != null) queue.add(local)
            }
        }
        selected.toList().sort { a, b -> a.toString() <=> b.toString() }
    }

    static String directoryFingerprint(List paths) {
        def directories = paths.flatten().findAll { value ->
            if (!(value instanceof Path)) throw new IllegalArgumentException('Expected a process path input')
            Files.isDirectory(real(value))
        }
        // Hugging Face snapshots commonly link individual weight files into
        // their blob cache. Hash those actual bytes; reject directory escapes
        // and cycles. Portable profile receipts retain the stricter default.
        PipelineHelpers.contentFingerprint(directories, true)
    }

    private static void addImport(List queue, Path parent, Path project, String name) {
        String relative = name.replace('.', '/')
        def candidates = [parent.resolve(relative + '.py'), project.resolve('bin').resolve(relative + '.py'),
                          parent.resolve(relative + '/__init__.py'), project.resolve('bin').resolve(relative + '/__init__.py')]
        def path = candidates.find { Files.isRegularFile(it) }
        if (path != null) queue.add(path)
    }

    private static Path real(def value) {
        Path path = value instanceof Path ? value : Paths.get(value.toString())
        // First call unwraps Nextflow TaskPath; second resolves a root symlink.
        path.toRealPath().toRealPath()
    }
}
