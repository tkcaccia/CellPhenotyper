/** Read-only host capability probes shared by workflow entry points. */
class HostRuntime {
    static String normalizeArch(def raw) {
        def value = (raw ?: '').toString().trim().toLowerCase()
        if (value in ['x86_64', 'amd64', 'x64', 'x86-64']) return 'amd64'
        if (value in ['aarch64', 'arm64', 'arm64v8', 'arm64/v8', 'armv8', 'armv8l']) return 'arm64'
        if (value == 'auto') return 'auto'
        value ?: 'unknown'
    }

    static String commandOutput(String command) {
        try {
            def process = ['bash', '-lc', command].execute()
            if (!process.waitFor(2, java.util.concurrent.TimeUnit.SECONDS)) {
                process.destroyForcibly()
                return ''
            }
            process.exitValue() == 0 ? process.in.text.trim() : ''
        } catch (Throwable ignored) { '' }
    }

    static boolean commandSucceeds(String command) {
        try {
            def process = ['bash', '-lc', command].execute()
            if (!process.waitFor(2, java.util.concurrent.TimeUnit.SECONDS)) {
                process.destroyForcibly()
                return false
            }
            process.exitValue() == 0
        } catch (Throwable ignored) { false }
    }

    static int positiveInt(def raw, int fallback) {
        try {
            def parsed = (raw ?: fallback).toString().trim().toInteger()
            parsed > 0 ? parsed : fallback
        } catch (Throwable ignored) { fallback }
    }

    static int memoryGb() {
        def commands = [
            'awk \'/MemTotal/ {printf "%d", int($2/1024/1024)}\' /proc/meminfo',
            'free -g | awk \'/^Mem:/ {print $2}\'',
            'sysctl -n hw.memsize 2>/dev/null | awk \'{printf "%d", int($1/1024/1024/1024)}\'',
            'awk \'$1 != "max" && $1 ~ /^[0-9]+$/ {printf "%d", int($1/1024/1024/1024)}\' /sys/fs/cgroup/memory.max 2>/dev/null',
            'awk \'$1 ~ /^[0-9]+$/ && $1 < 9000000000000000000 {printf "%d", int($1/1024/1024/1024)}\' /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null'
        ]
        def candidates = commands.collect { commandOutput(it) }.findAll { it && it ==~ /\d+/ }
        if (!candidates) return 8
        try {
            return Math.max(1, candidates.collect { it.toInteger() }.findAll { it > 0 }.min())
        } catch (Throwable ignored) { return 8 }
    }
}
