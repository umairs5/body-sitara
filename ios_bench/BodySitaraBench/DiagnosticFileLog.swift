import Foundation

/// Crash-surviving diagnostic log -- writes every `appendLog` line straight
/// to a file in the app's Documents folder (visible in the iOS Files app,
/// under "On My iPhone" > BodySitaraBench, no dev tools/Mac/Xcode needed),
/// using a raw POSIX `write()` + `fsync()` per line rather than buffered
/// Swift/Foundation I/O.
///
/// WHY THIS EXISTS (2026-08-19): the in-memory `log: [String]` array
/// BenchmarkView already keeps is lost the instant the process dies --
/// exactly what happened on a real device SIGSEGV, where the on-screen
/// panel only ever showed the last ~16 lines before the app vanished, and
/// the only path back to a root cause was a slow, occasionally-absent
/// system `.ips` crash report (see BackgroundReconstructor.swift's/
/// LightmapExtractor.swift's `withUnsafeMutableBufferPointer` crash-fix
/// doc comments for the first bug found this way). A durable, per-line-
/// flushed file sidesteps that dependency entirely: whatever ran before
/// the crash is already safely on disk the moment each `appendLog` call
/// returns, independent of whether iOS ever surfaces (or promptly
/// surfaces) a symbolicated crash report for a sideloaded/ad-hoc build.
///
/// WHY RAW POSIX I/O, NOT `FileHandle`/`String(contentsOf:)`-style APIs:
/// Foundation's higher-level file APIs can buffer internally, and a
/// buffered write that hasn't reached the kernel yet is exactly as fragile
/// as the in-memory array this is meant to replace. `write(2)` handed a
/// valid fd copies the bytes into the kernel's page cache immediately (survives
/// this PROCESS crashing), and `fsync(2)` additionally forces that page
/// cache to physical storage (survives a full device crash/reboot, not
/// just this process) -- the strongest durability guarantee available
/// without adding a third-party dependency.
final class DiagnosticFileLog {
    static let shared = DiagnosticFileLog()

    /// One fixed path per install (not per-run) -- runs accumulate,
    /// separated by a header line, so a single Files-app pull captures the
    /// whole session's history rather than only the most recent run. A
    /// fresh app install / Documents wipe starts a new file; nothing here
    /// depends on that path being stable across installs.
    let fileURL: URL = {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("diagnostic_log.txt")
    }()

    /// POSIX file descriptor, opened once and kept open for the process
    /// lifetime -- `open()`ing fresh per line would be safe but needlessly
    /// slow for what can be a 300-iteration-per-stage logging cadence;
    /// `write`+`fsync` per call is already the expensive, deliberate part.
    private var fd: Int32 = -1
    /// Every access (open-on-first-use, the fd itself, and each write) is
    /// serialized through this lock -- `appendLog` is called from both the
    /// main actor (UI-triggered lines) and `Task.detached` background
    /// closures (the per-frame `[diag]`/progress lines inside the heavy
    /// pipeline stages), so concurrent writers to the same fd are a real,
    /// not theoretical, possibility.
    private let lock = NSLock()

    private init() {}

    /// Appends one line (newline-terminated) and durably flushes it before
    /// returning. Never throws -- a logging failure must not be allowed to
    /// crash or otherwise perturb the benchmark run it exists to diagnose;
    /// worst case, this specific line is silently dropped and `print(line)`
    /// (BenchmarkView.appendLog's other sink) still ran.
    func append(_ line: String) {
        lock.lock()
        defer { lock.unlock() }

        if fd < 0 {
            let path = fileURL.path
            // O_APPEND makes every write() atomic-at-the-OS-level relative
            // to the current end of file, so concurrent callers (main actor
            // + a detached task) can't interleave and corrupt each other's
            // line -- the kernel serializes the position update itself,
            // which a manual seek-then-write from two threads could not
            // guarantee.
            fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0o644)
            guard fd >= 0 else { return }
            let header = "\n=== session start \(ISO8601DateFormatter().string(from: Date())) ===\n"
            writeRaw(header)
        }
        writeRaw(line + "\n")
    }

    /// Raw `write()` + `fsync()`. Must be called with `lock` already held.
    private func writeRaw(_ s: String) {
        guard fd >= 0 else { return }
        var bytes = Array(s.utf8)
        bytes.withUnsafeBufferPointer { buf in
            guard let base = buf.baseAddress else { return }
            var written = 0
            // A single write() is not guaranteed to consume the whole
            // buffer (POSIX allows a short write, e.g. if interrupted by a
            // signal) -- loop until every byte is actually accepted by the
            // kernel rather than assuming one call suffices, matching the
            // durability goal this type exists for.
            while written < buf.count {
                let n = write(fd, base + written, buf.count - written)
                if n <= 0 { break }
                written += n
            }
        }
        fsync(fd)
    }
}
