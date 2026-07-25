import SwiftUI
import CoreML

/// Benchmark harness UI: runs RIFE + big-LaMa inference N times each,
/// reports per-stage timing (build/run/postprocess) and the requested/
/// available compute-unit info from ComputePlanInspector. NOTE: this is a
/// weaker check than true per-operation ANE attribution (see
/// ComputePlanInspector.swift's docstring for why -- MLComputePlan's
/// exact Swift API couldn't be pinned down without real Apple docs
/// access) -- report benchmark numbers with that caveat explicit, not as
/// confirmed ANE-only timing.
struct BenchmarkView: View {
    @State private var log: [String] = []
    @State private var isRunning = false
    static let repetitions = 5 // matches Table 8's 5-repetition methodology

    var body: some View {
        NavigationView {
            VStack {
                Button(isRunning ? "Running..." : "Run Benchmark") {
                    Task { await runBenchmark() }
                }
                .disabled(isRunning)
                .padding()

                ScrollView {
                    Text(log.joined(separator: "\n"))
                        .font(.system(.footnote, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                }
            }
            .navigationTitle("bodySITARA iOS Bench")
        }
    }

    private func appendLog(_ line: String) {
        log.append(line)
        print(line)
    }

    private func runBenchmark() async {
        isRunning = true
        log = []
        appendLog("=== bodySITARA iOS Benchmark (iPhone 15 Pro Max target) ===")
        appendLog("Repetitions: \(Self.repetitions)")

        let config = MLModelConfiguration()
        config.computeUnits = .all // let CoreML pick ANE/GPU/CPU; verified below, not assumed

        do {
            try await runRifeBenchmark(config: config)
        } catch {
            appendLog("RIFE benchmark FAILED: \(error)")
        }

        do {
            try await runLamaBenchmark(config: config)
        } catch {
            appendLog("big-LaMa benchmark FAILED: \(error)")
        }

        isRunning = false
    }

    private func runRifeBenchmark(config: MLModelConfiguration) async throws {
        appendLog("\n--- RIFE (IFNet), \(RifeRunner.requiredSize)x\(RifeRunner.requiredSize) ---")

        guard let modelURL = Bundle.main.url(forResource: "RifeIFNet", withExtension: "mlmodelc") else {
            appendLog("RifeIFNet.mlmodelc not found in bundle -- skipping")
            return
        }
        let inspectedModel = try MLModel(contentsOf: modelURL, configuration: config)
        let planSummary = ComputePlanInspector.inspect(model: inspectedModel, configuration: config)
        appendLog("Compute config: \(planSummary.summary)")
        appendLog("NOTE: requested/available only -- per-op ANE placement not independently verified (see ComputePlanInspector.swift)")

        let runner = try RifeRunner(configuration: config)
        guard let frameA = TestFrameProvider.solidColorImage(size: RifeRunner.requiredSize, seed: 1),
              let frameB = TestFrameProvider.solidColorImage(size: RifeRunner.requiredSize, seed: 2) else {
            appendLog("Failed to generate test frames")
            return
        }

        var buildTimes: [Double] = []
        var runTimes: [Double] = []
        var postTimes: [Double] = []

        for i in 1...Self.repetitions {
            let (_, timing) = try runner.interpolateMidpoint(frameA: frameA, frameB: frameB)
            buildTimes.append(timing.buildMs)
            runTimes.append(timing.runMs)
            postTimes.append(timing.postprocessMs)
            appendLog("  rep \(i): build=\(String(format: "%.1f", timing.buildMs))ms run=\(String(format: "%.1f", timing.runMs))ms post=\(String(format: "%.1f", timing.postprocessMs))ms")
        }

        appendLog("  avg: build=\(String(format: "%.1f", buildTimes.average))ms run=\(String(format: "%.1f", runTimes.average))ms post=\(String(format: "%.1f", postTimes.average))ms")
    }

    private func runLamaBenchmark(config: MLModelConfiguration) async throws {
        appendLog("\n--- big-LaMa, \(LamaRunner.inputSize)x\(LamaRunner.inputSize) ---")

        guard let modelURL = Bundle.main.url(forResource: "BigLama", withExtension: "mlmodelc") else {
            appendLog("BigLama.mlmodelc not found in bundle -- skipping")
            return
        }
        let inspectedModel = try MLModel(contentsOf: modelURL, configuration: config)
        let planSummary = ComputePlanInspector.inspect(model: inspectedModel, configuration: config)
        appendLog("Compute config: \(planSummary.summary)")
        appendLog("NOTE: requested/available only -- per-op ANE placement not independently verified (see ComputePlanInspector.swift)")

        let runner = try LamaRunner(configuration: config)
        guard let image = TestFrameProvider.solidColorImage(size: LamaRunner.inputSize, seed: 3),
              let mask = TestFrameProvider.solidColorImage(size: LamaRunner.inputSize, seed: 4, grayscale: true) else {
            appendLog("Failed to generate test image/mask")
            return
        }

        var buildTimes: [Double] = []
        var runTimes: [Double] = []
        var postTimes: [Double] = []

        for i in 1...Self.repetitions {
            let (_, timing) = try runner.fill(image: image, mask: mask)
            buildTimes.append(timing.buildMs)
            runTimes.append(timing.runMs)
            postTimes.append(timing.postprocessMs)
            appendLog("  rep \(i): build=\(String(format: "%.1f", timing.buildMs))ms run=\(String(format: "%.1f", timing.runMs))ms post=\(String(format: "%.1f", timing.postprocessMs))ms")
        }

        appendLog("  avg: build=\(String(format: "%.1f", buildTimes.average))ms run=\(String(format: "%.1f", runTimes.average))ms post=\(String(format: "%.1f", postTimes.average))ms")
    }
}

private extension Array where Element == Double {
    var average: Double { isEmpty ? 0 : reduce(0, +) / Double(count) }
}
