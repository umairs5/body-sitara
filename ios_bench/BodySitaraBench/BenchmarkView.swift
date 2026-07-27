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
    @State private var previewImages: [(label: String, image: UIImage)] = []
    static let repetitions = 5 // matches Table 8's 5-repetition methodology

    var body: some View {
        NavigationView {
            VStack {
                Button(isRunning ? "Running..." : "Run Benchmark") {
                    Task { await runBenchmark() }
                }
                .disabled(isRunning)
                .padding()

                // Visual sanity-check gallery: input frame, reconstructed
                // background (before/after LaMa), lightmap, and final
                // composite -- added so the pipeline's OUTPUT can be
                // visually confirmed correct on-device, not just its
                // timing. A fast benchmark number is meaningless if the
                // stage it's timing produced garbage.
                if !previewImages.isEmpty {
                    ScrollView(.horizontal) {
                        HStack(alignment: .top) {
                            ForEach(previewImages, id: \.label) { item in
                                VStack {
                                    Text(item.label)
                                        .font(.caption)
                                    Image(uiImage: item.image)
                                        .resizable()
                                        .scaledToFit()
                                        .frame(width: 160, height: 160)
                                        .border(Color.gray)
                                }
                                .padding(.horizontal, 4)
                            }
                        }
                        .padding()
                    }
                    .frame(height: 220)
                }

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

    private func addPreview(_ label: String, _ cgImage: CGImage?) {
        guard let cgImage else { return }
        previewImages.append((label, UIImage(cgImage: cgImage)))
    }

    private func runBenchmark() async {
        isRunning = true
        log = []
        appendLog("=== bodySITARA iOS Benchmark (iPhone 15 Pro Max target) ===")
        appendLog("Repetitions: \(Self.repetitions)")

        let config = MLModelConfiguration()
        config.computeUnits = .all // let CoreML pick ANE/GPU/CPU; verified below, not assumed

        // ONLY the Tier2-Mobile System Cost stage runs here (2026-07-27):
        // RIFE is excluded per the project's RIFE scoping decision (see
        // memory: project_rife_decision_off.md) -- it's out of the
        // pipeline entirely, not just out of this specific report, so it
        // has no reason to run in this benchmark either. The standalone
        // LaMa-on-synthetic-frames benchmark (runLamaBenchmark, still
        // defined below for reference/reuse) is also dropped from this
        // run -- it's now redundant, since Background Reconstruction
        // below already calls LaMa for real, on real clip data, as part
        // of its own core-fill step.
        do {
            try await runTier2MobileSystemCostBenchmark(config: config)
        } catch {
            appendLog("Tier2-mobile system cost benchmark FAILED: \(error)")
        }

        isRunning = false
    }

    /// Native (STATIC/JITTER-path only) port of the three Tier2-mobile
    /// local pipeline stages, RIFE deliberately EXCLUDED per explicit
    /// scope decision (see project_rife_decision_off.md): Background
    /// Reconstruction (alignment pyramid + trimmed-mean + LaMa core-fill),
    /// Illumination Extraction (lightmap), and Final Compositing
    /// (alpha-blend PLACEHOLDER character + relight -- no real WanAnimate
    /// render used here, per scope decision). Runs fully on-device
    /// (alignment math included, not precomputed) so latency is a real,
    /// honest end-to-end number comparable to Android's per-stage
    /// measurements at the same granularity.
    private func runTier2MobileSystemCostBenchmark(config: MLModelConfiguration) async throws {
        appendLog("\n--- Tier2-Mobile System Cost (Bg Recon + Illum Extraction + Compositing, RIFE excluded) ---")
        previewImages = []

        guard let maskedURL = Bundle.main.url(forResource: "masked_video", withExtension: "mp4"),
              let maskURL = Bundle.main.url(forResource: "mask", withExtension: "mp4") else {
            appendLog("masked_video.mp4 / mask.mp4 not found in bundle -- skipping (see ios_bench/TestAssets/README)")
            return
        }
        appendLog("[diag] loading real clip frames...")
        let tLoadStart = CFAbsoluteTimeGetCurrent()
        let maskedVideo = try VideoFrameLoader.loadFrames(url: maskedURL)
        let maskVideo = try VideoFrameLoader.loadFrames(url: maskURL)
        let loadMs = (CFAbsoluteTimeGetCurrent() - tLoadStart) * 1000
        let n = min(maskedVideo.frames.count, maskVideo.frames.count)
        appendLog("[diag] loaded \(n) frames (\(maskedVideo.width)x\(maskedVideo.height)) in \(String(format: "%.0f", loadMs))ms")

        addPreview("Input frame 0\n(masked_video)", maskedVideo.frames[0])
        addPreview("Input mask 0\n(mask.mp4)", maskVideo.frames[0])
        addPreview("Input frame \(n/2)\n(mid-clip)", maskedVideo.frames[n / 2])

        appendLog("[diag] converting frames to RGB/mask buffers...")
        let colorBuffers = maskedVideo.frames[0..<n].map { RGBBuffer.from(cgImage: $0) }
        let maskBuffers = maskVideo.frames[0..<n].map { MaskBuffer.from(cgImage: $0) }

        appendLog("[diag] running Background Reconstruction (align pyramid + trimmed-mean, on-device)...")
        let reconResult = BackgroundReconstructor.reconstruct(colorFrames: colorBuffers, masks: maskBuffers)
        let neverRevealedPct = 100.0 * Double(reconResult.neverRevealed.filter { $0 }.count) / Double(reconResult.neverRevealed.count)
        appendLog("  align=\(String(format: "%.0f", reconResult.alignMs))ms trimmed-mean=\(String(format: "%.0f", reconResult.trimmedMeanMs))ms (never-revealed core: \(String(format: "%.1f", neverRevealedPct))%)")
        addPreview("Plate BEFORE LaMa\n(trimmed-mean)", reconResult.plateBeforeLama.toCGImage())
        addPreview("Never-revealed core\n(white=needs LaMa)", Self.maskPreviewImage(reconResult.neverRevealed, width: reconResult.plateBeforeLama.width, height: reconResult.plateBeforeLama.height))

        appendLog("[diag] running LaMa core-fill (once per clip, on never-revealed core only)...")
        let lamaRunner = try LamaRunner(configuration: config)
        var lamaTiming: LamaRunner.StageTiming!
        var backgroundFinal: RGBBuffer!
        try autoreleasepool {
            let (filled, timing) = try lamaRunner.fillPixels(plate: reconResult.plateBeforeLama, neverRevealed: reconResult.neverRevealed)
            backgroundFinal = filled
            lamaTiming = timing
        }
        let bgReconTotalMs = reconResult.alignMs + reconResult.trimmedMeanMs + lamaTiming.buildMs + lamaTiming.runMs + lamaTiming.postprocessMs
        appendLog("  LaMa: build=\(String(format: "%.1f", lamaTiming.buildMs))ms run=\(String(format: "%.1f", lamaTiming.runMs))ms post=\(String(format: "%.1f", lamaTiming.postprocessMs))ms")
        appendLog("  Background Reconstruction TOTAL: \(String(format: "%.0f", bgReconTotalMs))ms (\(n) frames, \(maskedVideo.width)x\(maskedVideo.height))")
        addPreview("Background FINAL\n(after LaMa)", backgroundFinal.toCGImage())

        appendLog("[diag] running Illumination Extraction (lightmap)...")
        let lightmapResult = LightmapExtractor.extract(from: backgroundFinal)
        appendLog("  Illumination Extraction: \(String(format: "%.1f", lightmapResult.totalMs))ms")
        addPreview("Lightmap", lightmapResult.lightmap.toCGImage())

        appendLog("[diag] running Final Compositing (placeholder character, relight EXCLUDED per scope decision)...")
        let (placeholderChar, placeholderAlpha) = Compositor.placeholderCharacter(width: backgroundFinal.width, height: backgroundFinal.height)
        let compositeOnlyResult = Compositor.compositeOnly(background: backgroundFinal, character: placeholderChar, alpha: placeholderAlpha)
        appendLog("  Final Compositing: composite=\(String(format: "%.1f", compositeOnlyResult.compositeMs))ms")
        addPreview("Composited\n(FINAL, no relight)", compositeOnlyResult.composited.toCGImage())

        appendLog("\n  SUMMARY (RIFE + relight excluded, per scope decision):")
        appendLog("    Background Reconstruction: \(String(format: "%.0f", bgReconTotalMs))ms")
        appendLog("    Illumination Extraction:   \(String(format: "%.1f", lightmapResult.totalMs))ms")
        appendLog("    Final Compositing:         \(String(format: "%.1f", compositeOnlyResult.compositeMs))ms")
        let totalMs = bgReconTotalMs + lightmapResult.totalMs + compositeOnlyResult.compositeMs
        appendLog("    TOTAL (3 stages):          \(String(format: "%.0f", totalMs))ms for \(n) src frames")
        appendLog("  NOTE: Final Compositing uses a PLACEHOLDER character cutout, not a real WanAnimate render -- tests compositing MATH cost only, not visual fidelity.")
        appendLog("  Scroll the image strip above to visually verify each stage's output before trusting these numbers.")
    }

    private static func maskPreviewImage(_ mask: [Bool], width: Int, height: Int) -> CGImage? {
        var raw = [UInt8](repeating: 0, count: width * height)
        for i in 0..<(width * height) { raw[i] = mask[i] ? 255 : 0 }
        let colorSpace = CGColorSpaceCreateDeviceGray()
        guard let provider = CGDataProvider(data: Data(raw) as CFData) else { return nil }
        return CGImage(
            width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 8,
            bytesPerRow: width, space: colorSpace,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
        )
    }

    private func runRifeBenchmark(config: MLModelConfiguration) async throws {
        appendLog("\n--- RIFE (IFNet), \(RifeRunner.requiredSize)x\(RifeRunner.requiredSize) ---")

        guard let modelURL = Bundle.main.url(forResource: "RifeIFNet", withExtension: "mlmodelc") else {
            appendLog("RifeIFNet.mlmodelc not found in bundle -- skipping")
            return
        }
        appendLog("[diag] found RifeIFNet.mlmodelc at \(modelURL.path)")

        appendLog("[diag] about to MLModel(contentsOf:configuration:)...")
        let inspectedModel = try MLModel(contentsOf: modelURL, configuration: config)
        appendLog("[diag] MLModel load succeeded")

        appendLog("[diag] about to ComputePlanInspector.inspect...")
        let planSummary = ComputePlanInspector.inspect(model: inspectedModel, configuration: config)
        appendLog("Compute config: \(planSummary.summary)")
        appendLog("NOTE: requested/available only -- per-op ANE placement not independently verified (see ComputePlanInspector.swift)")

        appendLog("[diag] about to RifeRunner(configuration:)...")
        let runner = try RifeRunner(configuration: config)
        appendLog("[diag] RifeRunner init succeeded")

        guard let frameA = TestFrameProvider.solidColorImage(size: RifeRunner.requiredSize, seed: 1),
              let frameB = TestFrameProvider.solidColorImage(size: RifeRunner.requiredSize, seed: 2) else {
            appendLog("Failed to generate test frames")
            return
        }
        appendLog("[diag] test frames generated")

        var buildTimes: [Double] = []
        var runTimes: [Double] = []
        var postTimes: [Double] = []

        for i in 1...Self.repetitions {
            appendLog("[diag] rep \(i): about to interpolateMidpoint...")
            // Same autoreleasepool fix as LamaRunner's loop -- RIFE's
            // per-rep tensors are even larger (two 1280x1280x3 inputs +
            // output), same risk of ARC not reclaiming CoreML buffers
            // fast enough between back-to-back large inferences.
            var timing: RifeRunner.StageTiming!
            try autoreleasepool {
                let result = try runner.interpolateMidpoint(frameA: frameA, frameB: frameB)
                timing = result.1
            }
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
        appendLog("[diag] found BigLama.mlmodelc at \(modelURL.path)")

        appendLog("[diag] about to MLModel(contentsOf:configuration:)...")
        let inspectedModel = try MLModel(contentsOf: modelURL, configuration: config)
        appendLog("[diag] MLModel load succeeded")

        appendLog("[diag] about to ComputePlanInspector.inspect...")
        let planSummary = ComputePlanInspector.inspect(model: inspectedModel, configuration: config)
        appendLog("Compute config: \(planSummary.summary)")
        appendLog("NOTE: requested/available only -- per-op ANE placement not independently verified (see ComputePlanInspector.swift)")

        appendLog("[diag] about to LamaRunner(configuration:)...")
        let runner = try LamaRunner(configuration: config)
        appendLog("[diag] LamaRunner init succeeded")

        guard let image = TestFrameProvider.solidColorImage(size: LamaRunner.inputSize, seed: 3),
              let mask = TestFrameProvider.solidColorImage(size: LamaRunner.inputSize, seed: 4, grayscale: true) else {
            appendLog("Failed to generate test image/mask")
            return
        }
        appendLog("[diag] test image/mask generated")

        var buildTimes: [Double] = []
        var runTimes: [Double] = []
        var postTimes: [Double] = []

        for i in 1...Self.repetitions {
            appendLog("[diag] rep \(i): about to fill...")
            // Real on-device test (iPhone 15 Pro Max) crashed hard on rep 2
            // (rep 1 succeeded: build=3.7ms run=1473.3ms post=8.1ms) --
            // large 1280x1280 CoreML input/output MLMultiArrays are
            // Objective-C-backed and can outlive their expected scope
            // until the next autorelease-pool drain, so back-to-back large
            // inferences can pile up memory faster than ARC alone
            // reclaims it. autoreleasepool forces a drain after every
            // single repetition -- the standard, well-documented fix for
            // this exact "works once, crashes on repeat" CoreML pattern.
            var timing: LamaRunner.StageTiming!
            try autoreleasepool {
                let result = try runner.fill(image: image, mask: mask)
                timing = result.1
            }
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
