import SwiftUI
import CoreML

/// Benchmark dashboard: pick/stage real test clips (ClipLibrary +
/// ClipPickerView), run the Tier2-mobile pipeline port, watch live
/// per-stage progress (PipelineTimelineView), inspect every stage's
/// output as a playable video (not just a static thumbnail), and compare
/// input vs. final result side-by-side. The raw diagnostic log from the
/// original harness is preserved in full, just demoted to a collapsible
/// panel instead of dominating the screen.
///
/// NOTE: this is a weaker check than true per-operation ANE attribution
/// (see ComputePlanInspector.swift's docstring for why -- MLComputePlan's
/// exact Swift API couldn't be pinned down without real Apple docs
/// access) -- report benchmark numbers with that caveat explicit, not as
/// confirmed ANE-only timing.
struct BenchmarkView: View {
    @StateObject private var clipLibrary = ClipLibrary()

    @State private var log: [String] = []
    @State private var isRunning = false
    @State private var previewImages: [(label: String, image: UIImage)] = []
    @State private var outputVideos: [(label: String, url: URL)] = []
    @State private var stages: [PipelineStage] = Self.initialStages
    @State private var isLogExpanded = false
    @State private var viewerClip: (title: String, url: URL)?
    @State private var showCompare = false
    @State private var lastRunSummary: String?

    static let repetitions = 5 // matches Table 8's 5-repetition methodology

    static let initialStages: [PipelineStage] = [
        PipelineStage(name: "Load & Align", icon: "align.horizontal.left"),
        PipelineStage(name: "Background Reconstruction", icon: "photo.on.rectangle"),
        PipelineStage(name: "Illumination Extraction", icon: "sun.max"),
        PipelineStage(name: "Final Compositing", icon: "square.stack.3d.up"),
        PipelineStage(name: "Export", icon: "square.and.arrow.down"),
    ]

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 16) {
                    ClipPickerView(library: clipLibrary)

                    runControlCard

                    if stages.contains(where: { $0.status != .pending }) {
                        PipelineTimelineView(stages: stages)
                    }

                    if !previewImages.isEmpty {
                        galleryCard
                    }

                    if outputVideos.count >= 2 {
                        compareCard
                    }

                    logCard
                }
                .padding(16)
            }
            .background(Theme.pageBackground)
            .navigationTitle("bodySITARA Bench")
        }
        .fullScreenCover(item: Binding(
            get: { viewerClip.map { IdentifiableURL(title: $0.title, url: $0.url) } },
            set: { if $0 == nil { viewerClip = nil } }
        )) { clip in
            VideoViewerSheet(title: clip.title, url: clip.url)
        }
        .fullScreenCover(isPresented: $showCompare) {
            if let before = outputVideos.first(where: { $0.label.contains("Masked") }) ?? outputVideos.first,
               let after = outputVideos.first(where: { $0.label.contains("Final") }) ?? outputVideos.last {
                CompareView(beforeTitle: before.label, beforeURL: before.url, afterTitle: after.label, afterURL: after.url)
            }
        }
    }

    // MARK: - Run control

    private var runControlCard: some View {
        Card {
            VStack(alignment: .leading, spacing: 12) {
                SectionHeader(icon: "play.circle", title: "Run")

                Button {
                    Task { await runBenchmark() }
                } label: {
                    HStack {
                        if isRunning {
                            ProgressView().controlSize(.small).tint(.white)
                        } else {
                            Image(systemName: "play.fill")
                        }
                        Text(isRunning ? "Running…" : "Run Benchmark")
                            .fontWeight(.semibold)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 12)
                }
                .buttonStyle(.borderedProminent)
                .disabled(isRunning)

                HStack(spacing: 12) {
                    Label("\(Self.repetitions) reps", systemImage: "arrow.triangle.2.circlepath")
                    Label(clipLibrary.activePreset?.name ?? "Bundled sample", systemImage: "film")
                }
                .font(.caption)
                .foregroundStyle(.secondary)

                if let lastRunSummary {
                    Divider()
                    Text(lastRunSummary)
                        .font(.system(.caption, design: .rounded))
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    // MARK: - Gallery

    private var galleryCard: some View {
        Card {
            VStack(alignment: .leading, spacing: 12) {
                SectionHeader(icon: "photo.stack", title: "Visual Check")
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(alignment: .top, spacing: 12) {
                        ForEach(Array(previewImages.enumerated()), id: \.offset) { _, item in
                            GalleryTile(label: item.label, image: item.image)
                        }
                    }
                }
                if !outputVideos.isEmpty {
                    Text("Exported videos")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 10) {
                            ForEach(outputVideos, id: \.label) { item in
                                Button {
                                    viewerClip = (item.label, item.url)
                                } label: {
                                    Label(item.label, systemImage: "play.circle.fill")
                                        .font(.caption)
                                }
                                .buttonStyle(.bordered)
                                .controlSize(.small)
                            }
                        }
                    }
                }
            }
        }
    }

    private var compareCard: some View {
        Card {
            VStack(alignment: .leading, spacing: 10) {
                SectionHeader(icon: "rectangle.split.2x1", title: "Compare")
                Text("Play the input clip and the final result side-by-side.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Button {
                    showCompare = true
                } label: {
                    Label("Open Compare View", systemImage: "play.rectangle.on.rectangle")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
            }
        }
    }

    // MARK: - Log

    private var logCard: some View {
        Card {
            VStack(alignment: .leading, spacing: 8) {
                Button {
                    withAnimation(.snappy) { isLogExpanded.toggle() }
                } label: {
                    HStack {
                        SectionHeader(icon: "terminal", title: "Diagnostic Log (\(log.count) lines)")
                        Image(systemName: isLogExpanded ? "chevron.up" : "chevron.down")
                            .foregroundStyle(.secondary)
                    }
                }
                .buttonStyle(.plain)

                if isLogExpanded {
                    ScrollView {
                        Text(log.joined(separator: "\n"))
                            .font(.system(.caption2, design: .monospaced))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                    }
                    .frame(maxHeight: 320)
                }
            }
        }
    }

    // MARK: - Stage helpers

    private func setStage(_ name: String, status: StageStatus, timings: [(String, Double)] = [], method: (String, Bool)? = nil, detail: [String] = []) {
        guard let idx = stages.firstIndex(where: { $0.name == name }) else { return }
        stages[idx].status = status
        if !timings.isEmpty { stages[idx].timings = timings }
        if let method { stages[idx].methodBadge = (method.0, method.1) }
        if !detail.isEmpty { stages[idx].detailLines.append(contentsOf: detail) }
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
        previewImages = []
        outputVideos = []
        lastRunSummary = nil
        stages = Self.initialStages
        appendLog("=== bodySITARA iOS Benchmark (iPhone 15 Pro Max target) ===")
        appendLog("Repetitions: \(Self.repetitions)")
        appendLog("Clip: \(clipLibrary.activePreset?.name ?? "bundled sample")")

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
            if let running = stages.first(where: { $0.status == .running })?.name {
                setStage(running, status: .failed(error.localizedDescription))
            }
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

        // Prefer a staged clip preset; fall back to the bundled sample
        // clip, matching the app's original behavior when nothing has
        // been staged yet.
        let maskedURL: URL
        let maskURL: URL
        if let active = clipLibrary.activeClip {
            maskedURL = active.masked
            maskURL = active.mask
        } else if let bundledMasked = Bundle.main.url(forResource: "masked_video", withExtension: "mp4"),
                  let bundledMask = Bundle.main.url(forResource: "mask", withExtension: "mp4") {
            maskedURL = bundledMasked
            maskURL = bundledMask
        } else {
            appendLog("No clip staged and no bundled sample found -- pick a clip in Test Clips above.")
            return
        }

        setStage("Load & Align", status: .running)
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
        let framesToConvert = Array(maskedVideo.frames[0..<n])
        let masksToConvert = Array(maskVideo.frames[0..<n])
        let (colorBuffers, maskBuffers) = await Task.detached(priority: .userInitiated) {
            (framesToConvert.map { RGBBuffer.from(cgImage: $0) }, masksToConvert.map { MaskBuffer.from(cgImage: $0) })
        }.value
        setStage("Load & Align", status: .done, timings: [("load", loadMs)], detail: ["\(n) frames @ \(maskedVideo.width)x\(maskedVideo.height)"])

        setStage("Background Reconstruction", status: .running)
        appendLog("[diag] running Background Reconstruction (align pyramid + trimmed-mean, on-device)...")
        // The alignment pyramid + per-pixel trimmed-mean aggregation is the
        // single most expensive step in this pipeline (multi-second on a
        // 1264x1264 clip) and BenchmarkView's methods are implicitly
        // @MainActor (it's a SwiftUI View) -- calling it directly here
        // would run all of that math on the main thread despite the outer
        // Task{}, freezing the UI for the full duration (observed on-device
        // 2026-07-31: the app appeared "stuck" during a ~13s run). Hop to a
        // detached background task for the actual compute; only the
        // appendLog/setStage/addPreview calls that follow need the main actor.
        let reconResult = await Task.detached(priority: .userInitiated) {
            BackgroundReconstructor.reconstruct(colorFrames: colorBuffers, masks: maskBuffers)
        }.value
        let corePctPreview = 100.0 * Double(reconResult.core.filter { $0 }.count) / Double(reconResult.core.count)
        appendLog("  align=\(String(format: "%.0f", reconResult.alignMs))ms trimmed-mean=\(String(format: "%.0f", reconResult.trimmedMeanMs))ms (neural/push-pull core: \(String(format: "%.1f", corePctPreview))%)")
        addPreview("Plate BEFORE fill\n(trimmed-mean)", reconResult.plateBeforeLama.toCGImage())
        addPreview("Core\n(white=needs fill)", Self.maskPreviewImage(reconResult.core, width: reconResult.plateBeforeLama.width, height: reconResult.plateBeforeLama.height))

        appendLog("[diag] running core-fill (bbox-cropped LaMa, or push-pull if core > 35% of frame)...")
        let lamaRunner = try LamaRunner(configuration: config)
        let plateForFill = reconResult.plateBeforeLama
        let coreForFill = reconResult.core
        let coreFillResult: LamaRunner.CoreFillResult = try await Task.detached(priority: .userInitiated) {
            try autoreleasepool {
                try lamaRunner.fillCore(plate: plateForFill, core: coreForFill)
            }
        }.value
        let backgroundFinal = coreFillResult.filled
        let lamaTiming = coreFillResult.timing
        let lamaStageMs = (lamaTiming?.buildMs ?? 0) + (lamaTiming?.runMs ?? 0) + (lamaTiming?.postprocessMs ?? 0)
        let bgReconTotalMs = reconResult.alignMs + reconResult.trimmedMeanMs + lamaStageMs
        let usedPushPull = coreFillResult.method.hasPrefix("push-pull")
        if let t = lamaTiming {
            appendLog("  core-fill (\(coreFillResult.method)): build=\(String(format: "%.1f", t.buildMs))ms run=\(String(format: "%.1f", t.runMs))ms post=\(String(format: "%.1f", t.postprocessMs))ms")
        } else {
            appendLog("  core-fill: \(coreFillResult.method) (no LaMa call)")
        }
        appendLog("  Background Reconstruction TOTAL: \(String(format: "%.0f", bgReconTotalMs))ms (\(n) frames, \(maskedVideo.width)x\(maskedVideo.height))")
        addPreview("Background FINAL\n(after core-fill)", backgroundFinal.toCGImage())
        setStage("Background Reconstruction", status: .done,
                  timings: [("align", reconResult.alignMs), ("trim", reconResult.trimmedMeanMs), ("fill", lamaStageMs)],
                  method: (usedPushPull ? "push-pull ⚠" : "lama-crop", usedPushPull),
                  detail: [coreFillResult.method, "core: \(String(format: "%.1f", coreFillResult.corePct))% of frame"])

        // Real per-frame background VIDEO: for EACH frame i, the hole that
        // needs filling is THAT FRAME's own mask (maskBuffers[i]), not
        // frame 0's -- the person moves between frames, so the hole is a
        // different shape/position every frame. Using frame 0's mask for
        // all frames (an earlier version of this code did exactly that)
        // is wrong two ways at once: it pastes a fake static patch onto
        // real background the person has already moved away from, AND it
        // fails to cover the person's ACTUAL current position in frames
        // where they've moved elsewhere. Fixed per explicit correction
        // (2026-07-27): each frame samples its OWN mask region from the
        // aggregated backgroundFinal plate (which represents "what's
        // really behind wherever the person was, aggregated across the
        // whole clip"), everywhere else uses that frame's real pixels.
        let reconstructedBgFrames: [CGImage] = await Task.detached(priority: .userInitiated) {
            var frames: [CGImage] = []
            for i in 0..<n {
                var outR = colorBuffers[i].r, outG = colorBuffers[i].g, outB = colorBuffers[i].b
                let frameMask = maskBuffers[i].isPerson
                for p in 0..<(backgroundFinal.width * backgroundFinal.height) where frameMask[p] {
                    outR[p] = backgroundFinal.r[p]
                    outG[p] = backgroundFinal.g[p]
                    outB[p] = backgroundFinal.b[p]
                }
                let frameBuf = RGBBuffer(r: outR, g: outG, b: outB, width: backgroundFinal.width, height: backgroundFinal.height)
                if let cg = frameBuf.toCGImage() { frames.append(cg) }
            }
            return frames
        }.value
        appendLog("  reconstructed-background video: \(reconstructedBgFrames.count) frames, each using its OWN mask for the fill region")

        setStage("Illumination Extraction", status: .running)
        appendLog("[diag] running Illumination Extraction (lightmap)...")
        let lightmapResult = LightmapExtractor.extract(from: backgroundFinal)
        appendLog("  Illumination Extraction: \(String(format: "%.1f", lightmapResult.totalMs))ms")
        addPreview("Lightmap", lightmapResult.lightmap.toCGImage())
        setStage("Illumination Extraction", status: .done, timings: [("total", lightmapResult.totalMs)])

        // Step 4: composite the ORIGINAL grey silhouette (per-frame, real
        // clip content) onto the lightmap -- this is the actual
        // outbound-to-server signal per the confirmed pipeline (2026-07-27):
        // silhouette-over-lightmap is what gets sent, not the raw silhouette
        // alone. Uses each frame's own person mask as alpha (person region
        // opaque, background transparent -- only the silhouette shape
        // itself needs to reach the server, not the reconstructed
        // background around it).
        appendLog("[diag] compositing original silhouette onto lightmap (outbound-to-server signal, all \(n) frames)...")
        let lightmapForComposite = lightmapResult.lightmap
        let silhouetteOnLightmapFrames: [CGImage] = await Task.detached(priority: .userInitiated) {
            var frames: [CGImage] = []
            for i in 0..<n {
                let alpha = maskBuffers[i].isPerson.map { $0 ? Float(1) : Float(0) }
                let result = Compositor.compositeOnly(background: lightmapForComposite, character: colorBuffers[i], alpha: alpha)
                if let cg = result.composited.toCGImage() { frames.append(cg) }
            }
            return frames
        }.value
        appendLog("  silhouette-on-lightmap: \(silhouetteOnLightmapFrames.count) frames composited")
        if let mid = silhouetteOnLightmapFrames[safe: n / 2] {
            addPreview("Silhouette-on-Lightmap\n(TO SERVER)", mid)
        }

        // Steps 5-6: the server call (WanAnimate) can't be made from this
        // local benchmark -- per explicit scope decision, a PLACEHOLDER
        // avatar stands in for the real synthetic-avatar-over-lightmap
        // response. Step 6 composites that returned avatar onto the
        // RECONSTRUCTED background (not the lightmap) to produce the
        // final video.
        setStage("Final Compositing", status: .running)
        appendLog("[diag] simulating server response (PLACEHOLDER avatar, no real WanAnimate call)...")
        let (placeholderChar, placeholderAlpha) = Compositor.placeholderCharacter(width: backgroundFinal.width, height: backgroundFinal.height)

        appendLog("[diag] running Final Compositing: placeholder avatar onto per-frame reconstructed background (relight EXCLUDED per scope decision)...")
        let (finalFrames, totalCompositeMs): ([CGImage], Double) = await Task.detached(priority: .userInitiated) {
            var frames: [CGImage] = []
            var totalMs = 0.0
            for i in 0..<n {
                let frameBg = RGBBuffer.from(cgImage: reconstructedBgFrames[i])
                let result = Compositor.compositeOnly(background: frameBg, character: placeholderChar, alpha: placeholderAlpha)
                totalMs += result.compositeMs
                if let cg = result.composited.toCGImage() { frames.append(cg) }
            }
            return (frames, totalMs)
        }.value
        appendLog("  Final Compositing: \(finalFrames.count) frames, composite total=\(String(format: "%.1f", totalCompositeMs))ms")
        if let mid = finalFrames[safe: n / 2] {
            addPreview("FINAL\n(avatar on reconstructed bg, no relight)", mid)
        }
        setStage("Final Compositing", status: .done, timings: [("composite", totalCompositeMs)])

        // Video export: save the real pipeline stages as .mp4 files so
        // they can be viewed/scrubbed on-device via Photos, not just
        // inspected as single still frames. Also kept in-memory (via
        // outputVideos) so the Gallery/Compare cards can play them
        // in-app immediately, without a round-trip through Photos.
        setStage("Export", status: .running)
        appendLog("\n[diag] encoding output videos (\(n) real frames each, not repeated stills)...")
        do {
            let tmpDir = FileManager.default.temporaryDirectory
            let fps: Int32 = 10 // matches the bundled clip's ~10fps sampling

            outputVideos.append((label: "Masked Input", url: maskedURL))

            if !reconstructedBgFrames.isEmpty {
                let bgURL = tmpDir.appendingPathComponent("reconstructed_background.mp4")
                try await Task.detached(priority: .userInitiated) {
                    try VideoEncoder.encode(frames: reconstructedBgFrames, fps: fps, outputURL: bgURL)
                }.value
                outputVideos.append((label: "Reconstructed Background", url: bgURL))
                try await VideoEncoder.saveToPhotoLibrary(url: bgURL)
                appendLog("  Saved reconstructed_background.mp4 to Photos (\(reconstructedBgFrames.count) frames)")
            }

            if !silhouetteOnLightmapFrames.isEmpty {
                let silURL = tmpDir.appendingPathComponent("silhouette_on_lightmap.mp4")
                try await Task.detached(priority: .userInitiated) {
                    try VideoEncoder.encode(frames: silhouetteOnLightmapFrames, fps: fps, outputURL: silURL)
                }.value
                outputVideos.append((label: "Silhouette on Lightmap", url: silURL))
                try await VideoEncoder.saveToPhotoLibrary(url: silURL)
                appendLog("  Saved silhouette_on_lightmap.mp4 to Photos (\(silhouetteOnLightmapFrames.count) frames)")
            }

            if !finalFrames.isEmpty {
                let finalURL = tmpDir.appendingPathComponent("final_output.mp4")
                try await Task.detached(priority: .userInitiated) {
                    try VideoEncoder.encode(frames: finalFrames, fps: fps, outputURL: finalURL)
                }.value
                outputVideos.append((label: "Final Output", url: finalURL))
                try await VideoEncoder.saveToPhotoLibrary(url: finalURL)
                appendLog("  Saved final_output.mp4 to Photos (\(finalFrames.count) frames)")
                appendLog("  NOTE: avatar itself is a static PLACEHOLDER cutout (no per-frame motion) -- background behind it is real per-frame video; a real WanAnimate avatar would also move per-frame.")
            }
            setStage("Export", status: .done, detail: outputVideos.map { $0.label })
        } catch {
            appendLog("  Video export/save FAILED: \(error)")
            setStage("Export", status: .failed(error.localizedDescription))
        }

        appendLog("\n  SUMMARY (RIFE + relight excluded, per scope decision):")
        appendLog("    Background Reconstruction: \(String(format: "%.0f", bgReconTotalMs))ms")
        appendLog("    Illumination Extraction:   \(String(format: "%.1f", lightmapResult.totalMs))ms")
        appendLog("    Final Compositing:         \(String(format: "%.1f", totalCompositeMs))ms")
        let totalMs = bgReconTotalMs + lightmapResult.totalMs + totalCompositeMs
        appendLog("    TOTAL (3 stages):          \(String(format: "%.0f", totalMs))ms for \(n) src frames")
        appendLog("  NOTE: Final Compositing uses a PLACEHOLDER character cutout, not a real WanAnimate render -- tests compositing MATH cost only, not visual fidelity.")
        appendLog("  Scroll the image strip above to visually verify each stage's output before trusting these numbers.")

        lastRunSummary = "Total \(String(format: "%.0f", totalMs))ms for \(n) frames · core-fill: \(coreFillResult.method)"
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

private struct IdentifiableURL: Identifiable {
    let title: String
    let url: URL
    var id: String { url.path }
}

private struct GalleryTile: View {
    let label: String
    let image: UIImage
    @State private var isZoomed = false

    var body: some View {
        VStack(spacing: 6) {
            Image(uiImage: image)
                .resizable()
                .scaledToFill()
                .frame(width: 150, height: 150)
                .clipShape(RoundedRectangle(cornerRadius: Theme.smallCornerRadius, style: .continuous))
                .onTapGesture { isZoomed = true }
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .lineLimit(2)
        }
        .frame(width: 150)
        .sheet(isPresented: $isZoomed) {
            ZoomedImageView(label: label, image: image)
        }
    }
}

private struct ZoomedImageView: View {
    let label: String
    let image: UIImage
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationView {
            ScrollView([.horizontal, .vertical]) {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: .infinity)
            }
            .navigationTitle(label.replacingOccurrences(of: "\n", with: " "))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

private extension Array where Element == Double {
    var average: Double { isEmpty ? 0 : reduce(0, +) / Double(count) }
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
