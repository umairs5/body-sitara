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
        // Stream-decode straight to the compact per-frame types
        // (RGBBuffer8/PackedMaskFrame), converting each CGImage the moment
        // it's decoded instead of first materializing a [CGImage] for the
        // whole clip. This is a THIRD, previously-unaddressed memory sink
        // upstream of the two already-fixed ones (BackgroundReconstructor's
        // internal aligned-frame stack, and colorBuffers/maskBuffers below
        // being UInt8/bit-packed instead of Float32/[Bool]): the earlier
        // `VideoFrameLoader.loadFrames` returned ALL decoded CGImages for a
        // clip at once, so loading BOTH masked_video and mask.mp4 held
        // ~3.9GB of CGImage backing stores simultaneously on a real
        // 300-frame/1280x1280 clip -- enough to jetsam-kill on its own,
        // matching the observed "crash during Load & Align, before
        // Background Reconstruction" symptom (2026-07-31). See
        // VideoFrameLoader.swift's header doc for the full before/after
        // memory math. Only index 0 is captured as a CGImage during this
        // pass (for the "Input frame 0"/"Input mask 0" previews below) --
        // the mid-clip preview needs index n/2, which isn't known until
        // BOTH videos are loaded (n depends on both counts), so it's
        // reconstructed from the already-decoded colorBuffers afterward
        // (single O(1) frame, not a re-decode).
        let maskedVideo = try VideoFrameLoader.loadFramesAsRGBBuffer8(url: maskedURL, previewIndices: [0])
        let maskVideo = try VideoFrameLoader.loadFramesAsPackedMask(url: maskURL, previewIndices: [0])
        let loadMs = (CFAbsoluteTimeGetCurrent() - tLoadStart) * 1000
        let n = min(maskedVideo.frames.count, maskVideo.frames.count)
        // Hoisted up here (previously declared much later, right before the
        // export `do` block) because the three streaming video writers now
        // open BEFORE their respective per-frame loops run, not after --
        // they all need `fps` at construction time, and this is the
        // earliest point `n`/dimensions are known for all of them.
        let fps: Int32 = 10 // matches the bundled clip's ~10fps sampling
        appendLog("[diag] loaded \(n) frames (\(maskedVideo.width)x\(maskedVideo.height)) in \(String(format: "%.0f", loadMs))ms")

        // colorBuffers/maskBuffers hold EVERY frame of the clip
        // simultaneously for the rest of this function's duration (used at
        // multiple later pipeline stages, not just as reconstruct()'s
        // input) -- so they MUST be the compact UInt8/bit-packed types
        // (RGBBuffer8/PackedMaskFrame), never the Float32 RGBBuffer/[Bool]
        // MaskBuffer. This was a previously-fixed root cause of a real
        // on-device jetsam kill on a 300-frame/1280x1280 clip: an earlier
        // fix narrowed BackgroundReconstructor's INTERNAL aligned-frame
        // copy to UInt8, but this array -- the thing that fix's input
        // actually pointed at -- was still N full Float32 RGBBuffers. See
        // PixelBuffer.swift's RGBBuffer8/PackedMaskFrame doc comments for
        // the full before/after memory math (~5.9GB -> ~1.47GB for
        // colorBuffers alone at 300 frames/1280x1280). The loader above now
        // produces these arrays directly (no [CGImage] -> map{} step), so
        // trim to the shared frame count n.
        let colorBuffers = n == maskedVideo.frames.count ? maskedVideo.frames : Array(maskedVideo.frames[0..<n])
        let maskBuffers = n == maskVideo.frames.count ? maskVideo.frames : Array(maskVideo.frames[0..<n])

        addPreview("Input frame 0\n(masked_video)", maskedVideo.previews[0])
        addPreview("Input mask 0\n(mask.mp4)", maskVideo.previews[0])
        addPreview("Input frame \(n/2)\n(mid-clip)", colorBuffers[n / 2].toFloatRGBBuffer().toCGImage())
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
        appendLog("  self-decided method: \(reconResult.method.rawValue) -- \(reconResult.methodDetail)")
        appendLog("  align=\(String(format: "%.0f", reconResult.alignMs))ms trimmed-mean=\(String(format: "%.0f", reconResult.trimmedMeanMs))ms (neural/push-pull core: \(String(format: "%.1f", corePctPreview))%)")
        addPreview("Plate BEFORE fill\n(trimmed-mean)", reconResult.plateBeforeLama.toCGImage())
        addPreview("Core\n(white=needs fill)", Self.maskPreviewImage(reconResult.core, width: reconResult.plateBeforeLama.width, height: reconResult.plateBeforeLama.height))

        let lamaRunner = try LamaRunner(configuration: config)
        let backgroundFinal: RGBBuffer
        let lamaStageMs: Double
        let usedPushPull: Bool
        let coreFillMethodSummary: String
        let coreFillCorePct: Double
        // renderReconFrame(i): "what does frame i's reconstructed background
        // look like" -- the SAME per-frame computation each branch below
        // already needed for its bgWriter loop, now also reused by Final
        // Compositing (Loop 3, further down) instead of Loop 3 reading back
        // a stored [CGImage] array. This is option (a) from the task's two
        // alternatives for Loop 3's dependency on Loop 1's output: recompute
        // frame i's reconstructed-background content from the same source
        // data (backgroundFinal/colorBuffers[i]/maskBuffers[i], or
        // windows for the DYNAMIC branch) rather than (b) decoding it back
        // from the just-written .mp4. (a) was chosen because every input
        // renderReconFrame needs (backgroundFinal, colorBuffers, maskBuffers,
        // windows) is ALREADY cheaply resident for the rest of this
        // function's duration regardless -- colorBuffers/maskBuffers are
        // the function-wide compact arrays, backgroundFinal/windows are O(1)
        // plates -- so recomputing costs one extra pass of the same cheap
        // per-pixel paste/blend math (milliseconds, not the LaMa/alignment
        // cost), with no new decode, no new file I/O, and byte-for-byte the
        // same pixels Loop 1 wrote (same function, same inputs, called
        // twice instead of read back once). Option (b) would have added a
        // full video decode (AVAssetReader + CIContext, the same streaming
        // pass VideoFrameLoader already does) per Final Compositing run
        // purely to undo work already done in memory this same function
        // call -- clearly the worse trade here.
        let renderReconFrame: (Int) -> RGBBuffer

        // Reconstructed-background frames are no longer accumulated into a
        // [CGImage] array (that was ~1.97GB resident for a 300-frame
        // 1280x1280 clip, plus it stayed alive afterward because Loop 3 --
        // Final Compositing below -- used to read it back via
        // reconstructedBgFrames[i]). Instead each frame is written straight
        // to reconstructed_background.mp4 as it's computed, via this
        // streaming writer opened BEFORE either branch's loop runs.
        // bgReconFrameCount tracks how many frames were actually written
        // (replaces the old `!reconstructedBgFrames.isEmpty` check at
        // export time -- both branches iterate 0..<n, so this is always n
        // once the writer finishes, but tracking it explicitly keeps the
        // export gate honest if a future branch could legitimately write 0
        // frames). One frame (index n/2) is additionally kept as a real
        // CGImage for the mid-clip gallery preview -- see reconMidPreview
        // below -- since the gallery needs an actual UIImage-backed frame,
        // not a video file, for that one specific stage snapshot.
        let tmpDir = FileManager.default.temporaryDirectory
        let bgURL = tmpDir.appendingPathComponent("reconstructed_background.mp4")
        let bgWriter = try StreamingVideoWriter(outputURL: bgURL, width: maskedVideo.width, height: maskedVideo.height, fps: fps)
        var bgReconFrameCount = 0
        var reconMidPreview: CGImage?

        switch reconResult.method {
        case .staticJitter:
            // STATIC/JITTER: a single global plate, core-filled ONCE per
            // clip, then pasted into each frame's own mask region -- the
            // pre-existing behavior, unchanged.
            appendLog("[diag] running core-fill (bbox-cropped LaMa, or push-pull if core > 35% of frame)...")
            let plateForFill = reconResult.plateBeforeLama
            let coreForFill = reconResult.core
            let coreFillResult: LamaRunner.CoreFillResult = try await Task.detached(priority: .userInitiated) {
                try autoreleasepool {
                    try lamaRunner.fillCore(plate: plateForFill, core: coreForFill)
                }
            }.value
            backgroundFinal = coreFillResult.filled
            let lamaTiming = coreFillResult.timing
            lamaStageMs = (lamaTiming?.buildMs ?? 0) + (lamaTiming?.runMs ?? 0) + (lamaTiming?.postprocessMs ?? 0)
            usedPushPull = coreFillResult.method.hasPrefix("push-pull")
            coreFillMethodSummary = coreFillResult.method
            coreFillCorePct = coreFillResult.corePct
            if let t = lamaTiming {
                appendLog("  core-fill (\(coreFillResult.method)): build=\(String(format: "%.1f", t.buildMs))ms run=\(String(format: "%.1f", t.runMs))ms post=\(String(format: "%.1f", t.postprocessMs))ms")
            } else {
                appendLog("  core-fill: \(coreFillResult.method) (no LaMa call)")
            }

            // Real per-frame background VIDEO: for EACH frame i, the hole
            // that needs filling is THAT FRAME's own mask (maskBuffers[i]),
            // not frame 0's -- the person moves between frames, so the
            // hole is a different shape/position every frame. Using frame
            // 0's mask for all frames (an earlier version of this code did
            // exactly that) is wrong two ways at once: it pastes a fake
            // static patch onto real background the person has already
            // moved away from, AND it fails to cover the person's ACTUAL
            // current position in frames where they've moved elsewhere.
            // Fixed per explicit correction (2026-07-27): each frame
            // samples its OWN mask region from the aggregated
            // backgroundFinal plate (which represents "what's really
            // behind wherever the person was, aggregated across the whole
            // clip"), everywhere else uses that frame's real pixels.
            // renderReconFrame for this branch: promote frame i's UInt8
            // color to Float32 and paste backgroundFinal into its mask
            // region -- exactly the per-frame computation this branch
            // always did, just factored out so Loop 3 (Final Compositing)
            // can call it again later instead of reading back a stored
            // array. `backgroundFinal` is captured by reference to the
            // already-assigned `let` above (Swift allows this since it's
            // definitely-initialized by this point in the branch).
            renderReconFrame = { i in
                var outR = colorBuffers[i].r.map { Float($0) }
                var outG = colorBuffers[i].g.map { Float($0) }
                var outB = colorBuffers[i].b.map { Float($0) }
                let frameMask = maskBuffers[i].unpacked().isPerson
                for p in 0..<(backgroundFinal.width * backgroundFinal.height) where frameMask[p] {
                    outR[p] = backgroundFinal.r[p]
                    outG[p] = backgroundFinal.g[p]
                    outB[p] = backgroundFinal.b[p]
                }
                return RGBBuffer(r: outR, g: outG, b: outB, width: backgroundFinal.width, height: backgroundFinal.height)
            }

            // Streams straight to bgWriter instead of building a [CGImage]
            // array: each iteration computes ONE frame's reconstructed-
            // background pixels via renderReconFrame, writes it to
            // reconstructed_background.mp4, then lets it fall out of scope
            // before the next iteration -- at most one frame's worth of
            // CGImage/RGBBuffer is resident at a time, versus the old
            // design's full N-frame [CGImage] array (~1.97GB at 300
            // frames/1280x1280). The n/2 frame is also captured into
            // reconMidPreview for the gallery, mirroring how
            // VideoFrameLoader captures specific preview indices during its
            // own streaming decode pass rather than keeping everything.
            (bgReconFrameCount, reconMidPreview) = try await Task.detached(priority: .userInitiated) {
                var count = 0
                var midPreview: CGImage?
                for i in 0..<n {
                    let frameBuf = renderReconFrame(i)
                    if let cg = frameBuf.toCGImage() {
                        try bgWriter.append(cg)
                        count += 1
                        if i == n / 2 { midPreview = cg }
                    }
                }
                return (count, midPreview)
            }.value

        case .dynamicWindowed:
            // DYNAMIC: no single global plate exists by construction --
            // each window covers only its own frame range. Core-fill runs
            // PER WINDOW (fillWindowCores), then every frame's own
            // background is rendered via compositeFrame(), which blends
            // every covering window's plate (trapezoidal cross-fade,
            // exposure-matched) into that frame's hole region.
            let windows = reconResult.dynamicWindows ?? []
            appendLog("[diag] running per-window core-fill (\(windows.count) window(s), bbox-cropped LaMa or push-pull if a window's core > 35%)...")
            let methods: [String] = await Task.detached(priority: .userInitiated) {
                autoreleasepool {
                    BackgroundReconstructor.fillWindowCores(windows, lamaRunner: lamaRunner)
                }
            }.value
            for (idx, m) in methods.enumerated() { appendLog("  window \(idx) [\(windows[idx].start)-\(windows[idx].end)): \(m)") }
            usedPushPull = methods.contains { $0.hasPrefix("push-pull") }
            lamaStageMs = 0 // per-window LaMa timing isn't broken out individually here; align/trim already include window-build cost
            coreFillMethodSummary = methods.joined(separator: "; ")
            coreFillCorePct = 100.0 * Double(reconResult.core.filter { $0 }.count) / Double(reconResult.core.count)
            backgroundFinal = windows.first?.plate ?? reconResult.plateBeforeLama

            appendLog("[diag] compositing per-frame windowed background (\(n) frames, trapezoidal cross-fade across \(windows.count) window(s))...")
            // renderReconFrame for this branch: compositeFrame blends every
            // covering window's plate into frame i's hole region
            // (trapezoidal cross-fade, exposure-matched) -- same per-frame
            // computation this branch always did, factored out so Loop 3
            // can call it again later instead of reading back a stored
            // array. `windows` is captured by this closure.
            renderReconFrame = { i in
                BackgroundReconstructor.compositeFrame(colorBuffers[i].toFloatRGBBuffer(), mask: maskBuffers[i].unpacked(), frameIndex: i, windows: windows)
            }

            // Same streaming-to-bgWriter rewrite as the STATIC/JITTER
            // branch above -- see that branch's comment for the full
            // reasoning. Only the per-frame compute differs (compositeFrame
            // vs. the direct paste, both now behind renderReconFrame); the
            // write-then-discard discipline is identical.
            (bgReconFrameCount, reconMidPreview) = try await Task.detached(priority: .userInitiated) {
                var count = 0
                var midPreview: CGImage?
                for i in 0..<n {
                    let composited = renderReconFrame(i)
                    if let cg = composited.toCGImage() {
                        try bgWriter.append(cg)
                        count += 1
                        if i == n / 2 { midPreview = cg }
                    }
                }
                return (count, midPreview)
            }.value
        }

        // Close out reconstructed_background.mp4 now that both branches'
        // loop has fully finished writing to it. Saved to Photos and
        // registered in outputVideos here (rather than in the old shared
        // "Export" do-block at the very end of this function) because the
        // writer -- and the only CGImages it ever touched -- belongs to
        // THIS stage; there is no reason to defer closing the file until
        // after Illumination Extraction/Final Compositing run.
        try await bgWriter.finish()
        outputVideos.append((label: "Masked Input", url: maskedURL))
        outputVideos.append((label: "Reconstructed Background", url: bgURL))
        try await VideoEncoder.saveToPhotoLibrary(url: bgURL)
        if let reconMidPreview {
            addPreview("Reconstructed Background\n(mid-clip)", reconMidPreview)
        }

        let bgReconTotalMs = reconResult.alignMs + reconResult.trimmedMeanMs + lamaStageMs
        appendLog("  Background Reconstruction TOTAL: \(String(format: "%.0f", bgReconTotalMs))ms (\(n) frames, \(maskedVideo.width)x\(maskedVideo.height))")
        addPreview("Background FINAL\n(after core-fill)", backgroundFinal.toCGImage())
        // Method badge surfaces the self-decided STATIC/JITTER vs DYNAMIC
        // (windowed) branch -- the same "did a fallback path trigger?"
        // at-a-glance role the badge already played for LaMa vs push-pull,
        // now covering the higher-level algorithm choice too (PipelineStage
        // only has one badge slot -- see PipelineStageView.swift, out of
        // scope to modify -- so the push-pull anti-hallucination warning is
        // folded in as a suffix rather than getting a second badge).
        let methodBadgeText = reconResult.method == .dynamicWindowed
            ? "DYNAMIC windowed\(usedPushPull ? " + push-pull ⚠" : "")"
            : "STATIC/JITTER\(usedPushPull ? " + push-pull ⚠" : "")"
        setStage("Background Reconstruction", status: .done,
                  timings: [("align", reconResult.alignMs), ("trim", reconResult.trimmedMeanMs), ("fill", lamaStageMs)],
                  method: (methodBadgeText, usedPushPull),
                  detail: [reconResult.methodDetail, coreFillMethodSummary, "core: \(String(format: "%.1f", coreFillCorePct))% of frame"])
        appendLog("  reconstructed-background video: \(bgReconFrameCount) frames written to reconstructed_background.mp4, each using its OWN mask for the fill region")
        appendLog("  Saved reconstructed_background.mp4 to Photos (\(bgReconFrameCount) frames)")

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
        // Streams to silhouette_on_lightmap.mp4 the same way Loop 1 streams
        // to reconstructed_background.mp4 above -- opened here (right
        // before this loop, dimensions/fps already known), appended to
        // inside the loop, closed right after. Previously this built a
        // [CGImage] for all n frames (~1.97GB at 300 frames/1280x1280)
        // purely so it could be handed to VideoEncoder.encode(frames:...)
        // at export time and to grab index n/2 for the gallery -- neither
        // need survives the loop now: export is inline via silWriter, and
        // the n/2 CGImage is captured directly into silMidPreview as it's
        // produced.
        let silURL = tmpDir.appendingPathComponent("silhouette_on_lightmap.mp4")
        let silWriter = try StreamingVideoWriter(outputURL: silURL, width: lightmapForComposite.width, height: lightmapForComposite.height, fps: fps)
        let (silFrameCount, silMidPreview): (Int, CGImage?) = try await Task.detached(priority: .userInitiated) {
            var count = 0
            var midPreview: CGImage?
            for i in 0..<n {
                let alpha = maskBuffers[i].unpacked().isPerson.map { $0 ? Float(1) : Float(0) }
                let result = Compositor.compositeOnly(background: lightmapForComposite, character: colorBuffers[i].toFloatRGBBuffer(), alpha: alpha)
                if let cg = result.composited.toCGImage() {
                    try silWriter.append(cg)
                    count += 1
                    if i == n / 2 { midPreview = cg }
                }
            }
            return (count, midPreview)
        }.value
        try await silWriter.finish()
        outputVideos.append((label: "Silhouette on Lightmap", url: silURL))
        try await VideoEncoder.saveToPhotoLibrary(url: silURL)
        appendLog("  silhouette-on-lightmap: \(silFrameCount) frames composited, saved to Photos")
        if let silMidPreview {
            addPreview("Silhouette-on-Lightmap\n(TO SERVER)", silMidPreview)
        }

        // Steps 5-6: the server call (WanAnimate) can't be made from this
        // local benchmark. When the active preset has a real
        // cloud-generated character staged (ClipPreset.hasCharacter --
        // e.g. CASE1's synthetic_person_p1.mp4/synthetic_alpha_p1.mp4),
        // use it instead of the PLACEHOLDER avatar so this stage can be
        // visually/quantitatively compared against Android's real
        // reference outputs. Falls back to the exact previous placeholder
        // behavior, unchanged, whenever no character is staged (the
        // bundled sample clip, and any preset that hasn't had a character
        // staged) -- see ClipPreset.hasCharacter's doc comment for why a
        // preset without one is still perfectly usable. Step 6 composites
        // the avatar (real or placeholder) onto the RECONSTRUCTED
        // background (not the lightmap) to produce the final video.
        setStage("Final Compositing", status: .running)

        let realCharacter = clipLibrary.activeCharacter
        // realCharacterData is loaded here (once, before the per-frame
        // loop) rather than inside it -- same streaming-then-hold-compact-
        // arrays discipline as colorBuffers/maskBuffers above, NOT a
        // per-frame re-decode. The character video loads via the SAME
        // VideoFrameLoader.loadFramesAsRGBBuffer8 streaming path used for
        // masked_video.mp4 (compact RGBBuffer8, UInt8, 3 bytes/pixel -- same
        // element type/memory class as colorBuffers, not a return to
        // Float32 or [CGImage]). The alpha matte loads via
        // loadFramesAsGrayBuffer8 instead (added in the 2026-08-03
        // memory-budget pass): it only ever needs 1 channel's worth of
        // information (see GrayBuffer8's doc comment in PixelBuffer.swift),
        // so storing it as a 3-channel RGBBuffer8 like an earlier version of
        // this code did was a real, avoidable 3x memory cost (~1.44GB ->
        // ~0.48GB at 300 frames/1264x1264) with zero corresponding benefit --
        // nothing ever read the redundant G/B channels. This is the fourth
        // video loaded into a full-clip compact array by this function
        // (masked, mask, character, alpha); all four stay bounded/compact,
        // just not all the same element type anymore.
        struct RealCharacterData {
            let character: [RGBBuffer8]
            let alpha: [GrayBuffer8]
            let width: Int
            let height: Int
        }
        // `let`, not `var`: assigned exactly once via this immediately-
        // invoked closure rather than mutated in place after a `var`
        // declaration -- Swift 6 strict concurrency rejects a captured
        // `var` inside the `Task.detached` closure below ("reference to
        // captured var 'realCharacterData' in concurrently-executing
        // code", hit for real in CI 2026-08-03) even though this function
        // only ever reads it after this point. A `let` bound once, up
        // front, sidesteps that check entirely and is honestly the more
        // correct shape anyway -- nothing downstream needs to reassign it.
        let realCharacterData: RealCharacterData? = try {
            guard let realCharacter else { return nil }
            appendLog("[diag] staged character detected (\(clipLibrary.activePreset?.name ?? "preset")) -- loading REAL synthetic character + alpha matte instead of the placeholder avatar...")
            let charVideo = try VideoFrameLoader.loadFramesAsRGBBuffer8(url: realCharacter.character)
            let alphaVideo = try VideoFrameLoader.loadFramesAsGrayBuffer8(url: realCharacter.alpha)
            guard charVideo.width == maskedVideo.width && charVideo.height == maskedVideo.height else {
                throw NSError(domain: "BenchmarkView", code: 10, userInfo: [NSLocalizedDescriptionKey:
                    "character video is \(charVideo.width)x\(charVideo.height) but the masked/background clip is \(maskedVideo.width)x\(maskedVideo.height) -- refusing to composite mismatched resolutions (would silently misalign or crash mid-loop). Re-export the character video at the clip's resolution."])
            }
            guard alphaVideo.width == maskedVideo.width && alphaVideo.height == maskedVideo.height else {
                throw NSError(domain: "BenchmarkView", code: 11, userInfo: [NSLocalizedDescriptionKey:
                    "character alpha video is \(alphaVideo.width)x\(alphaVideo.height) but the masked/background clip is \(maskedVideo.width)x\(maskedVideo.height) -- refusing to composite mismatched resolutions. Re-export the alpha matte at the clip's resolution."])
            }
            appendLog("  loaded \(charVideo.frames.count) character frames + \(alphaVideo.frames.count) alpha frames (\(charVideo.width)x\(charVideo.height))")
            return RealCharacterData(character: charVideo.frames, alpha: alphaVideo.frames, width: charVideo.width, height: charVideo.height)
        }()

        // Frame-count policy: extend the existing n = min(maskedCount,
        // maskCount) pattern to also bound by the character/alpha frame
        // counts when a real character is staged, so a shorter
        // character/alpha clip can't run the loop past the end of its own
        // arrays. CASE1's 4 inputs are all exactly 300 frames, so nFinal
        // == n there; this only matters for future clips where a
        // generated character render came back short (a realistic cloud
        // failure mode -- e.g. WanAnimate truncating on an error frame).
        let nFinal: Int
        if let rc = realCharacterData {
            nFinal = min(n, rc.character.count, rc.alpha.count)
            if nFinal < n {
                appendLog("  [warn] character/alpha clip (\(rc.character.count)/\(rc.alpha.count) frames) is shorter than the masked/background clip (\(n) frames) -- truncating Final Compositing to \(nFinal) frames.")
            }
        } else {
            nFinal = n
            appendLog("[diag] simulating server response (PLACEHOLDER avatar, no real WanAnimate call -- a DIFFERENT synthetic frame is generated per source frame, not one cached still, so Compositing's timing reflects real per-frame data volume)...")
        }

        appendLog("[diag] running Final Compositing: \(realCharacterData != nil ? "REAL synthetic character" : "placeholder avatar") onto per-frame reconstructed background (relight EXCLUDED per scope decision)...")
        // Streams to final_output.mp4 exactly like Loop 1/Loop 2 above.
        // The one structural difference from before: this loop needs
        // "frame i's reconstructed-background content" as its own per-frame
        // input, which used to come from reading back Loop 1's
        // reconstructedBgFrames[i] CGImage array. That array no longer
        // exists (Loop 1 streams straight to disk and discards), so this
        // calls renderReconFrame(i) -- the SAME closure Loop 1 used to
        // produce that exact content -- to recompute it in place. See
        // renderReconFrame's declaration above the switch statement for the
        // full reasoning on why recomputing (option a) beats decoding
        // reconstructed_background.mp4 back from disk (option b) here.
        let finalURL = tmpDir.appendingPathComponent("final_output.mp4")
        let finalWriter = try StreamingVideoWriter(outputURL: finalURL, width: backgroundFinal.width, height: backgroundFinal.height, fps: fps)
        let (finalFrameCount, totalCompositeMs, finalMidPreview): (Int, Double, CGImage?) = try await Task.detached(priority: .userInitiated) {
            var count = 0
            var totalMs = 0.0
            var midPreview: CGImage?
            for i in 0..<nFinal {
                let frameBg = renderReconFrame(i)
                let character: RGBBuffer
                let alpha: [Float]
                if let rc = realCharacterData {
                    // Both arrays already live fully in memory (loaded
                    // above, bounded/compact like colorBuffers) -- indexing
                    // frame i here does not decode or allocate a new
                    // clip-wide array per frame, only per-frame Float32
                    // promotion of ONE frame (same pattern as
                    // colorBuffers[i].toFloatRGBBuffer() elsewhere in this
                    // function).
                    character = rc.character[i].toFloatRGBBuffer()
                    alpha = rc.alpha[i].alphaChannel8To01()
                } else {
                    (character, alpha) = Compositor.placeholderCharacter(width: backgroundFinal.width, height: backgroundFinal.height, frameIndex: i, totalFrames: nFinal)
                }
                let result = Compositor.compositeOnly(background: frameBg, character: character, alpha: alpha)
                totalMs += result.compositeMs
                if let cg = result.composited.toCGImage() {
                    try finalWriter.append(cg)
                    count += 1
                    if i == nFinal / 2 { midPreview = cg }
                }
            }
            return (count, totalMs, midPreview)
        }.value
        appendLog("  Final Compositing: \(finalFrameCount) frames, composite total=\(String(format: "%.1f", totalCompositeMs))ms")
        if let finalMidPreview {
            addPreview(realCharacterData != nil ? "FINAL\n(REAL character on reconstructed bg, no relight)" : "FINAL\n(avatar on reconstructed bg, no relight)", finalMidPreview)
        }
        setStage("Final Compositing", status: .done,
                  timings: [("composite", totalCompositeMs)],
                  method: realCharacterData != nil ? ("REAL character", false) : ("placeholder", false))

        // Video export: reconstructed_background.mp4 and
        // silhouette_on_lightmap.mp4 were already written+saved inline,
        // right after their own stage's loop finished (see bgWriter/
        // silWriter above) -- only final_output.mp4 (just produced) still
        // needs closing out and saving. All three videos ARE still real
        // .mp4 files on disk, viewable/scrubbable via Photos exactly as
        // before -- only WHEN each one gets closed/saved moved earlier,
        // from one shared do-block at the very end to right after each
        // stage's own loop, since there's no longer a stage-spanning
        // [CGImage] array forcing everything to wait until the end.
        setStage("Export", status: .running)
        appendLog("\n[diag] finishing output videos (\(n) real frames each\(nFinal != n ? ", except final_output.mp4 which is truncated to \(nFinal)" : ""), not repeated stills)...")
        do {
            try await finalWriter.finish()
            outputVideos.append((label: "Final Output", url: finalURL))
            try await VideoEncoder.saveToPhotoLibrary(url: finalURL)
            appendLog("  Saved final_output.mp4 to Photos (\(finalFrameCount) frames)")
            if realCharacterData != nil {
                appendLog("  NOTE: avatar is the REAL staged synthetic character + alpha matte (\(clipLibrary.activePreset?.name ?? "preset")), not the placeholder -- background behind it is real per-frame video.")
            } else {
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
        appendLog("    TOTAL (3 stages):          \(String(format: "%.0f", totalMs))ms for \(nFinal) src frames")
        if realCharacterData != nil {
            appendLog("  NOTE: Final Compositing used the REAL staged synthetic character + alpha matte -- tests both compositing MATH cost AND visual fidelity against Android's reference outputs.")
        } else {
            appendLog("  NOTE: Final Compositing uses a PLACEHOLDER character cutout, not a real WanAnimate render -- tests compositing MATH cost only, not visual fidelity.")
        }
        appendLog("  Scroll the image strip above to visually verify each stage's output before trusting these numbers.")

        lastRunSummary = "Total \(String(format: "%.0f", totalMs))ms for \(nFinal) frames · \(reconResult.method.rawValue) · core-fill: \(coreFillMethodSummary)\(realCharacterData != nil ? " · REAL character" : "")"
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
