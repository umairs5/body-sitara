import CoreGraphics
import Accelerate
import UIKit
import CoreML

/// Swift port of Danial's Android Reveal-and-Fill (BackgroundInpaint.kt),
/// matching this project's own validated Python reference
/// (scripts/reveal_and_fill_static.py) line-for-line in algorithm
/// structure for the STATIC/JITTER path, and BackgroundInpaint.kt's
/// windowed DYNAMIC path (computeTrajectory / buildWindowedPlates /
/// borrowAcrossWindows) for clips with real camera motion.
///
/// Runs fully on-device (no precomputed alignment) so the reported
/// Background Reconstruction latency is a real, honest end-to-end number,
/// comparable to Android's measured 33.59s/34.1s figures at the same
/// pipeline stage -- not just the LaMa call in isolation.
///
/// MEMORY MODEL (rewritten 2026-07-31 to fix a real OOM crash -- see
/// commit 7cefad6 "Shrink Tier2-mobile test clip to 50 frames/~10fps --
/// fixes on-device OOM crash"):
///
/// The ORIGINAL implementation held `alignedR`/`alignedG`/`alignedB`/
/// `alignedValid` as `[[Float]]` (one full-resolution Float32 plane PER
/// FRAME, for all N frames simultaneously) so `trimmedMean` could sort
/// each pixel's full temporal sample set. At 1264x1264, one Float32 RGB
/// frame is 1264*1264*3*4 bytes = ~19.17MB. A real 300-frame clip (10s at
/// 30fps) needs 300 * 19.17MB = ~5.75GB for the aligned RGB stack ALONE
/// (before the ~480MB for masks, and before the still-resident original
/// decoded `colorFrames` this function receives as a parameter) --
/// comfortably in jetsam-kill territory on a real device, confirmed by
/// the July crash: even the interim 300-frame attempt failed at
/// frame-loading time on an iPhone 15 Pro Max.
///
/// FIX: aligned per-frame planes are now stored as `[UInt8]` (one byte
/// per channel per pixel -- lossless for the purpose, since every source
/// pixel is already an 8-bit 0-255 value; sub-pixel precision only
/// matters DURING warping/gain, not in the at-rest stack) via
/// `AlignedFrameStack`, a 4x reduction vs Float32:
///   300 frames * 1264*1264*3*1 byte = ~1.44GB for the aligned RGB stack.
/// The per-pixel "valid" flags are packed into a bitset (`PackedBoolPlane`,
/// 1 bit/pixel instead of Swift's 1 BYTE/Bool), an 8x reduction on that
/// side: 300 * 1264*1264/8 bytes = ~60MB (was ~480MB as `[[Bool]]`).
/// Combined resident footprint for a 300-frame native-res clip:
/// ~1.44GB + ~60MB = ~1.5GB -- still substantial, but a documented,
/// bounded, ~4x-under-the-known-failure-threshold number instead of the
/// ~6.2GB the old Float32 design needed, and comfortably below the
/// multi-GB range that produced the confirmed jetsam kill.
///
/// `trimmedMean` fundamentally needs every valid sample for a pixel
/// gathered before it can sort and trim (a bounded reservoir would bias
/// the trim toward whichever samples happened to survive eviction, which
/// is not the same statistic Android computes) -- so this is NOT
/// restructured into a fully streaming single-pass accumulator. Instead,
/// `trimmedMean` now processes the frame in horizontal ROW BANDS
/// (`trimBandRows`), so the only per-call scratch allocation
/// (`samples: [Float]`) is a single small buffer reused across every
/// pixel/band rather than the caller needing a second full-frame-sized
/// scratch array -- this keeps the aggregation step's PEAK additional
/// memory small and predictable on top of the resident UInt8 stack,
/// rather than adding another full-resolution Float32 array (the
/// `plate`/`neverRevealed` outputs are still one full-res array each,
/// which is unavoidable -- they ARE the output).
///
/// In DYNAMIC mode, each window only holds its OWN frame range's
/// `AlignedFrameStack` (WINDOW_MAX=96 frames at most, not the whole
/// clip), and windows are built and torn down one at a time inside
/// `buildWindow` -- so DYNAMIC mode's peak resident aligned-stack memory
/// is bounded by one window's worth of frames, not O(N) or O(K*window),
/// even though the clip may need several windows to cover its full
/// length.
///
/// Exposure-gain and the trimmed-mean's sort/trim math still run in
/// Float32 (converted from the resident UInt8 sample on the fly, per
/// pixel, per frame) for numerical precision during aggregation -- only
/// the AT-REST storage moved to UInt8, matching the task's guidance to
/// keep accumulation precision while cutting resident footprint.
///
/// SECOND MEMORY GAP FIXED 2026-07-31 (this fix was INCOMPLETE the first
/// time): the rewrite above narrowed the INTERNAL aligned-frame copy this
/// file builds inside `reconstructStatic`/`reconstructDynamic`, but never
/// touched the INPUT to `reconstruct()`. `colorFrames` was still typed
/// `[RGBBuffer]` -- i.e. the CALLER (BenchmarkView.swift's
/// `colorBuffers`) had to build and hold N full-resolution Float32
/// RGBBuffers simultaneously just to call this function at all, before a
/// single byte of the (now-fixed) UInt8 stack existed. At 300 frames,
/// 1280x1280: 300 * 1280*1280*4 bytes/channel*3 channels = ~5.9GB for
/// THAT array alone -- comfortably larger than the ~1.5GB internal fix
/// this header already documented, which is why a real 300-frame on-
/// device run still jetsam-killed even after the earlier fix landed.
///
/// FIX: `reconstruct()` (and everything it calls: `computeTrajectory`,
/// `reconstructStatic`, `reconstructDynamic`, `buildWindow`,
/// `compositeFrame`) now takes `[RGBBuffer8]` (UInt8-native, see
/// PixelBuffer.swift) instead of `[RGBBuffer]`, and `[PackedMaskFrame]`
/// (bit-packed, reusing `PackedBoolPlane` below) instead of
/// `[MaskBuffer]`. Every call site that needs one FRAME's Float32/`[Bool]`
/// data converts that ONE frame on the fly (`.toFloatRGBBuffer()`/
/// `.unpacked()`), uses it for that frame's alignment/gain/trajectory
/// math, and lets it fall out of scope -- never retaining a second
/// N-length Float32/`[Bool]` array alongside the UInt8/packed one. See
/// PixelBuffer.swift's `RGBBuffer8`/`PackedMaskFrame` doc comments for the
/// full before/after memory math at this input-loading layer.
enum BackgroundReconstructor {
    /// px: dilate the person mask OUT before sampling, so codec-bleed /
    /// silhouette-edge pixels never contaminate the plate -- matches
    /// Android's MASK_DILATE (BackgroundInpaint.kt).
    static let maskDilate = 5
    /// Minimum number of valid (real-background) temporal samples a pixel
    /// needs before it's trusted as "revealed" -- matches Android's
    /// minCov = max(3, 5% of sampled frames). A pixel seen in only 1-2
    /// frames out of hundreds is not reliably real background (could be a
    /// sliver of alignment error), so it's still routed to the neural/
    /// push-pull core if it was EVER part of the hole union.
    static func minCoverage(sampledFrames: Int) -> Int { max(3, sampledFrames / 20) }

    /// Row-band size for trimmed-mean's per-pixel sample gathering -- keeps
    /// the scratch `samples` buffer's peak size bounded (band width) rather
    /// than needing a separate full-frame-sized scratch array. Purely a
    /// loop-structuring constant, doesn't change the result.
    static let trimBandRows = 64

    enum Method: String {
        case staticJitter = "STATIC/JITTER"
        case dynamicWindowed = "DYNAMIC (windowed)"
    }

    struct Result {
        let plateBeforeLama: RGBBuffer
        /// True core: pixels with < minCoverage real samples AND part of
        /// the hole union -- these need neural/push-pull fill. Distinct
        /// from `neverRevealed` (zero samples) per Android's core vs.
        /// never-revealed distinction (BackgroundInpaint.kt line 422-426).
        let core: [Bool]
        /// Union of the (dilated) hole across every aligned frame -- the
        /// full region that ever needed reconstruction.
        let union: [Bool]
        let neverRevealed: [Bool]
        let alignMs: Double
        let trimmedMeanMs: Double
        /// Which branch actually ran -- Android self-decides STATIC/JITTER
        /// vs DYNAMIC windowed based on measured camera motion
        /// (computeTrajectory); surfaced here so the benchmark dashboard
        /// can show a method badge confirming which path executed on a
        /// given clip, the same way LaMa vs push-pull is already surfaced.
        let method: Method
        /// Diagnostic detail for the method badge/log (e.g. motion
        /// deviation, window count) -- empty for the STATIC path.
        let methodDetail: String
        /// Non-nil only in DYNAMIC mode: the built windows, needed by the
        /// caller to render each frame's own windowed composite via
        /// `BackgroundReconstructor.compositeFrame`. STATIC mode's callers
        /// instead paste `plateBeforeLama`-after-core-fill into each
        /// frame's own mask region directly (existing BenchmarkView logic).
        let dynamicWindows: [WindowPlate]?
        /// Per-window wall-clock ms for `buildWindow`, in build order --
        /// empty for STATIC. Added 2026-08-03 after a real DYNAMIC-path
        /// device run measured `alignMs=346727ms` for 4 windows (~87-96
        /// frames each), a ~7x-per-frame-alignment slowdown vs. the
        /// STATIC path's known-good ~41-45s/300-frame figures, WITHOUT any
        /// corresponding algorithmic difference found on close review of
        /// `buildWindow`/`alignPyramid`/`alignTranslation` (DYNAMIC's
        /// windowPyramidLevels is a strict SUBSET of STATIC's default
        /// levels -- 2 levels vs. 3, missing exactly the expensive native-
        /// resolution search level -- so per-frame cost should be equal or
        /// LOWER in DYNAMIC, not higher; total frame-alignment count across
        /// all windows with the logged len=96/overlap=16/hop=80 params is
        /// only ~1.15-1.3x STATIC's per-clip count, not 7-8x). No
        /// algorithmic root cause was found despite a full trace of the
        /// call graph -- this per-window breakdown is added so the NEXT
        /// real device run can show whether the slowdown is roughly UNIFORM
        /// across all 4 windows (consistent with progressive thermal
        /// throttling over the sustained ~5.8-minute single-stage compute
        /// burst -- plausible since this is the first clip ever to run
        /// `buildWindow` long enough on real hardware to hit sustained
        /// thermal pressure) or concentrated in one outlier window
        /// (which would instead point at a content-specific issue in that
        /// window's frame range, e.g. unusually large search-radius misses
        /// forcing repeated fallback behavior) -- rather than guessing
        /// which explanation is correct without the data to distinguish
        /// them.
        let perWindowAlignMs: [Double]
    }

    // MARK: - Packed / compact storage

    /// 1 bit per pixel instead of Swift's 1 BYTE per `Bool` -- an 8x
    /// reduction for the per-frame "valid" (not-hole) plane, which is the
    /// second-largest resident allocation after the UInt8 RGB stack (see
    /// the memory-model doc comment above).
    struct PackedBoolPlane {
        private(set) var words: [UInt64]
        let count: Int

        init(_ src: [Bool]) {
            count = src.count
            var w = [UInt64](repeating: 0, count: (count + 63) / 64)
            for i in 0..<count where src[i] {
                w[i >> 6] |= (1 << UInt64(i & 63))
            }
            words = w
        }

        @inline(__always) func get(_ i: Int) -> Bool {
            (words[i >> 6] >> UInt64(i & 63)) & 1 != 0
        }
    }

    /// A single frame's aligned color data, stored at 1 byte/channel/pixel
    /// (UInt8) instead of the original 4 bytes/channel/pixel (Float32) --
    /// the core of the memory fix. `valid` is bit-packed (see
    /// PackedBoolPlane). Exposure gain is already baked into `r`/`g`/`b`
    /// at construction time (applied once, in Float32, then rounded/
    /// clamped down to UInt8) -- matches the original code's per-frame
    /// gain-then-store order, just narrowing the "at rest" representation.
    struct AlignedFrame {
        var r: [UInt8]
        var g: [UInt8]
        var b: [UInt8]
        var valid: PackedBoolPlane
    }

    /// Bounded-footprint stack of aligned frames for one aggregation pass
    /// (either the whole clip in STATIC mode, or one window's frame range
    /// in DYNAMIC mode). Frames are appended one at a time as they're
    /// aligned, so the original full-resolution Float32 `RGBBuffer` for
    /// frame i is never retained past the point where frame i has been
    /// warped+gained+packed into this stack.
    final class AlignedFrameStack {
        private(set) var frames: [AlignedFrame] = []
        let width: Int
        let height: Int

        init(width: Int, height: Int) {
            self.width = width
            self.height = height
        }

        func append(r: [Float], g: [Float], b: [Float], hole: [Bool]) {
            let n = width * height
            var r8 = [UInt8](repeating: 0, count: n)
            var g8 = [UInt8](repeating: 0, count: n)
            var b8 = [UInt8](repeating: 0, count: n)
            for i in 0..<n {
                r8[i] = UInt8(min(255, max(0, r[i].rounded())))
                g8[i] = UInt8(min(255, max(0, g[i].rounded())))
                b8[i] = UInt8(min(255, max(0, b[i].rounded())))
            }
            let validBools = hole.map { !$0 }
            frames.append(AlignedFrame(r: r8, g: g8, b: b8, valid: PackedBoolPlane(validBools)))
        }
    }

    /// 3-level coarse->fine->native alignment pyramid, per the documented
    /// levels: (100px, +-16px), (300px, +-4px), (native, +-3px). Returns
    /// (dx, dy) subpixel shift of `tgt` relative to `ref`.
    static func alignPyramid(ref: [Float], tgt: [Float], width: Int, height: Int) -> (dx: Double, dy: Double) {
        alignPyramid(ref: ref, tgt: tgt, width: width, height: height, levels: [(100, 16), (300, 4), (max(width, height), 3)])
    }

    /// Generalized pyramid entry used by both the STATIC path's 3-level
    /// pyramid and the DYNAMIC path's lighter 2-level WINDOW_PYRAMID
    /// (no native-resolution level -- windows are aligned ref-to-ref and
    /// frame-to-ref far more often than the single STATIC plate, so the
    /// per-alignment cost is kept down by skipping the expensive
    /// native-res refinement step, matching Android's WINDOW_PYRAMID).
    static func alignPyramid(ref: [Float], tgt: [Float], width: Int, height: Int, levels: [(targetDim: Int, radius: Int)]) -> (dx: Double, dy: Double) {
        var totalDx = 0.0
        var totalDy = 0.0
        var curTgt = tgt

        for (targetDim, radius) in levels {
            let scale = min(1.0, Double(targetDim) / Double(max(width, height)))
            let (refS, tgtS, w2, h2): ([Float], [Float], Int, Int)
            if scale < 1.0 {
                let rw = max(1, Int(Double(width) * scale))
                let rh = max(1, Int(Double(height) * scale))
                refS = resizeArea(ref, width: width, height: height, newW: rw, newH: rh)
                tgtS = resizeArea(curTgt, width: width, height: height, newW: rw, newH: rh)
                w2 = rw; h2 = rh
            } else {
                refS = ref; tgtS = curTgt; w2 = width; h2 = height
            }

            let (dx, dy) = alignTranslation(refGray: refS, tgtGray: tgtS, width: w2, height: h2, searchRadius: radius)
            let (adjDx, adjDy) = scale < 1.0 ? (dx / scale, dy / scale) : (dx, dy)
            totalDx += adjDx
            totalDy += adjDy

            curTgt = warpTranslate(tgt, width: width, height: height, dx: -totalDx, dy: -totalDy, nearest: false)
        }
        return (totalDx, totalDy)
    }

    /// SAD-based integer-pixel translation search within +-searchRadius on
    /// a central patch, then parabolic sub-pixel refinement per axis --
    /// matches align_translation()/parabolic_subpixel() in the Python ref.
    private static func alignTranslation(refGray: [Float], tgtGray: [Float], width: Int, height: Int, searchRadius: Int) -> (Double, Double) {
        let cy0 = height / 2
        let cx0 = width / 2
        let half = min(height, width) / 4
        guard half > 0 else { return (0, 0) }

        // Out-of-bounds probes return `bestSadRef` (a real, finite fallback
        // value), matching Python's sad_at()'s `return best_sad` -- NOT
        // infinity. This matters for the parabolic sub-pixel refinement:
        // an earlier version returned .infinity here, which silently
        // zeroed sub-pixel refinement (via the isFinite guard below)
        // whenever a probe neighbor was out of bounds, contributing to
        // the on-device alignment bug alongside the warpTranslate sign
        // error fixed in the same pass.
        func sadAt(_ ddx: Int, _ ddy: Int, fallback: Double) -> Double {
            let y0 = cy0 - half + ddy
            let x0 = cx0 - half + ddx
            guard y0 >= 0, x0 >= 0, y0 + 2 * half <= height, x0 + 2 * half <= width else { return fallback }
            var sum = 0.0
            for y in 0..<(2 * half) {
                let refRow = (cy0 - half + y) * width + (cx0 - half)
                let tgtRow = (y0 + y) * width + x0
                for x in 0..<(2 * half) {
                    sum += abs(Double(refGray[refRow + x]) - Double(tgtGray[tgtRow + x]))
                }
            }
            return sum
        }

        var bestSad = Double.infinity
        var bestDx = 0
        var bestDy = 0
        for dy in -searchRadius...searchRadius {
            for dx in -searchRadius...searchRadius {
                let s = sadAt(dx, dy, fallback: .infinity)
                if s < bestSad {
                    bestSad = s
                    bestDx = dx
                    bestDy = dy
                }
            }
        }

        func parabolic(_ m1: Double, _ zero: Double, _ p1: Double) -> Double {
            let denom = m1 - 2 * zero + p1
            guard abs(denom) >= 1e-6 else { return 0.0 }
            return 0.5 * (m1 - p1) / denom
        }

        let sxLeft = sadAt(bestDx - 1, bestDy, fallback: bestSad)
        let sxRight = sadAt(bestDx + 1, bestDy, fallback: bestSad)
        let syUp = sadAt(bestDx, bestDy - 1, fallback: bestSad)
        let syDown = sadAt(bestDx, bestDy + 1, fallback: bestSad)
        let sx = parabolic(sxLeft, bestSad, sxRight)
        let sy = parabolic(syUp, bestSad, syDown)

        return (Double(bestDx) + sx, Double(bestDy) + sy)
    }

    private static func resizeArea(_ src: [Float], width: Int, height: Int, newW: Int, newH: Int) -> [Float] {
        var srcBuf = src
        var dstBuf = [Float](repeating: 0, count: newW * newH)
        var srcVImage = vImage_Buffer(data: &srcBuf, height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: width * MemoryLayout<Float>.size)
        var dstVImage = vImage_Buffer(data: &dstBuf, height: vImagePixelCount(newH), width: vImagePixelCount(newW), rowBytes: newW * MemoryLayout<Float>.size)
        vImageScale_PlanarF(&srcVImage, &dstVImage, nil, vImage_Flags(kvImageHighQualityResampling))
        return dstBuf
    }

    /// Translates `src` (width x height, single channel or interleaved via
    /// caller looping per-channel) by (dx, dy) using bilinear or
    /// nearest-neighbor sampling with edge-replicate border handling --
    /// matches cv2.warpAffine(src, [[1,0,-dx],[0,1,-dy]], ...) for color
    /// and borderValue=1 (i.e. "treat out-of-bounds as person/hole") for
    /// masks, selected via `nearest`.
    ///
    /// SIGN CONVENTION (verified directly against real cv2.warpAffine
    /// output, 2026-07-27 -- an earlier version of this function had this
    /// backwards, which produced the on-device smeared/doubled-frame bug):
    /// cv2.warpAffine with M=[[1,0,-dx],[0,1,-dy]] samples
    /// dst(x,y) = src(x + dx, y + dy), NOT src(x - dx, y - dy). Confirmed
    /// with a real test: a stripe at src column 10, total_dx=5, ends up at
    /// dst column 5 (i.e. dst(x)=src(x+dx) -> the stripe that WAS at x=10
    /// is now read out at x=10-dx=5). So the correct sample point is
    /// `x + dx`, not `x - dx`.
    static func warpTranslate(_ src: [Float], width: Int, height: Int, dx: Double, dy: Double, nearest: Bool) -> [Float] {
        var out = [Float](repeating: 0, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                let sx = Double(x) + dx
                let sy = Double(y) + dy
                out[y * width + x] = nearest
                    ? sampleNearestReplicate(src, width: width, height: height, x: sx, y: sy)
                    : sampleBilinearReplicate(src, width: width, height: height, x: sx, y: sy)
            }
        }
        return out
    }

    private static func sampleNearestReplicate(_ src: [Float], width: Int, height: Int, x: Double, y: Double) -> Float {
        let xi = max(0, min(width - 1, Int(x.rounded())))
        let yi = max(0, min(height - 1, Int(y.rounded())))
        return src[yi * width + xi]
    }

    private static func sampleBilinearReplicate(_ src: [Float], width: Int, height: Int, x: Double, y: Double) -> Float {
        let x0 = Int(floor(x)), y0 = Int(floor(y))
        let fx = Float(x - Double(x0)), fy = Float(y - Double(y0))
        func clampX(_ v: Int) -> Int { max(0, min(width - 1, v)) }
        func clampY(_ v: Int) -> Int { max(0, min(height - 1, v)) }
        let x0c = clampX(x0), x1c = clampX(x0 + 1), y0c = clampY(y0), y1c = clampY(y0 + 1)
        let v00 = src[y0c * width + x0c], v10 = src[y0c * width + x1c]
        let v01 = src[y1c * width + x0c], v11 = src[y1c * width + x1c]
        let top = v00 * (1 - fx) + v10 * fx
        let bottom = v01 * (1 - fx) + v11 * fx
        return top * (1 - fy) + bottom * fy
    }

    /// Bilinear sample of a full RGBBuffer plate at fractional (x, y) --
    /// used by cross-window borrowing and per-frame windowed compositing
    /// to pull a real pixel from a (possibly different) plate at a
    /// shifted world coordinate.
    static func bilinearRGB(_ plate: RGBBuffer, x: Double, y: Double) -> (r: Float, g: Float, b: Float) {
        let w = plate.width, h = plate.height
        return (
            sampleBilinearReplicate(plate.r, width: w, height: h, x: x, y: y),
            sampleBilinearReplicate(plate.g, width: w, height: h, x: x, y: y),
            sampleBilinearReplicate(plate.b, width: w, height: h, x: x, y: y)
        )
    }

    /// In-place binary dilation by radius `r` (separable max-filter over a
    /// [Bool] plane) -- matches Android's dilateInPlace(). Only ever adds
    /// `true` pixels, never removes them.
    static func dilate(_ mask: [Bool], width: Int, height: Int, radius: Int) -> [Bool] {
        guard radius > 0 else { return mask }
        var tmp = [Bool](repeating: false, count: mask.count)
        for y in 0..<height {
            let row = y * width
            for x in 0..<width {
                var on = false
                var d = -radius
                while d <= radius {
                    let xx = x + d
                    if xx >= 0, xx < width, mask[row + xx] { on = true; break }
                    d += 1
                }
                tmp[row + x] = on
            }
        }
        var out = [Bool](repeating: false, count: mask.count)
        for x in 0..<width {
            for y in 0..<height {
                var on = false
                var d = -radius
                while d <= radius {
                    let yy = y + d
                    if yy >= 0, yy < height, tmp[yy * width + x] { on = true; break }
                    d += 1
                }
                out[y * width + x] = on
            }
        }
        return out
    }

    /// Mean luma over the non-hole (real background) pixels of a frame --
    /// used to exposure-normalize each frame's samples before aggregation
    /// (matches Android's meanLuma()/referenceMeanLuma()).
    private static func meanLuma(_ frame: RGBBuffer, hole: [Bool]) -> Double {
        var sum = 0.0
        var count = 0
        for i in 0..<(frame.width * frame.height) where !hole[i] {
            sum += (Double(frame.r[i]) + Double(frame.g[i]) + Double(frame.b[i])) / 3.0
            count += 1
        }
        return count > 0 ? sum / Double(count) : 128.0
    }

    /// Per-pixel temporal trimmed-mean across all aligned frames in
    /// `stack`: for each pixel, gather values from every frame where it's
    /// real background (not person-hole), sort, and mean the middle 60%
    /// (drop lowest/highest 20% of the VALID samples, matching
    /// robustCenter() / trimmed_mean_vectorized()'s per-pixel-n_valid
    /// semantics -- not a fixed global trim count). Runs once per color
    /// channel, in horizontal row bands (`trimBandRows`) purely to keep
    /// the loop structure friendly to a bounded scratch buffer -- the
    /// result is identical to a single flat loop over all pixels. Returns
    /// the aggregated plate channel plus `neverRevealed` (zero real
    /// samples) and `lowCoverage` (some samples, but fewer than minCov --
    /// still routed to the core if part of the hole union at the call
    /// site). Matches Android's per-pixel k>=minCov / 0<k<minCov / k==0
    /// three-way split (aggregatePlate() line 422-426).
    static func trimmedMean(stack: AlignedFrameStack, channel: KeyPath<AlignedFrame, [UInt8]>, trimFrac: Float = 0.20) -> (plate: [Float], neverRevealed: [Bool], lowCoverage: [Bool]) {
        let n = stack.frames.count
        let width = stack.width, height = stack.height
        let pixCount = width * height
        let minCov = minCoverage(sampledFrames: n)
        var plate = [Float](repeating: 0, count: pixCount)
        var neverRevealed = [Bool](repeating: true, count: pixCount)
        var lowCoverage = [Bool](repeating: false, count: pixCount)

        var samples = [Float](repeating: 0, count: n)
        var band = 0
        while band < height {
            let bandEnd = min(band + trimBandRows, height)
            for y in band..<bandEnd {
                for x in 0..<width {
                    let p = y * width + x
                    var count = 0
                    for f in 0..<n where stack.frames[f].valid.get(p) {
                        samples[count] = Float(stack.frames[f][keyPath: channel][p])
                        count += 1
                    }
                    guard count > 0 else { continue }
                    neverRevealed[p] = false
                    lowCoverage[p] = count < minCov

                    let validSlice = samples[0..<count].sorted()
                    let lo = Int(Float(count) * trimFrac)
                    let hi = count - lo
                    if hi <= lo {
                        plate[p] = validSlice.reduce(0, +) / Float(count)
                    } else {
                        let kept = validSlice[lo..<hi]
                        plate[p] = kept.reduce(0, +) / Float(kept.count)
                    }
                }
            }
            band = bandEnd
        }
        return (plate, neverRevealed, lowCoverage)
    }

    /// Full stage: self-decides between the STATIC/JITTER single-plate
    /// path and the DYNAMIC windowed path based on measured camera motion
    /// (computeTrajectory), mirroring Android's
    /// `if (traj.needsWindowing) { ...windowed... } else { ...static... }`
    /// structure (BackgroundInpaint.kt).
    ///
    /// `colorFrames`/`masks` are UInt8-native/bit-packed (`RGBBuffer8`/
    /// `PackedMaskFrame`), NOT the Float32 `RGBBuffer`/`MaskBuffer` this
    /// function used before 2026-07-31 -- see the enum's header doc for
    /// why (this is the fix for the input-layer OOM gap the previous
    /// UInt8-stack rewrite missed).
    static func reconstruct(colorFrames: [RGBBuffer8], masks: [PackedMaskFrame]) -> Result {
        precondition(!colorFrames.isEmpty, "need at least 1 frame")
        let width = colorFrames[0].width
        let height = colorFrames[0].height
        let n = colorFrames.count

        let traj = computeTrajectory(colorFrames: colorFrames, masks: masks, width: width, height: height)
        if traj.needsWindowing && n >= WINDOW_MIN {
            return reconstructDynamic(colorFrames: colorFrames, masks: masks, width: width, height: height, traj: traj)
        }
        return reconstructStatic(colorFrames: colorFrames, masks: masks, width: width, height: height, traj: traj)
    }

    // MARK: - STATIC/JITTER path

    /// Aligns every frame to frame 0, dilates each frame's mask (keeps
    /// codec-bleed/silhouette-edge pixels out of the plate samples,
    /// matching MASK_DILATE), exposure-normalizes each frame's samples to
    /// the reference frame's mean luma before aggregating, then
    /// trimmed-mean-aggregates all channels -- via the bounded-memory
    /// `AlignedFrameStack` (UInt8 storage, see the type's/enum's memory-
    /// model doc comments) rather than the old `[[Float]]` design.
    ///
    /// `colorFrames`/`masks` arrive as `RGBBuffer8`/`PackedMaskFrame` (see
    /// enum header) -- each frame is promoted to Float32/`[Bool]` via
    /// `.toFloatRGBBuffer()`/`.unpacked()` ONLY for the duration of that
    /// frame's own loop iteration (`frame0`, and `colorFrames[i]`
    /// per-iteration below), so at most 1-2 Float32 frames (reference +
    /// current) are resident at once, never all N.
    private static func reconstructStatic(colorFrames: [RGBBuffer8], masks: [PackedMaskFrame], width: Int, height: Int, traj: Traj) -> Result {
        let n = colorFrames.count
        let pixCount = width * height

        let tAlignStart = CFAbsoluteTimeGetCurrent()
        let frame0 = colorFrames[0].toFloatRGBBuffer()
        let mask0 = masks[0].unpacked()
        let refGray = frame0.grayscale()
        let refHole0 = dilate(mask0.isPerson, width: width, height: height, radius: maskDilate)
        let refMeanLuma = meanLuma(frame0, hole: refHole0)

        let stack = AlignedFrameStack(width: width, height: height)
        stack.append(r: frame0.r, g: frame0.g, b: frame0.b, hole: refHole0)
        var union = refHole0

        for i in 1..<n {
            // Dilate this frame's mask, align, exposure-gain, and pack
            // straight into the UInt8 stack -- the source frame's Float32
            // promotion (`frameI`, a LOCAL to this iteration only) is
            // never additionally retained by this function once packed;
            // the caller's `colorFrames[i]` stays UInt8 the whole time.
            let frameI = colorFrames[i].toFloatRGBBuffer()
            let hole = dilate(masks[i].unpacked().isPerson, width: width, height: height, radius: maskDilate)
            let tgtGray = frameI.grayscale()
            let (dx, dy) = alignPyramid(ref: refGray, tgt: tgtGray, width: width, height: height)

            // Exposure-normalize this frame's samples to the reference's
            // mean luma before they enter the temporal stack -- otherwise
            // auto-exposure drift across frames blurs/smears the trimmed-
            // mean plate even with perfect alignment (matches Android's
            // per-frame gain in aggregatePlate(), clamped [0.85, 1.18]).
            let frameMeanLuma = meanLuma(frameI, hole: hole)
            let gain = Float(min(1.18, max(0.85, refMeanLuma / max(1.0, frameMeanLuma))))
            let gainedR = frameI.r.map { min(255, max(0, $0 * gain)) }
            let gainedG = frameI.g.map { min(255, max(0, $0 * gain)) }
            let gainedB = frameI.b.map { min(255, max(0, $0 * gain)) }

            let warpedR = warpTranslate(gainedR, width: width, height: height, dx: dx, dy: dy, nearest: false)
            let warpedG = warpTranslate(gainedG, width: width, height: height, dx: dx, dy: dy, nearest: false)
            let warpedB = warpTranslate(gainedB, width: width, height: height, dx: dx, dy: dy, nearest: false)
            let maskF = hole.map { $0 ? Float(1) : Float(0) }
            let warpedMaskF = warpTranslate(maskF, width: width, height: height, dx: dx, dy: dy, nearest: true)
            let warpedHole = warpedMaskF.map { $0 >= 0.5 }

            stack.append(r: warpedR, g: warpedG, b: warpedB, hole: warpedHole)
            for p in 0..<pixCount where warpedHole[p] { union[p] = true }
        }
        let alignMs = (CFAbsoluteTimeGetCurrent() - tAlignStart) * 1000

        let tTrimStart = CFAbsoluteTimeGetCurrent()
        let (rPlate, neverRevealedR, lowCovR) = trimmedMean(stack: stack, channel: \.r)
        let (gPlate, _, _) = trimmedMean(stack: stack, channel: \.g)
        let (bPlate, _, _) = trimmedMean(stack: stack, channel: \.b)
        var rPlateOut = rPlate, gPlateOut = gPlate, bPlateOut = bPlate
        var neverRevealed = neverRevealedR

        // True core = pixels the temporal aggregation can't be trusted for:
        // either never revealed at all, or revealed in too few frames to
        // trust (lowCoverage) while still being part of the hole union at
        // some point -- matches Android's core flag (aggregatePlate() line
        // 422-426), NOT a simple "revealed anywhere" test.
        var core = [Bool](repeating: false, count: pixCount)
        for p in 0..<pixCount {
            if neverRevealed[p] { core[p] = true }
            else if lowCovR[p] && union[p] { core[p] = true }
        }

        // Only pixels the REFERENCE frame's own mask actually covered need
        // reconstruction at all -- everywhere else is already real,
        // untouched background in frame 0. Restricting to this region
        // (rather than reconstructing the whole frame) eliminates
        // alignment-error risk everywhere outside the person's silhouette:
        // an imperfect cross-frame warp can only ever blur pixels that
        // genuinely needed filling, not perfectly good background that
        // never needed touching in the first place. This was the root
        // cause of the full-frame smearing seen on-device (2026-07-27) --
        // confirmed by the same fix in the Python reference
        // (reveal_and_fill_static.py) producing a clean, sharp result on
        // the same clip once applied.
        // Reuses `mask0`/`frame0` (already promoted above for the
        // reference-frame alignment/luma math) rather than re-deriving
        // from `masks[0]`/`colorFrames[0]` a second time.
        let needsReconstruction = mask0.isPerson
        for p in 0..<pixCount where !needsReconstruction[p] {
            rPlateOut[p] = frame0.r[p]
            gPlateOut[p] = frame0.g[p]
            bPlateOut[p] = frame0.b[p]
            neverRevealed[p] = false
            core[p] = false
            union[p] = false
        }
        let trimmedMeanMs = (CFAbsoluteTimeGetCurrent() - tTrimStart) * 1000

        let plate = RGBBuffer(r: rPlateOut, g: gPlateOut, b: bPlateOut, width: width, height: height)
        let detail = String(format: "devC=%.1fpx (R_MAX=%.1fpx) -- single plate covers the clip's motion", traj.devC, rMax(width: width))
        return Result(plateBeforeLama: plate, core: core, union: union, neverRevealed: neverRevealed, alignMs: alignMs, trimmedMeanMs: trimmedMeanMs, method: .staticJitter, methodDetail: detail, dynamicWindows: nil, perWindowAlignMs: [])
    }

    // MARK: - Motion detection (STATIC vs DYNAMIC self-decision)

    private static let coarseDim = 100
    private static let coarseSearch = 16
    private static let rMaxBase = 60.0

    struct Traj {
        let devC: Double
        let needsWindowing: Bool
    }

    private static func rMax(width: Int) -> Double { rMaxBase * Double(width) / 640.0 }

    private static func dimsFor(width: Int, height: Int, target: Int) -> (dw: Int, dh: Int) {
        let s = min(1.0, Double(target) / Double(max(width, height)))
        return (max(2, Int(Double(width) * s)), max(2, Int(Double(height) * s)))
    }

    /// Downscales a color frame + its person mask to (dw, dh) and returns
    /// (grayscale, valid) -- valid = NOT the (undilated) person-hole,
    /// matching Android's downGrayValid() used by the coarse motion
    /// pre-pass. Deliberately cheap: area resize via the existing vImage
    /// helper, no alignment/gain applied (this is only for the motion
    /// PRE-PASS, not the real aggregation).
    private static func downGrayValid(color: RGBBuffer, mask: MaskBuffer, dw: Int, dh: Int) -> (gray: [Float], valid: [Bool]) {
        let gray = color.grayscale()
        let grayDown = resizeArea(gray, width: color.width, height: color.height, newW: dw, newH: dh)
        let holeF = mask.isPerson.map { $0 ? Float(1) : Float(0) }
        let holeDown = resizeArea(holeF, width: color.width, height: color.height, newW: dw, newH: dh)
        let valid = holeDown.map { $0 < 0.5 }
        return (grayDown, valid)
    }

    /// CHAINED global-motion trajectory: consecutive-pair shifts SUMMED, so
    /// the measurable range is UNBOUNDED (a frame-0-anchored detector
    /// saturates + aliases on a smooth pan). devC = worst deviation from
    /// the trajectory's bbox center; > R_MAX(w) means one aligned plate
    /// can't cover it -> windowing. Ported from Android's
    /// computeTrajectory() (BackgroundInpaint.kt).
    ///
    /// Only ever touches at most 16 SAMPLED frames (`idxs`, evenly spread
    /// across the clip) out of the full N -- each one is promoted to
    /// Float32/`[Bool]` via `.toFloatRGBBuffer()`/`.unpacked()`,
    /// immediately downscaled to `(dw, dh)` (`coarseDim`=100px) by
    /// `downGrayValid`, and the full-res Float32 promotion is discarded
    /// right after -- so this pre-pass never holds more than 16 small
    /// downscaled planes resident, regardless of N.
    static func computeTrajectory(colorFrames: [RGBBuffer8], masks: [PackedMaskFrame], width: Int, height: Int) -> Traj {
        let n = colorFrames.count
        guard n >= 3 else { return Traj(devC: 0, needsWindowing: false) }

        let cnt = min(n, 16)
        let idxs: [Int] = (0..<cnt).map { cnt > 1 ? $0 * (n - 1) / (cnt - 1) : 0 }
        let (dw, dh) = dimsFor(width: width, height: height, target: coarseDim)

        var grays: [[Float]] = []
        var valids: [[Bool]] = []
        for t in idxs {
            let (g, v) = downGrayValid(color: colorFrames[t].toFloatRGBBuffer(), mask: masks[t].unpacked(), dw: dw, dh: dh)
            grays.append(g); valids.append(v)
        }
        let m = grays.count
        guard m >= 2 else { return Traj(devC: 0, needsWindowing: false) }

        let up = Double(width) / Double(dw)
        var tx = [Double](repeating: 0, count: m)
        var ty = [Double](repeating: 0, count: m)
        for k in 1..<m {
            let (sdx, sdy) = alignTranslationMasked(refGray: grays[k - 1], refValid: valids[k - 1], tgtGray: grays[k], tgtValid: valids[k], width: dw, height: dh, searchRadius: coarseSearch)
            tx[k] = tx[k - 1] + sdx * up
            ty[k] = ty[k - 1] + sdy * up
        }
        var mnx = tx[0], mxx = tx[0], mny = ty[0], mxy = ty[0]
        for k in 1..<m {
            mnx = min(mnx, tx[k]); mxx = max(mxx, tx[k])
            mny = min(mny, ty[k]); mxy = max(mxy, ty[k])
        }
        let cx = (mnx + mxx) / 2, cy = (mny + mxy) / 2
        var devC = 0.0
        for k in 0..<m {
            let d = hypot(tx[k] - cx, ty[k] - cy)
            if d > devC { devC = d }
        }
        return Traj(devC: devC, needsWindowing: devC > rMax(width: width))
    }

    /// Same integer-search + parabolic-subpixel `alignTranslation`, but
    /// respecting a valid-pixel mask (person-hole excluded) in the SAD sum
    /// -- the coarse motion pre-pass and cross-window ref-to-ref alignment
    /// both need to ignore the moving subject, not just raw pixel
    /// intensity, or a large/near-frame person would dominate the SAD and
    /// masquerade as camera motion.
    private static func alignTranslationMasked(refGray: [Float], refValid: [Bool], tgtGray: [Float], tgtValid: [Bool], width: Int, height: Int, searchRadius: Int) -> (Double, Double) {
        let cy0 = height / 2
        let cx0 = width / 2
        let half = min(height, width) / 4
        guard half > 0 else { return (0, 0) }

        func sadAt(_ ddx: Int, _ ddy: Int, fallback: Double) -> Double {
            let y0 = cy0 - half + ddy
            let x0 = cx0 - half + ddx
            guard y0 >= 0, x0 >= 0, y0 + 2 * half <= height, x0 + 2 * half <= width else { return fallback }
            var sum = 0.0
            for y in 0..<(2 * half) {
                let refRowBase = (cy0 - half + y) * width + (cx0 - half)
                let tgtRowBase = (y0 + y) * width + x0
                for x in 0..<(2 * half) {
                    let ri = refRowBase + x, ti = tgtRowBase + x
                    guard refValid[ri], tgtValid[ti] else { continue }
                    sum += abs(Double(refGray[ri]) - Double(tgtGray[ti]))
                }
            }
            return sum
        }

        var bestSad = Double.infinity
        var bestDx = 0
        var bestDy = 0
        for dy in -searchRadius...searchRadius {
            for dx in -searchRadius...searchRadius {
                let s = sadAt(dx, dy, fallback: .infinity)
                if s < bestSad {
                    bestSad = s
                    bestDx = dx
                    bestDy = dy
                }
            }
        }

        func parabolic(_ m1: Double, _ zero: Double, _ p1: Double) -> Double {
            let denom = m1 - 2 * zero + p1
            guard abs(denom) >= 1e-6 else { return 0.0 }
            return 0.5 * (m1 - p1) / denom
        }

        let sxLeft = sadAt(bestDx - 1, bestDy, fallback: bestSad)
        let sxRight = sadAt(bestDx + 1, bestDy, fallback: bestSad)
        let syUp = sadAt(bestDx, bestDy - 1, fallback: bestSad)
        let syDown = sadAt(bestDx, bestDy + 1, fallback: bestSad)
        let sx = parabolic(sxLeft, bestSad, sxRight)
        let sy = parabolic(syUp, bestSad, syDown)

        return (Double(bestDx) + sx, Double(bestDy) + sy)
    }

    // MARK: - DYNAMIC windowed path

    private static let WINDOW_MIN = 16
    private static let WINDOW_MAX = 96
    private static let MAX_WINDOWS = 8
    private static let windowPyramidLevels: [(targetDim: Int, radius: Int)] = [(100, 16), (300, 4)]

    /// One window's aggregated plate + bookkeeping needed for cross-window
    /// borrowing and per-frame compositing. `refIndex` is the source-frame
    /// index this window aligned everything to (the MIDDLE frame of the
    /// window, per Android, not frame 0) -- kept so `Aligner` can compute
    /// ref-to-ref shifts against other windows without re-deriving it.
    final class WindowPlate {
        var plate: RGBBuffer
        var core: [Bool]
        let union: [Bool]
        let aligner: Aligner
        let start: Int
        let end: Int
        let overlap: Int

        init(plate: RGBBuffer, core: [Bool], union: [Bool], aligner: Aligner, start: Int, end: Int, overlap: Int) {
            self.plate = plate; self.core = core; self.union = union
            self.aligner = aligner; self.start = start; self.end = end; self.overlap = overlap
        }
    }

    /// Per-window alignment context: the window's reference-frame grayscale
    /// (kept, small: one grayscale plane) plus per-frame shifts relative to
    /// that reference, so `shiftOf(frame:)` (per-frame compositing) and
    /// `refShiftTo(other:)` (cross-window borrowing) don't need to re-run
    /// the SAD search from scratch every time they're queried.
    final class Aligner {
        let refIndex: Int
        let refGray: [Float]
        private var shifts: [Int: (dx: Double, dy: Double)] = [:]
        let width: Int
        let height: Int

        init(refIndex: Int, refGray: [Float], width: Int, height: Int) {
            self.refIndex = refIndex; self.refGray = refGray
            self.width = width; self.height = height
        }

        func setShift(_ frame: Int, dx: Double, dy: Double) { shifts[frame] = (dx, dy) }
        func shiftOf(_ frame: Int) -> (dx: Double, dy: Double) { shifts[frame] ?? (0, 0) }

        /// Shift (dx, dy) such that `other.ref(x+dx, y+dy) ~= self.ref(x,y)`
        /// -- computed via the same multi-level SAD+parabolic search as the
        /// main aligner, between the two windows' reference-frame grayscale
        /// planes (both already resident, one plane each -- cheap).
        func refShiftTo(_ other: Aligner) -> (dx: Double, dy: Double) {
            BackgroundReconstructor.alignPyramid(ref: refGray, tgt: other.refGray, width: width, height: height, levels: BackgroundReconstructor.windowPyramidLevels)
        }
    }

    /// v (px/frame) estimate driving the window-length formula in
    /// buildWindowedPlates -- matches Android's `vPerFrame` local.
    private static func perFrameVelocity(devC: Double, n: Int) -> Double {
        n > 1 ? (2.0 * devC / Double(n - 1)) : 0.0
    }

    /// Builds and tears down each window's `AlignedFrameStack` ONE AT A
    /// TIME inside `buildWindow` (called from the loop below) -- so this
    /// function's own peak resident aligned-frame memory is bounded by a
    /// SINGLE window's worth of frames (WINDOW_MAX=96) at any instant, not
    /// O(N) or O(K*window) across however many windows the clip needs.
    /// `colorFrames`/`masks` themselves stay UInt8/bit-packed for the
    /// full clip the whole time this function runs (only passed BY
    /// REFERENCE into `buildWindow`, which does its own per-frame,
    /// per-window Float32/`[Bool]` promotion) -- so this input-layer fix
    /// doesn't undermine DYNAMIC mode's existing per-window boundedness,
    /// it just shrinks what "the whole clip" costs to keep resident
    /// alongside any one window's stack.
    private static func reconstructDynamic(colorFrames: [RGBBuffer8], masks: [PackedMaskFrame], width: Int, height: Int, traj: Traj) -> Result {
        let n = colorFrames.count
        let pixCount = width * height

        let tAlignStart = CFAbsoluteTimeGetCurrent()

        let rMaxW = rMax(width: width)
        let vPerFrame = perFrameVelocity(devC: traj.devC, n: n)
        let lenRaw = vPerFrame > 0.01 ? Int(2.0 * rMaxW / vPerFrame) : WINDOW_MAX
        let len = max(WINDOW_MIN, min(WINDOW_MAX, lenRaw))
        let overlap = max(4, len / 6)
        var hop = max(1, len - overlap)
        if n > len {
            // Widen the hop to cap the window count at MAX_WINDOWS on long
            // clips (bounds retained memory + per-window LaMa/align cost).
            // MUST stay <= len: a hop > len would skip frames entirely --
            // windowWeight() returns 0 for any frame outside every window's
            // [start,end), so a gap here silently drops those frames to the
            // PushPullFill fallback instead of the intended windowed
            // reconstruction, with no error or log signal (caught in
            // review, 2026-07-31: verified numerically at len=96, n=1000,
            // the uncapped formula produced hop=130 > len=96). Clamping to
            // len means windows can butt up edge-to-edge (zero overlap) in
            // the worst case, but every frame stays covered by at least one
            // window.
            hop = min(len, max(hop, (n - len + MAX_WINDOWS - 2) / (MAX_WINDOWS - 1)))
        }

        // Per-window timing (see Result.perWindowAlignMs's doc comment):
        // recorded so a real device run can show whether a slow total
        // alignMs is spread evenly across windows (thermal-throttling
        // signature) or concentrated in one window (content-specific
        // signature) -- the aggregate `alignMs` alone (as logged on the
        // first real DYNAMIC run, 2026-08-03: align=346727ms for 4
        // windows) can't distinguish those two cases.
        var windows: [WindowPlate] = []
        var perWindowMs: [Double] = []
        var start = 0
        while start < n {
            let end = min(start + len, n)
            let refT = (start + end) / 2
            let tWinStart = CFAbsoluteTimeGetCurrent()
            let win = buildWindow(colorFrames: colorFrames, masks: masks, width: width, height: height, start: start, end: end, refIndex: refT)
            perWindowMs.append((CFAbsoluteTimeGetCurrent() - tWinStart) * 1000)
            windows.append(win)
            if end >= n { break }
            start += hop
        }
        let alignMs = (CFAbsoluteTimeGetCurrent() - tAlignStart) * 1000

        let tTrimStart = CFAbsoluteTimeGetCurrent()
        borrowAcrossWindows(windows, width: width, height: height)

        // Union/core reported at the top level are the OR across all
        // windows -- matches "the full region that ever needed
        // reconstruction" semantics from the STATIC path, generalized to
        // however many windows contributed. Any remaining core pixel here
        // (a window's hole no neighbor could donate a real pixel for) still
        // needs a neural/push-pull fill -- the caller does that via
        // `fillWindowCores(_:lamaRunner:)` on `dynamicWindows` after this
        // function returns (same "reconstruct() hands back core, caller
        // fills it" split the STATIC path already uses), kept as a
        // per-window operation since a core pixel's neural fill only makes
        // sense in that window's own aligned coordinate frame.
        var unionAll = [Bool](repeating: false, count: pixCount)
        var coreAll = [Bool](repeating: false, count: pixCount)
        for w in windows {
            for p in 0..<pixCount {
                if w.union[p] { unionAll[p] = true }
                if w.core[p] { coreAll[p] = true }
            }
        }
        let trimmedMeanMs = (CFAbsoluteTimeGetCurrent() - tTrimStart) * 1000

        // `plateBeforeLama` reports the FIRST window's plate as a
        // representative preview image for the dashboard gallery --
        // DYNAMIC mode's real per-frame output comes from
        // `compositeFrame(frameIndex:)` over ALL windows, not a single
        // global plate (a single plate is exactly what STATIC mode has and
        // DYNAMIC mode does NOT, by construction).
        let previewPlate = windows.first?.plate ?? colorFrames[0].toFloatRGBBuffer()
        let detail = "devC=\(String(format: "%.1f", traj.devC))px > R_MAX=\(String(format: "%.1f", rMaxW))px -- \(windows.count) window(s), len=\(len) overlap=\(overlap) hop=\(hop)"
        return Result(plateBeforeLama: previewPlate, core: coreAll, union: unionAll, neverRevealed: coreAll, alignMs: alignMs, trimmedMeanMs: trimmedMeanMs, method: .dynamicWindowed, methodDetail: detail, dynamicWindows: windows, perWindowAlignMs: perWindowMs)
    }

    /// Aligns every frame within [start, end) to the window's MIDDLE frame
    /// (refIndex) and aggregates via the same trimmed-mean logic as the
    /// STATIC path, scoped to just this window's frames, via a fresh
    /// AlignedFrameStack that's released once this function returns --
    /// keeping DYNAMIC mode's peak resident aligned-frame memory bounded
    /// by ONE window's worth of frames, not the whole clip. Neural/push-
    /// pull core-fill for the window happens later, via
    /// `fillWindowCores(_:lamaRunner:)` (see its doc comment for why it's
    /// a separate call) -- matches Android's per-window body of
    /// buildWindowedPlates() up to (but not including) its fillCore() call.
    private static func buildWindow(colorFrames: [RGBBuffer8], masks: [PackedMaskFrame], width: Int, height: Int, start: Int, end: Int, refIndex: Int) -> WindowPlate {
        let pixCount = width * height
        let refFrame = colorFrames[refIndex].toFloatRGBBuffer()
        let refMask = masks[refIndex].unpacked()
        let refGray = refFrame.grayscale()
        let aligner = Aligner(refIndex: refIndex, refGray: refGray, width: width, height: height)

        let refHole = dilate(refMask.isPerson, width: width, height: height, radius: maskDilate)
        let refMeanLuma = meanLuma(refFrame, hole: refHole)

        let stack = AlignedFrameStack(width: width, height: height)
        var union = [Bool](repeating: false, count: pixCount)

        for i in start..<end {
            // Per-frame Float32/[Bool] promotion, scoped to this loop
            // iteration only (same pattern as reconstructStatic) -- keeps
            // this window's peak Float32 residency at O(1) frames, not
            // O(window length).
            let frameI = (i == refIndex) ? refFrame : colorFrames[i].toFloatRGBBuffer()
            let maskI = (i == refIndex) ? refMask : masks[i].unpacked()
            let hole = dilate(maskI.isPerson, width: width, height: height, radius: maskDilate)
            let dx: Double, dy: Double
            if i == refIndex {
                dx = 0; dy = 0
            } else {
                let tgtGray = frameI.grayscale()
                (dx, dy) = alignPyramid(ref: refGray, tgt: tgtGray, width: width, height: height, levels: windowPyramidLevels)
            }
            aligner.setShift(i, dx: dx, dy: dy)

            let frameMeanLuma = meanLuma(frameI, hole: hole)
            let gain = Float(min(1.18, max(0.85, refMeanLuma / max(1.0, frameMeanLuma))))
            let gainedR = frameI.r.map { min(255, max(0, $0 * gain)) }
            let gainedG = frameI.g.map { min(255, max(0, $0 * gain)) }
            let gainedB = frameI.b.map { min(255, max(0, $0 * gain)) }

            let isIdentity = (dx == 0 && dy == 0)
            let warpedR = isIdentity ? gainedR : warpTranslate(gainedR, width: width, height: height, dx: dx, dy: dy, nearest: false)
            let warpedG = isIdentity ? gainedG : warpTranslate(gainedG, width: width, height: height, dx: dx, dy: dy, nearest: false)
            let warpedB = isIdentity ? gainedB : warpTranslate(gainedB, width: width, height: height, dx: dx, dy: dy, nearest: false)
            let warpedHole: [Bool]
            if isIdentity {
                warpedHole = hole
            } else {
                let maskF = hole.map { $0 ? Float(1) : Float(0) }
                let warpedMaskF = warpTranslate(maskF, width: width, height: height, dx: dx, dy: dy, nearest: true)
                warpedHole = warpedMaskF.map { $0 >= 0.5 }
            }

            stack.append(r: warpedR, g: warpedG, b: warpedB, hole: warpedHole)
            for p in 0..<pixCount where warpedHole[p] { union[p] = true }
        }

        let (rPlate, neverRevealedR, lowCovR) = trimmedMean(stack: stack, channel: \.r)
        let (gPlate, _, _) = trimmedMean(stack: stack, channel: \.g)
        let (bPlate, _, _) = trimmedMean(stack: stack, channel: \.b)
        var rPlateOut = rPlate, gPlateOut = gPlate, bPlateOut = bPlate

        var core = [Bool](repeating: false, count: pixCount)
        for p in 0..<pixCount {
            if neverRevealedR[p] { core[p] = true }
            else if lowCovR[p] && union[p] { core[p] = true }
        }

        // Same "only the window's own reference-frame hole needs
        // reconstruction" restriction as STATIC, scoped to refIndex's
        // mask. Reuses `refMask`/`refFrame` (already promoted above)
        // rather than re-deriving from `masks[refIndex]`/
        // `colorFrames[refIndex]` a second time.
        let needsReconstruction = refMask.isPerson
        for p in 0..<pixCount where !needsReconstruction[p] {
            rPlateOut[p] = refFrame.r[p]
            gPlateOut[p] = refFrame.g[p]
            bPlateOut[p] = refFrame.b[p]
            core[p] = false
            union[p] = false
        }

        let plate = RGBBuffer(r: rPlateOut, g: gPlateOut, b: bPlateOut, width: width, height: height)
        let overlap = max(4, (end - start) / 6)
        return WindowPlate(plate: plate, core: core, union: union, aligner: aligner, start: start, end: end, overlap: overlap)
    }

    /// For each window's never-revealed core pixel, if an adjacent window
    /// (+-1, +-2) holds a REAL pixel at that world point (found via the
    /// ref->ref shift between the two windows' Aligners), copy it in and
    /// clear the core flag. Two rounds let reveals propagate across
    /// windows -- matches Android's borrowAcrossWindows().
    static func borrowAcrossWindows(_ windows: [WindowPlate], width: Int, height: Int) {
        guard windows.count >= 2 else { return }
        for _ in 0..<2 {
            for ki in windows.indices {
                let k = windows[ki]
                if !k.core.contains(true) { continue }
                for jd in [-1, 1, -2, 2] {
                    let ji = ki + jd
                    guard ji >= 0, ji < windows.count else { continue }
                    let j = windows[ji]
                    let shift = k.aligner.refShiftTo(j.aligner)
                    let dx = shift.dx, dy = shift.dy
                    for y in 0..<height {
                        for x in 0..<width {
                            let p = y * width + x
                            guard k.core[p] else { continue }
                            let sx = Double(x) + dx
                            let sy = Double(y) + dy
                            let ix = Int(sx.rounded()), iy = Int(sy.rounded())
                            guard ix >= 0, ix < width, iy >= 0, iy < height else { continue }
                            let jp = iy * width + ix
                            guard !j.core[jp] else { continue }
                            let (r, g, b) = bilinearRGB(j.plate, x: sx, y: sy)
                            k.plate.r[p] = r; k.plate.g[p] = g; k.plate.b[p] = b
                            k.core[p] = false
                        }
                    }
                }
            }
        }
    }

    /// Fills whatever core remains in each window AFTER cross-window
    /// borrowing (pixels no neighboring window could donate a real pixel
    /// for -- typically small slivers at the very start/end of the clip,
    /// or clips with only one window) via the SAME per-clip LamaRunner
    /// core-fill used by the STATIC path (bbox-cropped LaMa, or push-pull
    /// if a window's core exceeds 35% of the frame). Kept as an explicit
    /// post-`reconstruct()` step, NOT run inside `reconstruct()` itself,
    /// so `BackgroundReconstructor` (CoreGraphics/Accelerate/UIKit only)
    /// doesn't need to import CoreML / depend on LamaRunner's model-load
    /// lifecycle -- mirrors the existing STATIC-path call pattern in
    /// BenchmarkView.swift (`reconstruct()` returns `core`/`plateBeforeLama`,
    /// caller invokes `lamaRunner.fillCore` separately, off the main actor).
    /// Mutates each `WindowPlate.plate`/`core` in place and returns the
    /// per-window method strings (matching Android's per-window fillCore()
    /// log lines) for the caller to fold into a single method badge/log.
    ///
    /// PER-ITERATION `autoreleasepool` (added after the first real DYNAMIC-
    /// path device run crashed during/after this call, 2026-08-03): this
    /// loop calls `lamaRunner.fillCore` up to `MAX_WINDOWS` (8) times back
    /// to back, each one a full 1280x1280 CoreML LaMa inference. The
    /// caller (BenchmarkView.swift) wrapped the ENTIRE `fillWindowCores`
    /// call in a single `autoreleasepool { ... }` around the whole loop --
    /// which only drains ONCE, after all N windows' worth of
    /// Objective-C-backed CoreML MLMultiArray buffers have already piled
    /// up. That is exactly the crash pattern this exact codebase already
    /// root-caused and fixed once before, in `runLamaBenchmark`/
    /// `runRifeBenchmark` (see commit bb6b3a3, "Wrap each benchmark
    /// repetition in autoreleasepool -- fix rep-2 crash"): a real on-device
    /// test found LaMa's rep 1 succeeds cleanly but rep 2 (a second
    /// back-to-back large-tensor CoreML call with no pool drain in
    /// between) crashes hard, because large CoreML buffers can outlive
    /// their expected scope until the next autorelease-pool drain, and
    /// back-to-back large inferences pile up memory faster than ARC alone
    /// reclaims it. `fillWindowCores` has the identical shape (a tight
    /// loop of repeated large-tensor CoreML calls) but was missing the
    /// PER-CALL pool that fix established as the standard mitigation --
    /// the one-big-pool-around-the-whole-loop the call site had instead
    /// only protects against a single inference's leftover buffers, not
    /// against 4 (or up to 8) of them compounding across iterations, which
    /// is consistent with the observed crash landing during/right after
    /// "running per-window core-fill" on a real iPhone 15 Pro Max. Moving
    /// the pool IN HERE (one drain per window, not one for the whole
    /// batch) matches the established fix's granularity exactly.
    static func fillWindowCores(_ windows: [WindowPlate], lamaRunner: LamaRunner) -> [String] {
        var methods: [String] = []
        for win in windows {
            guard win.core.contains(true) else {
                methods.append("none (no core)")
                continue
            }
            autoreleasepool {
                do {
                    let result = try lamaRunner.fillCore(plate: win.plate, core: win.core)
                    win.plate = result.filled
                    methods.append(result.method)
                } catch {
                    methods.append("FAILED: \(error.localizedDescription)")
                }
            }
        }
        return methods
    }

    /// Trapezoidal window weight in [0,1]: full inside the window,
    /// smoothstep-ramped only in the overlap zones so adjacent windows
    /// crossfade -- every frame covered at weight ~1. Matches Android's
    /// windowWeight().
    static func windowWeight(_ win: WindowPlate, t: Int, isFirst: Bool, isLast: Bool) -> Float {
        guard t >= win.start, t < win.end else { return 0 }
        var wgt = 1.0
        let o = win.overlap
        if !isFirst, t < win.start + o {
            wgt = smoothstep((Double(t - win.start) + 0.5) / Double(o))
        }
        if !isLast, t >= win.end - o {
            wgt = min(wgt, smoothstep((Double(win.end - t) - 0.5) / Double(o)))
        }
        return Float(max(0, min(1, wgt)))
    }

    private static func smoothstep(_ x: Double) -> Double {
        let c = max(0, min(1, x))
        return c * c * (3 - 2 * c)
    }

    /// Per-frame windowed composite: for frame `t`, blends every covering
    /// window's plate (warped into frame t's coords via that window's
    /// Aligner.shiftOf(t), exposure-matched by meanLuma gain against the
    /// frame itself) weighted by `windowWeight`, normalizes by total
    /// weight, and alpha-blends into the frame's own (dilated) hole
    /// region. Falls back to PushPullFill for any hole pixel no window
    /// covers (rare, at clip edges). Caller (BenchmarkView) invokes this
    /// once per frame off the main actor, matching the existing
    /// Task.detached pattern for heavy per-frame work.
    static func compositeFrame(_ frame: RGBBuffer, mask: MaskBuffer, frameIndex: Int, windows: [WindowPlate]) -> RGBBuffer {
        let w = frame.width, h = frame.height
        let pixCount = w * h
        let hole = dilate(mask.isPerson, width: w, height: h, radius: maskDilate)
        guard hole.contains(true) else { return frame }

        var accumR = [Float](repeating: 0, count: pixCount)
        var accumG = [Float](repeating: 0, count: pixCount)
        var accumB = [Float](repeating: 0, count: pixCount)
        var accumW = [Float](repeating: 0, count: pixCount)

        let frameMeanLuma = meanLuma(frame, hole: hole)
        let emptyHole = [Bool](repeating: false, count: pixCount)

        for (idx, win) in windows.enumerated() {
            let wgt = windowWeight(win, t: frameIndex, isFirst: idx == 0, isLast: idx == windows.count - 1)
            guard wgt > 0 else { continue }
            let shift = win.aligner.shiftOf(frameIndex)
            let planeMeanLuma = meanLuma(win.plate, hole: emptyHole)
            let gain = Float(min(1.18, max(0.85, frameMeanLuma / max(1.0, planeMeanLuma))))
            // Sample the WINDOW's plate at (x - dx, y - dy) to land on the
            // same world point frame t's pixel (x,y) shows -- inverse of
            // the forward warp used when frames were aligned INTO the
            // window (see warpTranslate's SIGN CONVENTION doc comment:
            // dst(x,y) = src(x+dx, y+dy) forward, so recovering the
            // window-plate sample for a given frame pixel is src(x-dx,y-dy)).
            for y in 0..<h {
                for x in 0..<w {
                    let p = y * w + x
                    guard hole[p] else { continue }
                    let sx = Double(x) - shift.dx
                    let sy = Double(y) - shift.dy
                    let (r, g, b) = bilinearRGB(win.plate, x: sx, y: sy)
                    accumR[p] += r * gain * wgt
                    accumG[p] += g * gain * wgt
                    accumB[p] += b * gain * wgt
                    accumW[p] += wgt
                }
            }
        }

        var outR = frame.r, outG = frame.g, outB = frame.b
        var uncovered = [Bool](repeating: false, count: pixCount)
        for p in 0..<pixCount where hole[p] {
            if accumW[p] > 0 {
                outR[p] = min(255, max(0, accumR[p] / accumW[p]))
                outG[p] = min(255, max(0, accumG[p] / accumW[p]))
                outB[p] = min(255, max(0, accumB[p] / accumW[p]))
            } else {
                uncovered[p] = true
            }
        }
        var result = RGBBuffer(r: outR, g: outG, b: outB, width: w, height: h)
        if uncovered.contains(true) {
            result = PushPullFill.fill(plate: result, hole: uncovered)
        }
        return result
    }
}
