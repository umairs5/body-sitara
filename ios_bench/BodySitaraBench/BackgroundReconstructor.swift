import CoreGraphics
import Accelerate
import UIKit

/// Swift port of the STATIC/JITTER path of Danial's Android Reveal-and-Fill
/// (BackgroundInpaint.kt), matching this project's own validated Python
/// reference (scripts/reveal_and_fill_static.py) line-for-line in
/// algorithm structure -- NOT the DYNAMIC/windowed cross-window-borrowing
/// path (out of scope, per explicit instruction to port STATIC/JITTER
/// only, matching what's already been validated).
///
/// Runs fully on-device (no precomputed alignment) so the reported
/// Background Reconstruction latency is a real, honest end-to-end number,
/// comparable to Android's measured 33.59s/34.1s figures at the same
/// pipeline stage -- not just the LaMa call in isolation.
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
    }

    /// 3-level coarse->fine->native alignment pyramid, per the documented
    /// levels: (100px, +-16px), (300px, +-4px), (native, +-3px). Returns
    /// (dx, dy) subpixel shift of `tgt` relative to `ref`.
    static func alignPyramid(ref: [Float], tgt: [Float], width: Int, height: Int) -> (dx: Double, dy: Double) {
        var totalDx = 0.0
        var totalDy = 0.0
        var curTgt = tgt

        let levels: [(targetDim: Int, radius: Int)] = [(100, 16), (300, 4), (max(width, height), 3)]

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

    /// Per-pixel temporal trimmed-mean across all aligned frames: for each
    /// pixel, gather values from every frame where it's real background
    /// (not person-hole), sort, and mean the middle 60% (drop lowest/
    /// highest 20% of the VALID samples, matching robustCenter() /
    /// trimmed_mean_vectorized()'s per-pixel-n_valid semantics -- not a
    /// fixed global trim count). Runs once per color channel. Returns the
    /// aggregated plate channel plus `neverRevealed` (zero real samples)
    /// and `lowCoverage` (some samples, but fewer than minCov -- still
    /// routed to the core if part of the hole union at the call site).
    /// Matches Android's per-pixel k>=minCov / 0<k<minCov / k==0 three-way
    /// split (aggregatePlate() line 422-426).
    static func trimmedMean(channelStack: [[Float]], validStack: [[Bool]], width: Int, height: Int, trimFrac: Float = 0.20) -> (plate: [Float], neverRevealed: [Bool], lowCoverage: [Bool]) {
        let n = channelStack.count
        let pixCount = width * height
        let minCov = minCoverage(sampledFrames: n)
        var plate = [Float](repeating: 0, count: pixCount)
        var neverRevealed = [Bool](repeating: true, count: pixCount)
        var lowCoverage = [Bool](repeating: false, count: pixCount)

        var samples = [Float](repeating: 0, count: n)
        for p in 0..<pixCount {
            var count = 0
            for f in 0..<n where validStack[f][p] {
                samples[count] = channelStack[f][p]
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
        return (plate, neverRevealed, lowCoverage)
    }

    /// Full stage: align every frame to frame 0, dilate each frame's mask
    /// (keeps codec-bleed/silhouette-edge pixels out of the plate samples,
    /// matching MASK_DILATE), exposure-normalize each frame's samples to
    /// the reference frame's mean luma before aggregating, then
    /// trimmed-mean-aggregate all channels. Does NOT run LaMa/push-pull
    /// (caller does that separately via LamaRunner, matching the existing
    /// per-stage timing discipline).
    static func reconstruct(colorFrames: [RGBBuffer], masks: [MaskBuffer]) -> Result {
        precondition(!colorFrames.isEmpty, "need at least 1 frame")
        let width = colorFrames[0].width
        let height = colorFrames[0].height
        let n = colorFrames.count
        let pixCount = width * height

        let tAlignStart = CFAbsoluteTimeGetCurrent()
        let refGray = colorFrames[0].grayscale()

        // Dilate every frame's hole mask BEFORE alignment/sampling so the
        // person's silhouette edge (and any codec-bleed bordering it)
        // never counts as "real background" -- matches Android sampling
        // person-holes dilated by MASK_DILATE ahead of aggregatePlate().
        let dilatedHoles = masks.map { dilate($0.isPerson, width: width, height: height, radius: maskDilate) }
        let refMeanLuma = meanLuma(colorFrames[0], hole: dilatedHoles[0])

        var alignedR = [colorFrames[0].r]
        var alignedG = [colorFrames[0].g]
        var alignedB = [colorFrames[0].b]
        var alignedValid = [dilatedHoles[0].map { !$0 }]
        var union = dilatedHoles[0]

        for i in 1..<n {
            let tgtGray = colorFrames[i].grayscale()
            let (dx, dy) = alignPyramid(ref: refGray, tgt: tgtGray, width: width, height: height)

            // Exposure-normalize this frame's samples to the reference's
            // mean luma before they enter the temporal stack -- otherwise
            // auto-exposure drift across frames blurs/smears the trimmed-
            // mean plate even with perfect alignment (matches Android's
            // per-frame gain in aggregatePlate(), clamped [0.85, 1.18]).
            let frameMeanLuma = meanLuma(colorFrames[i], hole: dilatedHoles[i])
            let gain = Float(min(1.18, max(0.85, refMeanLuma / max(1.0, frameMeanLuma))))
            let gainedR = colorFrames[i].r.map { min(255, max(0, $0 * gain)) }
            let gainedG = colorFrames[i].g.map { min(255, max(0, $0 * gain)) }
            let gainedB = colorFrames[i].b.map { min(255, max(0, $0 * gain)) }

            alignedR.append(warpTranslate(gainedR, width: width, height: height, dx: dx, dy: dy, nearest: false))
            alignedG.append(warpTranslate(gainedG, width: width, height: height, dx: dx, dy: dy, nearest: false))
            alignedB.append(warpTranslate(gainedB, width: width, height: height, dx: dx, dy: dy, nearest: false))
            let maskF = dilatedHoles[i].map { $0 ? Float(1) : Float(0) }
            let warpedMaskF = warpTranslate(maskF, width: width, height: height, dx: dx, dy: dy, nearest: true)
            let warpedHole = warpedMaskF.map { $0 >= 0.5 }
            alignedValid.append(warpedHole.map { !$0 })
            for p in 0..<pixCount where warpedHole[p] { union[p] = true }
        }
        let alignMs = (CFAbsoluteTimeGetCurrent() - tAlignStart) * 1000

        let tTrimStart = CFAbsoluteTimeGetCurrent()
        let (rPlate, neverRevealedR, lowCovR) = trimmedMean(channelStack: alignedR, validStack: alignedValid, width: width, height: height)
        let (gPlate, _, _) = trimmedMean(channelStack: alignedG, validStack: alignedValid, width: width, height: height)
        let (bPlate, _, _) = trimmedMean(channelStack: alignedB, validStack: alignedValid, width: width, height: height)
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
        let needsReconstruction = masks[0].isPerson
        for p in 0..<pixCount where !needsReconstruction[p] {
            rPlateOut[p] = colorFrames[0].r[p]
            gPlateOut[p] = colorFrames[0].g[p]
            bPlateOut[p] = colorFrames[0].b[p]
            neverRevealed[p] = false
            core[p] = false
            union[p] = false
        }
        let trimmedMeanMs = (CFAbsoluteTimeGetCurrent() - tTrimStart) * 1000

        let plate = RGBBuffer(r: rPlateOut, g: gPlateOut, b: bPlateOut, width: width, height: height)
        return Result(plateBeforeLama: plate, core: core, union: union, neverRevealed: neverRevealed, alignMs: alignMs, trimmedMeanMs: trimmedMeanMs)
    }
}
