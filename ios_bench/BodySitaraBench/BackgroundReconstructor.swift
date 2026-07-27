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
    struct Result {
        let plateBeforeLama: RGBBuffer
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

        func sadAt(_ ddx: Int, _ ddy: Int) -> Double {
            let y0 = cy0 - half + ddy
            let x0 = cx0 - half + ddx
            guard y0 >= 0, x0 >= 0, y0 + 2 * half <= height, x0 + 2 * half <= width else { return .infinity }
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
                let s = sadAt(dx, dy)
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

        let sxLeft = sadAt(bestDx - 1, bestDy), sxRight = sadAt(bestDx + 1, bestDy)
        let syUp = sadAt(bestDx, bestDy - 1), syDown = sadAt(bestDx, bestDy + 1)
        let sx = sxLeft.isFinite && sxRight.isFinite ? parabolic(sxLeft, bestSad, sxRight) : 0.0
        let sy = syUp.isFinite && syDown.isFinite ? parabolic(syUp, bestSad, syDown) : 0.0

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
    /// matches cv2.warpAffine(..., borderMode=BORDER_REPLICATE) for color
    /// and borderValue=1 (i.e. "treat out-of-bounds as person/hole") for
    /// masks, selected via `nearest`.
    static func warpTranslate(_ src: [Float], width: Int, height: Int, dx: Double, dy: Double, nearest: Bool) -> [Float] {
        var out = [Float](repeating: 0, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                let sx = Double(x) - dx
                let sy = Double(y) - dy
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

    /// Per-pixel temporal trimmed-mean across all aligned frames: for each
    /// pixel, gather values from every frame where it's real background
    /// (not person-hole), sort, and mean the middle 60% (drop lowest/
    /// highest 20% of the VALID samples, matching robustCenter() /
    /// trimmed_mean_vectorized()'s per-pixel-n_valid semantics -- not a
    /// fixed global trim count). Runs once per color channel.
    static func trimmedMean(channelStack: [[Float]], validStack: [[Bool]], width: Int, height: Int, trimFrac: Float = 0.20) -> (plate: [Float], neverRevealed: [Bool]) {
        let n = channelStack.count
        let pixCount = width * height
        var plate = [Float](repeating: 0, count: pixCount)
        var neverRevealed = [Bool](repeating: true, count: pixCount)

        var samples = [Float](repeating: 0, count: n)
        for p in 0..<pixCount {
            var count = 0
            for f in 0..<n where validStack[f][p] {
                samples[count] = channelStack[f][p]
                count += 1
            }
            guard count > 0 else { continue }
            neverRevealed[p] = false

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
        return (plate, neverRevealed)
    }

    /// Full stage: align every frame to frame 0, then trimmed-mean-aggregate
    /// all channels. Does NOT run LaMa (caller does that separately via
    /// LamaRunner, matching the existing per-stage timing discipline).
    static func reconstruct(colorFrames: [RGBBuffer], masks: [MaskBuffer]) -> Result {
        precondition(!colorFrames.isEmpty, "need at least 1 frame")
        let width = colorFrames[0].width
        let height = colorFrames[0].height
        let n = colorFrames.count

        let tAlignStart = CFAbsoluteTimeGetCurrent()
        let refGray = colorFrames[0].grayscale()

        var alignedR = [colorFrames[0].r]
        var alignedG = [colorFrames[0].g]
        var alignedB = [colorFrames[0].b]
        var alignedValid = [masks[0].isPerson.map { !$0 }]

        for i in 1..<n {
            let tgtGray = colorFrames[i].grayscale()
            let (dx, dy) = alignPyramid(ref: refGray, tgt: tgtGray, width: width, height: height)
            alignedR.append(warpTranslate(colorFrames[i].r, width: width, height: height, dx: dx, dy: dy, nearest: false))
            alignedG.append(warpTranslate(colorFrames[i].g, width: width, height: height, dx: dx, dy: dy, nearest: false))
            alignedB.append(warpTranslate(colorFrames[i].b, width: width, height: height, dx: dx, dy: dy, nearest: false))
            let maskF = masks[i].isPerson.map { $0 ? Float(1) : Float(0) }
            let warpedMaskF = warpTranslate(maskF, width: width, height: height, dx: dx, dy: dy, nearest: true)
            alignedValid.append(warpedMaskF.map { $0 < 0.5 })
        }
        let alignMs = (CFAbsoluteTimeGetCurrent() - tAlignStart) * 1000

        let tTrimStart = CFAbsoluteTimeGetCurrent()
        let (rPlate, neverRevealed) = trimmedMean(channelStack: alignedR, validStack: alignedValid, width: width, height: height)
        let (gPlate, _) = trimmedMean(channelStack: alignedG, validStack: alignedValid, width: width, height: height)
        let (bPlate, _) = trimmedMean(channelStack: alignedB, validStack: alignedValid, width: width, height: height)
        let trimmedMeanMs = (CFAbsoluteTimeGetCurrent() - tTrimStart) * 1000

        let plate = RGBBuffer(r: rPlate, g: gPlate, b: bPlate, width: width, height: height)
        return Result(plateBeforeLama: plate, neverRevealed: neverRevealed, alignMs: alignMs, trimmedMeanMs: trimmedMeanMs)
    }
}
