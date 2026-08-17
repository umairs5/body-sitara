import Accelerate
import CoreGraphics

/// Swift port of Danial's lightmap extraction (Android `LightmapPhase.kt`,
/// Phase 1b), per Tier_2_Mobile_Comprehensive_Guide.md.
///
/// PAPER PARITY (Table 12, apples-to-apples pass): Android's `LightmapPhase.
/// run()` does NOT compute one lightmap for the whole clip -- it decodes
/// `background_reconstructed.mp4` and calls its per-frame `lightmapOf()`
/// **once per frame**, writing a full `light_map.mp4` (see LightmapPhase.kt
/// lines 162-168, `for (t in 0 until n) { ... lightmapOf(frame, ...) ... }`).
/// The earlier iOS benchmark called `extract(from:)` exactly ONCE on the
/// single reconstructed plate, which is why Table 12 originally showed
/// 32.1s (Android, N frames) vs 0.009s (iOS, 1 frame) -- a scope mismatch,
/// not a hardware comparison. `extractAll` below reproduces Android's real
/// per-frame cost: it loops the same downscale/blur/upscale over every
/// frame the caller supplies, matching `LightmapPhase.kt`'s loop 1:1.
/// `extract(from:)` (single-plate) is kept for callers that only need ONE
/// lightmap value (kept for source compatibility; not used by the Table 12
/// benchmark path anymore).
///
/// Also updated to Android's CURRENT defaults (`LightmapPhase.DEF_SMALL`/
/// `DEF_BLUR_RADIUS`/`DEF_BLUR_SIGMA`): 32x32 downscale, radius 3, sigma
/// 1.2, with an intermediate 128x128 blur waypoint on the upscale --
/// Android's own docs flag its OLD 48x48/radius-2/sigma-3 pair as "a BUG,
/// not merely weak" (a Gaussian needs radius >= 2*sigma, so that pair
/// truncated into a flat 5-tap box), so porting the retired defaults here
/// would have reproduced a known, already-fixed defect rather than parity.
enum LightmapExtractor {
    static let downscaleSize = 32
    static let blurRadius = 3
    static let sigma: Float = 1.2
    /// Intermediate upscale waypoint -- smooths away bilinear mach banding
    /// for a rounding error of the cost of a full-resolution Gaussian.
    /// Matches Android's `MID_DIM`/`MID_BLUR_RADIUS`/`MID_BLUR_SIGMA`.
    static let midDim = 128
    static let midBlurRadius = 2
    static let midSigma: Float = 1.0

    struct Result {
        let lightmap: RGBBuffer
        let totalMs: Double
    }

    struct AllFramesResult {
        /// One lightmap per input frame, same order.
        let lightmaps: [RGBBuffer]
        let totalMs: Double
        let msPerFrame: Double
    }

    /// Per-frame extraction over a whole clip -- the Table 12 apples-to-
    /// apples entry point. `backgrounds[i]` is frame i's OWN reconstructed
    /// background (i.e. `renderReconFrame(i)` in BenchmarkView, matching
    /// Android reading its own frame `t` from `background_reconstructed.mp4`
    /// inside the loop), not a single shared plate.
    static func extractAll(backgrounds: [RGBBuffer]) -> AllFramesResult {
        let t0 = CFAbsoluteTimeGetCurrent()
        var out: [RGBBuffer] = []
        out.reserveCapacity(backgrounds.count)
        for bg in backgrounds {
            out.append(lightmapOf(bg))
        }
        let totalMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        let perFrame = backgrounds.isEmpty ? 0 : totalMs / Double(backgrounds.count)
        return AllFramesResult(lightmaps: out, totalMs: totalMs, msPerFrame: perFrame)
    }

    /// Single-plate extraction (kept for source compatibility / callers
    /// that only need one lightmap value, e.g. a quick preview).
    static func extract(from background: RGBBuffer) -> Result {
        let t0 = CFAbsoluteTimeGetCurrent()
        let lightmap = lightmapOf(background)
        let totalMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        return Result(lightmap: lightmap, totalMs: totalMs)
    }

    /// downscale (bilinear low-pass) -> Gaussian -> TWO-STAGE upscale
    /// (s -> midDim -> full), matching `lightmapOf` in LightmapPhase.kt.
    private static func lightmapOf(_ background: RGBBuffer) -> RGBBuffer {
        let w = background.width, h = background.height
        let smallR = downscale(background.r, width: w, height: h, newSize: downscaleSize)
        let smallG = downscale(background.g, width: w, height: h, newSize: downscaleSize)
        let smallB = downscale(background.b, width: w, height: h, newSize: downscaleSize)

        let kernel = gaussianKernel1D(radius: blurRadius, sigma: sigma)
        let blurredR = separableBlur(smallR, size: downscaleSize, kernel: kernel)
        let blurredG = separableBlur(smallG, size: downscaleSize, kernel: kernel)
        let blurredB = separableBlur(smallB, size: downscaleSize, kernel: kernel)

        let mid = min(midDim, max(w, h))
        let finalR: [Float]; let finalG: [Float]; let finalB: [Float]
        if mid > downscaleSize {
            let midUpR = upscale(blurredR, fromSize: downscaleSize, toW: mid, toH: mid)
            let midUpG = upscale(blurredG, fromSize: downscaleSize, toW: mid, toH: mid)
            let midUpB = upscale(blurredB, fromSize: downscaleSize, toW: mid, toH: mid)
            let midKernel = gaussianKernel1D(radius: midBlurRadius, sigma: midSigma)
            let midBlurR = separableBlur(midUpR, size: mid, kernel: midKernel)
            let midBlurG = separableBlur(midUpG, size: mid, kernel: midKernel)
            let midBlurB = separableBlur(midUpB, size: mid, kernel: midKernel)
            finalR = upscale(midBlurR, fromSize: mid, toW: w, toH: h)
            finalG = upscale(midBlurG, fromSize: mid, toW: w, toH: h)
            finalB = upscale(midBlurB, fromSize: mid, toW: w, toH: h)
        } else {
            finalR = upscale(blurredR, fromSize: downscaleSize, toW: w, toH: h)
            finalG = upscale(blurredG, fromSize: downscaleSize, toW: w, toH: h)
            finalB = upscale(blurredB, fromSize: downscaleSize, toW: w, toH: h)
        }
        return RGBBuffer(r: finalR, g: finalG, b: finalB, width: w, height: h)
    }

    private static func gaussianKernel1D(radius: Int, sigma: Float) -> [Float] {
        let size = 2 * radius + 1
        var kernel = [Float](repeating: 0, count: size)
        var sum: Float = 0
        for i in 0..<size {
            let x = Float(i - radius)
            let v = exp(-(x * x) / (2 * sigma * sigma))
            kernel[i] = v
            sum += v
        }
        for i in 0..<size { kernel[i] /= sum }
        return kernel
    }

    private static func downscale(_ src: [Float], width: Int, height: Int, newSize: Int) -> [Float] {
        var srcBuf = src
        var dstBuf = [Float](repeating: 0, count: newSize * newSize)
        var srcVImage = vImage_Buffer(data: &srcBuf, height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: width * MemoryLayout<Float>.size)
        var dstVImage = vImage_Buffer(data: &dstBuf, height: vImagePixelCount(newSize), width: vImagePixelCount(newSize), rowBytes: newSize * MemoryLayout<Float>.size)
        vImageScale_PlanarF(&srcVImage, &dstVImage, nil, vImage_Flags(kvImageHighQualityResampling))
        return dstBuf
    }

    private static func upscale(_ src: [Float], fromSize: Int, toW: Int, toH: Int) -> [Float] {
        var srcBuf = src
        var dstBuf = [Float](repeating: 0, count: toW * toH)
        var srcVImage = vImage_Buffer(data: &srcBuf, height: vImagePixelCount(fromSize), width: vImagePixelCount(fromSize), rowBytes: fromSize * MemoryLayout<Float>.size)
        var dstVImage = vImage_Buffer(data: &dstBuf, height: vImagePixelCount(toH), width: vImagePixelCount(toW), rowBytes: toW * MemoryLayout<Float>.size)
        vImageScale_PlanarF(&srcVImage, &dstVImage, nil, vImage_Flags(kvImageHighQualityResampling))
        return dstBuf
    }

    /// Separable Gaussian: horizontal pass then vertical pass, edge-clamped
    /// (matches a standard bitmap Gaussian blur's border handling; the
    /// Android side does the same via its own pure-Kotlin separable
    /// implementation per the doc).
    private static func separableBlur(_ src: [Float], size: Int, kernel: [Float]) -> [Float] {
        let radius = kernel.count / 2
        var temp = [Float](repeating: 0, count: size * size)
        for y in 0..<size {
            for x in 0..<size {
                var acc: Float = 0
                for k in -radius...radius {
                    let xs = max(0, min(size - 1, x + k))
                    acc += src[y * size + xs] * kernel[k + radius]
                }
                temp[y * size + x] = acc
            }
        }
        var out = [Float](repeating: 0, count: size * size)
        for y in 0..<size {
            for x in 0..<size {
                var acc: Float = 0
                for k in -radius...radius {
                    let ys = max(0, min(size - 1, y + k))
                    acc += temp[ys * size + x] * kernel[k + radius]
                }
                out[y * size + x] = acc
            }
        }
        return out
    }
}
