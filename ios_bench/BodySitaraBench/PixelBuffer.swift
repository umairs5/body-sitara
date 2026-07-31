import CoreGraphics
import Accelerate
import UIKit

/// Minimal RGB(A)/gray raw-pixel access shared by BackgroundReconstructor,
/// LightmapExtractor, and Compositor -- avoids re-deriving CGImage
/// byte-layout math (bytesPerRow/bytesPerPixel offsets) in three places.
/// Uses vImage (Accelerate) buffers so the per-pixel math in the alignment
/// pyramid and trimmed-mean stays vectorized rather than falling back to
/// pure Swift loops over millions of pixels -- same performance concern
/// the Python reference's trimmed_mean_vectorized() docstring already
/// flagged (a plain Python/Swift nested loop over N frames x H x W pixels
/// is not feasible; needs SIMD/vDSP-style batch ops).
struct RGBBuffer {
    var r: [Float]
    var g: [Float]
    var b: [Float]
    let width: Int
    let height: Int

    static func from(cgImage: CGImage) -> RGBBuffer {
        let w = cgImage.width
        let h = cgImage.height
        var rArr = [Float](repeating: 0, count: w * h)
        var gArr = [Float](repeating: 0, count: w * h)
        var bArr = [Float](repeating: 0, count: w * h)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var raw = [UInt8](repeating: 0, count: w * h * 4)
        guard let ctx = CGContext(
            data: &raw, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
            space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return RGBBuffer(r: rArr, g: gArr, b: bArr, width: w, height: h)
        }
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))

        for i in 0..<(w * h) {
            rArr[i] = Float(raw[i * 4])
            gArr[i] = Float(raw[i * 4 + 1])
            bArr[i] = Float(raw[i * 4 + 2])
        }
        return RGBBuffer(r: rArr, g: gArr, b: bArr, width: w, height: h)
    }

    /// Standard luma (matches cv2.COLOR_BGR2GRAY's coefficients), used only
    /// for the alignment pyramid's SAD search -- same role as the Python
    /// reference's cv2.cvtColor(..., BGR2GRAY) step.
    func grayscale() -> [Float] {
        var out = [Float](repeating: 0, count: width * height)
        for i in 0..<(width * height) {
            out[i] = 0.114 * b[i] + 0.587 * g[i] + 0.299 * r[i]
        }
        return out
    }

    func toCGImage() -> CGImage? {
        var raw = [UInt8](repeating: 255, count: width * height * 4)
        for i in 0..<(width * height) {
            raw[i * 4] = UInt8(max(0, min(255, r[i].rounded())))
            raw[i * 4 + 1] = UInt8(max(0, min(255, g[i].rounded())))
            raw[i * 4 + 2] = UInt8(max(0, min(255, b[i].rounded())))
        }
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let provider = CGDataProvider(data: Data(raw) as CFData) else { return nil }
        return CGImage(
            width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: width * 4, space: colorSpace,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
        )
    }
}

/// Binary person/hole mask, sourced from the Tier1 segmentation output
/// (mask.mp4 convention: white=person/hole, matching Android/LamaRunner's
/// existing convention). true = person (excluded from background
/// aggregation), matching the Python reference's bool_masks.
///
/// NOTE: this `[Bool]`-backed struct is now used only for SINGLE-frame
/// values (what `BackgroundReconstructor.compositeFrame`, `dilate`, etc.
/// actually operate on per call) -- the full N-frame clip-wide array uses
/// `PackedMaskFrame` instead (see below), precisely so a 300-frame clip
/// never holds N of these `[Bool]` arrays resident at once (Swift's
/// `[Bool]` is 1 byte/element in practice, not bit-packed, so N of these
/// at 1280x1280 would be ~492MB for a 300-frame clip -- the same class of
/// problem `RGBBuffer`/`RGBBuffer8` below addresses for color).
struct MaskBuffer {
    var isPerson: [Bool]
    let width: Int
    let height: Int

    static func from(cgImage: CGImage, threshold: UInt8 = 127) -> MaskBuffer {
        let w = cgImage.width
        let h = cgImage.height
        var raw = [UInt8](repeating: 0, count: w * h)
        let colorSpace = CGColorSpaceCreateDeviceGray()
        guard let ctx = CGContext(
            data: &raw, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w,
            space: colorSpace, bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else {
            return MaskBuffer(isPerson: [Bool](repeating: false, count: w * h), width: w, height: h)
        }
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))
        let mask = raw.map { $0 > threshold }
        return MaskBuffer(isPerson: mask, width: w, height: h)
    }
}

/// UInt8-native RGB buffer -- 1 byte/channel/pixel instead of `RGBBuffer`'s
/// 4 bytes/channel/pixel (Float32). Exists SOLELY to hold the full N-frame
/// clip-wide color array (`BenchmarkView.colorBuffers`, and
/// `BackgroundReconstructor.reconstruct(colorFrames:masks:)`'s parameter)
/// without ever materializing N full-resolution Float32 `RGBBuffer`s
/// simultaneously -- that was the actual root cause of the on-device
/// jetsam kill on a real 300-frame/1280x1280 clip (2026-07-31): the
/// earlier "memory fix" (see BackgroundReconstructor.swift's header doc)
/// only narrowed the INTERNAL aligned-frame copy `reconstructStatic`/
/// `reconstructDynamic` build from the input, but the INPUT itself --
/// `colorBuffers: [RGBBuffer]`, built by
/// `framesToConvert.map { RGBBuffer.from(cgImage: $0) }` in
/// BenchmarkView.swift -- was still N full Float32 RGBBuffers, which at
/// 300 frames * 1280*1280*4 bytes/channel*3 channels = ~5.9GB dwarfed the
/// ~1.5GB the internal fix achieved. `RGBBuffer8` is that same array's
/// replacement element type: same shape, 4x smaller per element, decoded
/// straight from the CGImage's raw 8-bit samples (no lossy step -- the
/// source pixels ARE already 8-bit, so this isn't a quality reduction,
/// just skipping the Float32 promotion until a single frame actually
/// needs it for gain/warp math).
///
/// Single-frame/single-plate values (the reconstructed background plate,
/// the lightmap, one composited output frame, the placeholder character)
/// stay on the existing Float32 `RGBBuffer` unchanged -- those are O(1)
/// per call, not O(N), so the 4x-larger footprint there is immaterial and
/// keeping them Float32 avoids touching Compositor.swift/
/// LightmapExtractor.swift's math (which already assumes Float32
/// precision for blending/relight).
///
/// MEMORY MATH for a 300-frame, 1280x1280 clip (the crash-reproducing
/// case), at THIS input-loading layer specifically -- the gap the prior
/// UInt8-stack fix (BackgroundReconstructor's AlignedFrameStack) missed:
///   Before (colorBuffers: [RGBBuffer], Float32):
///     300 * 1280*1280*4 bytes/channel*3 channels = ~5.90GB
///   After (colorBuffers: [RGBBuffer8], UInt8):
///     300 * 1280*1280*1 byte/channel*3 channels  = ~1.47GB
///   -> 4x reduction, matching the ratio Float32->UInt8 always gives per
///   element. This is now comparable to (not stacked on top of)
///   AlignedFrameStack's own ~1.44GB STATIC-path peak, since the two
///   arrays are never long-lived at their full size at the same time in
///   the same way the old design had two independent ~6GB-class
///   allocations coexisting (colorBuffers input + the old [[Float]]
///   aligned copy).
struct RGBBuffer8 {
    var r: [UInt8]
    var g: [UInt8]
    var b: [UInt8]
    let width: Int
    let height: Int

    /// Decodes directly to UInt8 -- same CGContext draw as `RGBBuffer.from`,
    /// just stores the raw drawn bytes as-is instead of promoting each
    /// sample to Float32 immediately. This is the exact byte value a
    /// Float32 `RGBBuffer.from(cgImage:)` would have produced before its
    /// `Float(raw[...])` cast, so there is no precision loss introduced by
    /// this type relative to the previous behavior -- Float32 promotion
    /// just happens later (per-frame, on demand) instead of eagerly for
    /// every frame up front.
    static func from(cgImage: CGImage) -> RGBBuffer8 {
        let w = cgImage.width
        let h = cgImage.height
        var rArr = [UInt8](repeating: 0, count: w * h)
        var gArr = [UInt8](repeating: 0, count: w * h)
        var bArr = [UInt8](repeating: 0, count: w * h)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var raw = [UInt8](repeating: 0, count: w * h * 4)
        guard let ctx = CGContext(
            data: &raw, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
            space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return RGBBuffer8(r: rArr, g: gArr, b: bArr, width: w, height: h)
        }
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))

        for i in 0..<(w * h) {
            rArr[i] = raw[i * 4]
            gArr[i] = raw[i * 4 + 1]
            bArr[i] = raw[i * 4 + 2]
        }
        return RGBBuffer8(r: rArr, g: gArr, b: bArr, width: w, height: h)
    }

    /// Promotes this ONE frame to a Float32 `RGBBuffer`, for the (many)
    /// call sites that need per-frame Float32 math (exposure gain,
    /// grayscale/alignment, compositing) but only ever need ONE frame's
    /// worth at a time -- the temporary `RGBBuffer` this returns is not
    /// retained by the caller past that single use, so this never
    /// reintroduces the O(N) Float32 residency the bug fix removes.
    func toFloatRGBBuffer() -> RGBBuffer {
        var rF = [Float](repeating: 0, count: width * height)
        var gF = [Float](repeating: 0, count: width * height)
        var bF = [Float](repeating: 0, count: width * height)
        for i in 0..<(width * height) {
            rF[i] = Float(r[i]); gF[i] = Float(g[i]); bF[i] = Float(b[i])
        }
        return RGBBuffer(r: rF, g: gF, b: bF, width: width, height: height)
    }
}

/// Bit-packed (1 bit/pixel) mask storage for the full N-frame clip-wide
/// mask array (`BenchmarkView.maskBuffers`) -- reuses the same
/// `BackgroundReconstructor.PackedBoolPlane` type already defined and
/// used internally for the per-frame "valid" plane in `AlignedFrameStack`,
/// rather than reinventing bit-packing here, per an 8x reduction vs.
/// `[MaskBuffer]`/`[Bool]` (Swift's `[Bool]` costs 1 byte/element in
/// practice): a 300-frame 1280x1280 clip needs
/// 300 * 1280*1280/8 bytes = ~60MB packed, vs. ~492MB as `[MaskBuffer]`.
struct PackedMaskFrame {
    let plane: BackgroundReconstructor.PackedBoolPlane
    let width: Int
    let height: Int

    static func from(cgImage: CGImage, threshold: UInt8 = 127) -> PackedMaskFrame {
        let mb = MaskBuffer.from(cgImage: cgImage, threshold: threshold)
        return PackedMaskFrame(plane: BackgroundReconstructor.PackedBoolPlane(mb.isPerson), width: mb.width, height: mb.height)
    }

    /// Unpacks this ONE frame back to a `MaskBuffer` (`[Bool]`), for call
    /// sites that need the existing `[Bool]`-based API (`dilate`,
    /// `compositeFrame`, per-pixel `.map` alpha construction) -- same
    /// single-frame-scope, non-retained-across-the-loop pattern as
    /// `RGBBuffer8.toFloatRGBBuffer()`.
    func unpacked() -> MaskBuffer {
        var out = [Bool](repeating: false, count: width * height)
        for i in 0..<(width * height) { out[i] = plane.get(i) }
        return MaskBuffer(isPerson: out, width: width, height: height)
    }
}
